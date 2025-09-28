import sys
import os
import json
import random
import uuid
import time
import numpy as np
from queue import Empty, Queue
from threading import Event, Thread
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
companies = {}

AI_MODEL_DIR = "ai_models"
SHARED_MODELS_PATH = os.path.join(AI_MODEL_DIR, "shared_models.json")
os.makedirs(AI_MODEL_DIR, exist_ok=True)

SHORT_MARGIN_RATE = 1.0

pg.setConfigOption("background", "#121212")
pg.setConfigOption("foreground", "#E6E6E6")
pg.setConfigOptions(antialias=True)


class _GUICallData:
    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.reply = None
        self.reply_event = Event()


class AITrader:
    ACTIONS = ["buy", "sell", "short", "hold"]

    def __init__(self, idx, money=1000.0, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.idx = idx
        self.money = money
        self.reserved = 0.0
        self.stocks = {}
        self.shorts = {}
        self.history = [money]            # overall value timeline
        self.money_history = [money]      # cash+reserved timeline
        self.reserved_history = [0.0]     # reserved (margin) timeline
        self.stocks_history = [0.0]       # market value of long stocks
        self.shorts_history = [0.0]       # market cost of shorts
        self.stock_position_history = [0.0]  # net shares (long - short) for selected company
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.model_path = os.path.join(AI_MODEL_DIR, f"ai_model_{idx}.json")
        self.load_model()

    def state_for(self, comp):
        hist = comp["history"]
        if len(hist) < 5:
            return "cold|1.00"
        window = hist[-5:]
        trend = 1 if window[-1] > window[0] * 1.001 else (-1 if window[-1] < window[0] * 0.999 else 0)
        ratio = round(window[-1] / (np.mean(window) + 1e-9), 2)
        return f"{trend}|{ratio:.2f}"

    def ensure_state(self, s):
        if s not in self.q_table:
            self.q_table[s] = {a: 0.0 for a in self.ACTIONS}

    def choose_action(self, state):
        self.ensure_state(state)
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)
        items = list(self.q_table[state].items())
        maxv = max(v for a, v in items)
        best = [a for a, v in items if v == maxv]
        return random.choice(best)

    def update_q(self, reward, new_state):
        if self.last_state is None or self.last_action is None:
            return
        self.ensure_state(self.last_state)
        self.ensure_state(new_state)
        old_q = self.q_table[self.last_state][self.last_action]
        max_future = max(self.q_table[new_state].values()) if self.q_table[new_state] else 0.0
        new_q = old_q + self.alpha * (reward + self.gamma * max_future - old_q)
        self.q_table[self.last_state][self.last_action] = new_q

    def save_model(self):
        try:
            with open(self.model_path, "w") as f:
                json.dump(self.q_table, f)
        except Exception as e:
            print("Failed saving AI model:", e)

    def load_model(self):
        try:
            if os.path.exists(SHARED_MODELS_PATH):
                with open(SHARED_MODELS_PATH, "r") as f:
                    shared = json.load(f)
                key = str(self.idx)
                if key in shared:
                    self.q_table = shared[key]
                    for s in list(self.q_table.keys()):
                        for a in self.ACTIONS:
                            self.q_table[s].setdefault(a, 0.0)
                    return
            if os.path.exists(self.model_path):
                with open(self.model_path, "r") as f:
                    self.q_table = json.load(f)
                for s in list(self.q_table.keys()):
                    for a in self.ACTIONS:
                        self.q_table[s].setdefault(a, 0.0)
        except Exception as e:
            print("Failed loading AI model:", e)
            self.q_table = {}


class StockMarketApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Market Simulator")
        self.resize(1400, 900)

        self.player = {"money": 1000.0, "reserved": 0.0, "stocks": {}, "history": [1000.0], "shorts": {}}
        self.player_stockpos_history = [0.0]
        self.player_stocks_history = [0.0]
        self.player_reserved_history = [0.0]
        self.player_overall_history = [self.player["money"] + self.player.get("reserved", 0.0)]

        self.ai_players = [AITrader(i) for i in range(3)]
        self.selected_company = None

        app = QtWidgets.QApplication.instance()
        if app:
            app.setStyle("Fusion")
            dark_palette = QtGui.QPalette()
            dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(18, 18, 18))
            dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(230, 230, 230))
            dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
            dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(24, 24, 24))
            dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(230, 230, 230))
            dark_palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(230, 230, 230))
            dark_palette.setColor(QtGui.QPalette.Text, QtGui.QColor(230, 230, 230))
            dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(40, 40, 40))
            dark_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(230, 230, 230))
            dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(60, 120, 180))
            dark_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
            app.setPalette(dark_palette)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(splitter)

        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        controls_box = QtWidgets.QGroupBox("Trade Controls")
        controls_layout = QtWidgets.QVBoxLayout(controls_box)
        controls_layout.setSpacing(8)

        self.text_input = QtWidgets.QLineEdit()
        self.text_input.setPlaceholderText("Search stock (or click suggestion)")
        self.text_input.textChanged.connect(self.update_autofill)
        controls_layout.addWidget(self.text_input)

        suggestions_layout = QtWidgets.QGridLayout()
        suggestions_layout.setSpacing(6)
        self.suggestion_buttons = []
        for i in range(10):
            btn = QtWidgets.QPushButton("")
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda checked, b=btn: self.show_chart(b.text()))
            self.suggestion_buttons.append(btn)
            r, c = divmod(i, 2)
            suggestions_layout.addWidget(btn, r, c)
        controls_layout.addLayout(suggestions_layout)

        amt_layout = QtWidgets.QHBoxLayout()
        self.amount_entry = QtWidgets.QLineEdit("0")
        validator = QtGui.QDoubleValidator(0.0, 1e12, 2)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.amount_entry.setValidator(validator)
        self.amount_entry.setFixedHeight(30)
        amt_layout.addWidget(QtWidgets.QLabel("Amount (€):"))
        amt_layout.addWidget(self.amount_entry)
        controls_layout.addLayout(amt_layout)

        trade_btn_layout = QtWidgets.QHBoxLayout()
        self.buy_btn = QtWidgets.QPushButton("Buy")
        self.sell_btn = QtWidgets.QPushButton("Sell")
        self.short_btn = QtWidgets.QPushButton("Short")
        for b in (self.buy_btn, self.sell_btn, self.short_btn):
            b.setFixedHeight(34)
        self.buy_btn.clicked.connect(self.player_buy_action)
        self.sell_btn.clicked.connect(self.player_sell_action)
        self.short_btn.clicked.connect(self.player_short_action)
        trade_btn_layout.addWidget(self.buy_btn)
        trade_btn_layout.addWidget(self.sell_btn)
        trade_btn_layout.addWidget(self.short_btn)
        controls_layout.addLayout(trade_btn_layout)

        left_layout.addWidget(controls_box)

        portfolio_box = QtWidgets.QGroupBox("Your Portfolio")
        portfolio_layout = QtWidgets.QVBoxLayout(portfolio_box)
        portfolio_layout.setSpacing(6)

        self.holdings_layout = QtWidgets.QVBoxLayout()
        self.holdings_layout.setSpacing(4)
        portfolio_layout.addLayout(self.holdings_layout)

        audit_layout = QtWidgets.QFormLayout()
        self.cash_label = QtWidgets.QLabel()
        self.reserved_label = QtWidgets.QLabel()
        audit_layout.addRow("Cash total:", self.cash_label)
        audit_layout.addRow("Reserved:", self.reserved_label)
        portfolio_layout.addLayout(audit_layout)

        left_layout.addWidget(portfolio_box)
        left_layout.addStretch(1)

        left_widget.setStyleSheet("""
            QGroupBox { font-weight: 600; border: 1px solid #2c2c2c; border-radius: 6px; padding: 8px; color: #E6E6E6;}
            QPushButton { background: #1f1f1f; border: 1px solid #333; border-radius: 4px; color: #E6E6E6; }
            QPushButton:pressed { background: #151515; }
            QLabel { color: #E6E6E6; }
            QLineEdit { background: #1e1e1e; color: #E6E6E6; border: 1px solid #333; padding: 4px; }
        """)

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(10)

        self.stock_plot = pg.PlotWidget(title="")
        self.stock_plot.setFixedHeight(220)
        self.stock_plot.showGrid(x=True, y=True, alpha=0.25)
        self.stock_curve = self.stock_plot.plot(pen=pg.mkPen('#FFD54F', width=2))
        right_layout.addWidget(self.stock_plot)

        self.player_plot = pg.PlotWidget(title="Player: Net pos (shares), Stocks value (€), Reserved (€), Overall (€)")
        self.player_plot.setFixedHeight(220)
        self.player_plot.showGrid(x=True, y=True, alpha=0.25)
        self.player_stockpos_curve = self.player_plot.plot(pen=pg.mkPen('#4CAF50', width=2))
        self.player_stocks_curve = self.player_plot.plot(pen=pg.mkPen('#2196F3', width=2))
        self.player_reserved_curve = self.player_plot.plot(pen=pg.mkPen('#FF9800', width=2, style=QtCore.Qt.DashLine))
        self.player_overall_curve = self.player_plot.plot(pen=pg.mkPen('#F44336', width=2))
        l = self.player_plot.addLegend(offset=(10, 10))
        try:
            l.setLabelTextColor('#E6E6E6')
        except Exception:
            pass
        l.addItem(self.player_stockpos_curve, "Net position (shares)")
        l.addItem(self.player_stocks_curve, "Stocks value (€)")
        l.addItem(self.player_reserved_curve, "Reserved (€)")
        l.addItem(self.player_overall_curve, "Overall value (€)")
        right_layout.addWidget(self.player_plot)

        ai_plots_widget = QtWidgets.QWidget()
        ai_layout = QtWidgets.QGridLayout(ai_plots_widget)
        ai_layout.setSpacing(6)
        self.ai_curves = []
        for i in range(3):
            pw = pg.PlotWidget(title=f"AI {i+1}")
            pw.setFixedHeight(140)
            pw.showGrid(x=True, y=True, alpha=0.18)
            stockpos_curve = pw.plot(pen=pg.mkPen('#4CAF50', width=2))
            stocks_curve = pw.plot(pen=pg.mkPen('#2196F3', width=2))
            reserved_curve = pw.plot(pen=pg.mkPen('#FF9800', width=2, style=QtCore.Qt.DashLine))
            overall_curve = pw.plot(pen=pg.mkPen('#F44336', width=2))
            legend = pw.addLegend(offset=(10, 10))
            try:
                legend.setLabelTextColor('#E6E6E6')
            except Exception:
                pass
            legend.addItem(stockpos_curve, "Net position")
            legend.addItem(stocks_curve, "Stocks value")
            legend.addItem(reserved_curve, "Reserved")
            legend.addItem(overall_curve, "Overall value")
            self.ai_curves.append((pw, {"stockpos": stockpos_curve, "stocks": stocks_curve, "reserved": reserved_curve, "overall": overall_curve}))
            ai_layout.addWidget(pw, i, 0)
        right_layout.addWidget(ai_plots_widget)

        right_widget.setStyleSheet("QWidget { font-family: Arial; color: #E6E6E6 }")

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([360, 1040])
        splitter.setStretchFactor(1, 1)

        self.define_stock_market(50)
        self.update_autofill()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_stock_prices)
        self.timer.start(1000)

        self.save_timer = QtCore.QTimer()
        self.save_timer.timeout.connect(self.save_ai_models)
        self.save_timer.start(15000)

        self.call_queue = Queue()
        self.thread = Thread(target=self.threadFn, daemon=True)
        self.thread.start()

        QtWidgets.QApplication.instance().aboutToQuit.connect(self.save_ai_models)

    def make_gui_call(self, fn, *args, **kwargs):
        data = _GUICallData(fn, args, kwargs)
        self.call_queue.put(data)
        QtCore.QTimer.singleShot(0, self.gui_call_handler)
        data.reply_event.wait()
        return data.reply

    def gui_call_handler(self):
        try:
            while True:
                data = self.call_queue.get_nowait()
                try:
                    data.reply = data.fn(*data.args, **data.kwargs)
                except Exception as e:
                    data.reply = e
                finally:
                    data.reply_event.set()
        except Empty:
            pass

    def threadFn(self):
        result = self.make_gui_call(self.get_entry_text)
        print("Thread read entry text (via make_gui_call):", result)

    def get_entry_text(self):
        return self.text_input.text()

    def set_entry_text(self, text):
        self.text_input.setText(text)

    def generate_stock_price_series(self, start_price, mu=0.0005, sigma=0.02, days=365):
        shocks = np.random.normal(0, 1, days)
        prices = start_price * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) + sigma * shocks))
        return np.maximum(prices, 1.0).tolist()

    def define_stock_market(self, num_companies):
        while len(companies) < num_companies:
            name = ''.join(random.choices(characters, k=3))
            if name not in companies:
                start_price = random.uniform(50, 200)
                history = self.generate_stock_price_series(start_price)
                companies[name] = {"name": name, "stock_price": history[-1], "history": history}

    def update_autofill(self):
        user_input = self.text_input.text().strip().upper()
        matches = [n for n in companies if n.startswith(user_input)]
        items = matches[:10] if matches else list(companies.keys())[:10]
        for btn, name in zip(self.suggestion_buttons, items):
            btn.setText(name)
        for btn in self.suggestion_buttons[len(items):]:
            btn.setText("")

    def show_chart(self, name):
        if name:
            self.selected_company = name
            self.update_charts()

    def get_amount(self):
        try:
            val = float(self.amount_entry.text())
        except Exception:
            val = 0.0
        return max(0.0, val)

    def _get_field(self, entity, field):
        if isinstance(entity, AITrader):
            return getattr(entity, field)
        return entity[field]

    def _set_field(self, entity, field, value):
        if isinstance(entity, AITrader):
            setattr(entity, field, value)
        else:
            entity[field] = value

    def entity_stocks_value(self, entity):
        stocks = self._get_field(entity, "stocks")
        return sum(qty * companies[n]["stock_price"] for n, qty in stocks.items())

    def entity_shorts_cost(self, entity):
        shorts = self._get_field(entity, "shorts")
        total_cost = 0.0
        for name, positions in shorts.items():
            for pos in positions:
                total_cost += pos.get("qty", 0.0) * companies[name]["stock_price"]
        return total_cost

    def entity_cash_total(self, entity):
        money = self._get_field(entity, "money")
        reserved = self._get_field(entity, "reserved") if "reserved" in (entity if isinstance(entity, dict) else vars(entity)) else 0.0
        return money + reserved

    def entity_net_position(self, entity, company_name):
        long_qty = self._get_field(entity, "stocks").get(company_name, 0.0)
        short_qty = 0.0
        for pos in self._get_field(entity, "shorts").get(company_name, []):
            short_qty += pos.get("qty", 0.0)
        return long_qty - short_qty

    def trade(self, player_obj, company_name, action, amount_eur):
        if company_name not in companies or amount_eur <= 0.0:
            return
        price = companies[company_name]["stock_price"]

        money = self._get_field(player_obj, "money")
        reserved = self._get_field(player_obj, "reserved") if "reserved" in (player_obj if isinstance(player_obj, dict) else vars(player_obj)) else 0.0
        stocks = self._get_field(player_obj, "stocks")
        shorts = self._get_field(player_obj, "shorts")

        if action == "sell":
            owned = stocks.get(company_name, 0.0)
            sell_qty = min(owned, amount_eur / price)
            if sell_qty > 0:
                money += sell_qty * price
                new_owned = owned - sell_qty
                if new_owned <= 1e-12:
                    stocks.pop(company_name, None)
                else:
                    stocks[company_name] = new_owned
            self._set_field(player_obj, "money", money)
            try:
                self._set_field(player_obj, "reserved", reserved)
            except Exception:
                pass
            self._set_field(player_obj, "stocks", stocks)
            self._set_field(player_obj, "shorts", shorts)
            return

        max_allowed_proceeds = money / SHORT_MARGIN_RATE if SHORT_MARGIN_RATE > 0 else amount_eur
        use_amount = min(amount_eur, max_allowed_proceeds)
        if use_amount <= 0:
            self._set_field(player_obj, "money", money)
            self._set_field(player_obj, "reserved", reserved)
            self._set_field(player_obj, "stocks", stocks)
            self._set_field(player_obj, "shorts", shorts)
            return

        qty = use_amount / price

        if action == "buy":
            buy_amt = min(use_amount, money)
            buy_qty = buy_amt / price
            money -= buy_amt
            stocks[company_name] = stocks.get(company_name, 0.0) + buy_qty

        elif action == "short":
            proceeds = use_amount
            margin_required = proceeds * SHORT_MARGIN_RATE
            if money + 1e-12 < margin_required:
                proceeds = money / SHORT_MARGIN_RATE if SHORT_MARGIN_RATE > 0 else 0.0
                qty = proceeds / price
                margin_required = proceeds * SHORT_MARGIN_RATE
            if proceeds <= 0:
                self._set_field(player_obj, "money", money)
                self._set_field(player_obj, "reserved", reserved)
                return
            money += proceeds
            money -= margin_required
            reserved += margin_required
            pos_id = str(uuid.uuid4())
            shorts.setdefault(company_name, []).append({
                "id": pos_id,
                "qty": qty,
                "entry_price": price,
                "duration_days": 10,
                "start_time": time.time(),
                "proceeds": proceeds,
                "reserved": margin_required
            })

        self._set_field(player_obj, "money", money)
        try:
            self._set_field(player_obj, "reserved", reserved)
        except Exception:
            pass
        self._set_field(player_obj, "stocks", stocks)
        self._set_field(player_obj, "shorts", shorts)

    def check_shorts_expiry(self):
        now = time.time()
        self._check_entity_shorts(self.player, now)
        for ai in self.ai_players:
            self._check_entity_shorts(ai, now)

    def _check_entity_shorts(self, entity, now):
        shorts = self._get_field(entity, "shorts")
        money = self._get_field(entity, "money")
        reserved = self._get_field(entity, "reserved") if "reserved" in (entity if isinstance(entity, dict) else vars(entity)) else 0.0
        updated_shorts = {}
        for name, positions in list(shorts.items()):
            still_open = []
            for pos in positions:
                elapsed_days = (now - pos["start_time"]) / 1.0
                if elapsed_days >= pos["duration_days"]:
                    current_price = companies[name]["stock_price"]
                    buy_cost = pos["qty"] * current_price
                    proceeds = pos.get("proceeds", 0.0)
                    reserved_amt = pos.get("reserved", 0.0)
                    realized_pnl = proceeds - buy_cost
                    money -= buy_cost
                    money += reserved_amt
                    reserved -= reserved_amt
                    owner = "Player" if entity is self.player else f"AI{entity.idx+1}"
                    print(f"{owner} closed short on {name}: P/L {realized_pnl:.2f} (proceeds {proceeds:.2f}, buy_cost {buy_cost:.2f})")
                else:
                    still_open.append(pos)
            if still_open:
                updated_shorts[name] = still_open
        self._set_field(entity, "shorts", updated_shorts)
        self._set_field(entity, "money", money)
        try:
            self._set_field(entity, "reserved", reserved)
        except Exception:
            pass

    def calc_portfolio_value(self, entity):
        money = self._get_field(entity, "money")
        reserved = self._get_field(entity, "reserved") if "reserved" in (entity if isinstance(entity, dict) else vars(entity)) else 0.0
        stocks = self._get_field(entity, "stocks")
        shorts = self._get_field(entity, "shorts")
        value = money + reserved
        value += sum(stocks.get(n, 0.0) * companies[n]["stock_price"] for n in stocks)
        value += sum((pos["entry_price"] - companies[name]["stock_price"]) * pos["qty"]
                     for name, positions in shorts.items() for pos in positions)
        return value

    def ai_trade(self):
        for ai in self.ai_players:
            for name, comp in companies.items():
                state = ai.state_for(comp)
                action = ai.choose_action(state)
                prev_val = self.calc_portfolio_value(ai)

                if action == "buy":
                    amt = ai.money * 0.10
                    self.trade(ai, name, "buy", amt)
                elif action == "sell":
                    owned_qty = ai.stocks.get(name, 0.0)
                    if owned_qty > 1e-9:
                        amt = owned_qty * comp["stock_price"] * 0.5
                        self.trade(ai, name, "sell", amt)
                elif action == "short":
                    amt = ai.money * 0.10
                    self.trade(ai, name, "short", amt)

                new_val = self.calc_portfolio_value(ai)
                reward = new_val - prev_val
                ai.update_q(reward, state)
                ai.last_state = state
                ai.last_action = action

            ai_total = self.calc_portfolio_value(ai)
            ai_stocks_val = self.entity_stocks_value(ai)
            ai_reserved = ai.reserved
            ai.history.append(ai_total)
            ai.stocks_history.append(ai_stocks_val)
            ai.shorts_history.append(self.entity_shorts_cost(ai))
            ai.money_history.append(self.entity_cash_total(ai))
            ai.reserved_history.append(ai_reserved)
            net_pos = self.entity_net_position(ai, self.selected_company) if self.selected_company else 0.0
            ai.stock_position_history.append(net_pos)

    def update_stock_prices(self):
        mu, sigma = 0.0005, 0.02
        for comp in companies.values():
            shock = np.random.normal()
            comp['stock_price'] *= np.exp(mu - 0.5 * sigma ** 2 + sigma * shock)
            comp['history'].append(comp['stock_price'])
            if len(comp['history']) > 365:
                comp['history'].pop(0)

        self.check_shorts_expiry()
        self.ai_trade()
        self.update_charts()

    def save_ai_models(self):
        combined = {}
        for ai in self.ai_players:
            try:
                ai.save_model()
                combined[str(ai.idx)] = ai.q_table
            except Exception as e:
                print("Error saving AI model:", e)
        try:
            with open(SHARED_MODELS_PATH, "w") as f:
                json.dump(combined, f)
        except Exception as e:
            print("Error saving shared AI models:", e)

    def player_buy_action(self):
        if self.selected_company:
            amt = self.get_amount()
            self.trade(self.player, self.selected_company, "buy", amt)
            self.update_charts()

    def player_sell_action(self):
        if self.selected_company:
            amt = self.get_amount()
            self.trade(self.player, self.selected_company, "sell", amt)
            self.update_charts()

    def player_short_action(self):
        if self.selected_company:
            amt = self.get_amount()
            self.trade(self.player, self.selected_company, "short", amt)
            self.update_charts()

    def update_charts(self):
        if self.selected_company:
            data = companies[self.selected_company]["history"]
            self.stock_curve.setData(data)
            self.stock_plot.setTitle(f"Stock: {self.selected_company}  (price €{companies[self.selected_company]['stock_price']:.2f})")
        else:
            self.stock_plot.setTitle("")

        player_overall = self.calc_portfolio_value(self.player)
        self.player["history"].append(player_overall)

        p_net_pos = self.entity_net_position(self.player, self.selected_company) if self.selected_company else 0.0
        p_stocks_val = self.entity_stocks_value(self.player)
        p_reserved = self.player.get("reserved", 0.0)
        p_overall = player_overall

        self.player_stockpos_history.append(p_net_pos)
        self.player_stocks_history.append(p_stocks_val)
        self.player_reserved_history.append(p_reserved)
        self.player_overall_history.append(p_overall)

        self.player_plot.setTitle("Player: Net pos (shares), Stocks value (€), Reserved (€), Overall (€)")
        self.player_stockpos_curve.setData(self.player_stockpos_history)
        self.player_stocks_curve.setData(self.player_stocks_history)
        self.player_reserved_curve.setData(self.player_reserved_history)
        self.player_overall_curve.setData(self.player_overall_history)

        for i, (pw, curves) in enumerate(self.ai_curves):
            ai = self.ai_players[i]
            curves["stockpos"].setData(ai.stock_position_history)
            curves["stocks"].setData(ai.stocks_history)
            curves["reserved"].setData(ai.reserved_history)
            curves["overall"].setData(ai.history)

        while self.holdings_layout.count():
            item = self.holdings_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        for name, qty in self.player["stocks"].items():
            price = companies[name]["stock_price"]
            value = qty * price
            btn = QtWidgets.QPushButton(f"{name}  ·  Shares: {qty:.4f}  ·  Price: €{price:.2f}  ·  Value: €{value:.2f}")
            btn.setStyleSheet("text-align:left; padding:6px; color:#E6E6E6; background:#222; border:1px solid #333;")
            btn.clicked.connect(lambda checked, n=name: self.show_chart(n))
            self.holdings_layout.addWidget(btn)

        for name, positions in self.player["shorts"].items():
            for pos in positions:
                current_price = companies[name]["stock_price"]
                pnl = (pos['entry_price'] - current_price) * pos['qty']
                reserved = pos.get("reserved", 0.0)
                btn = QtWidgets.QPushButton(
                    f"{name} SHORT  ·  Qty: {pos['qty']:.4f}  ·  Entry: €{pos['entry_price']:.2f}  ·  Current: €{current_price:.2f}  ·  P/L: €{pnl:.2f}  ·  Reserved: €{reserved:.2f}"
                )
                btn.setStyleSheet("color: #F88A8A; text-align:left; padding:6px; background:#220000; border:1px solid #440000;")
                btn.clicked.connect(lambda checked, n=name: self.show_chart(n))
                self.holdings_layout.addWidget(btn)

        cash_total = self.player["money"]
        reserved_total = self.player.get("reserved", 0.0)
        self.cash_label.setText(f"€{cash_total:,.2f}")
        self.reserved_label.setText(f"€{reserved_total:,.2f}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = StockMarketApp()
    window.show()
    sys.exit(app.exec_())
