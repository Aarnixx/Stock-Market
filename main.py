import random
import uuid
import time
import numpy as np
from threading import Timer
from customtkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
companies = {}

root = CTk()
root.geometry("1920x1080")
root.title("Stock market")
left_panel = CTkFrame(root)
left_panel.pack(side="left", fill="y", padx=5, pady=5)

overview_frame = CTkFrame(left_panel)
overview_frame.pack(fill="x", padx=5, pady=5)

text_input = CTkTextbox(left_panel, height=25)
text_input.pack(fill="x", padx=10, pady=5)

suggestion_frame = CTkFrame(left_panel)
suggestion_frame.pack(fill="both", expand=True, padx=5, pady=5)

MAX_SUGGESTIONS = 10
suggestion_buttons = [CTkButton(suggestion_frame, text="") for _ in range(MAX_SUGGESTIONS)]
for idx, btn in enumerate(suggestion_buttons):
    row = idx // 2
    col = idx % 2
    btn.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
for c in range(2):
    suggestion_frame.grid_columnconfigure(c, weight=1)
for r in range(5):
    suggestion_frame.grid_rowconfigure(r, weight=1)

btn_frame_buy_sell = CTkFrame(left_panel)
btn_frame_buy_sell.pack(fill="x", pady=10)

btn_1 = CTkButton(btn_frame_buy_sell, text="Buy")
btn_1.pack(side="left", expand=True, padx=10)

btn_2 = CTkButton(btn_frame_buy_sell, text="Sell")
btn_2.pack(side="right", expand=True, padx=10)

btn_3 = CTkButton(btn_frame_buy_sell, text="Short")
btn_3.pack(side="bottom", expand=True, padx=10)

stock_labels = []
selected_company = None

right_panel = CTkFrame(root)
right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

fig_stock, ax_stock = plt.subplots(figsize=(6, 2.0))
fig_stock.set_constrained_layout(True)
canvas_stock = FigureCanvasTkAgg(fig_stock, master=right_panel)
canvas_stock_widget = canvas_stock.get_tk_widget()
canvas_stock_widget.pack(fill="x", padx=5, pady=5)
line_stock, = ax_stock.plot([], linewidth=1)
canvas_stock.draw()

ai_figs, ai_axes, ai_canvases, ai_lines = [], [], [], []
ai_money_labels, ai_profit_labels = [], []

for i in range(3):
    f, a = plt.subplots(figsize=(6, 1.0))
    f.set_constrained_layout(True)
    c = FigureCanvasTkAgg(f, master=right_panel)
    w = c.get_tk_widget()
    w.pack(fill="x", padx=5, pady=5)
    line, = a.plot([], linewidth=1)
    ai_figs.append(f)
    ai_axes.append(a)
    ai_canvases.append(c)
    ai_lines.append(line)
    c.draw()
    lbl_frame = CTkFrame(right_panel)
    lbl_frame.pack(fill="x", padx=10, pady=2)
    money_lbl = CTkLabel(lbl_frame, text=f"AI{i+1} Money: €0.00", anchor="w")
    profit_lbl = CTkLabel(lbl_frame, text=f"AI{i+1} P/L: €0.00", anchor="w")
    money_lbl.pack(side="left", padx=5)
    profit_lbl.pack(side="right", padx=5)
    ai_money_labels.append(money_lbl)
    ai_profit_labels.append(profit_lbl)

player_money_fig, player_money_ax = plt.subplots(figsize=(6, 1.5))
player_money_fig.set_constrained_layout(True)
player_money_canvas = FigureCanvasTkAgg(player_money_fig, master=right_panel)
player_money_canvas_widget = player_money_canvas.get_tk_widget()
player_money_canvas_widget.pack(fill="x", padx=5, pady=5)
line_player, = player_money_ax.plot([], linewidth=1)
player_money_canvas.draw()

player_lbl_frame = CTkFrame(right_panel)
player_lbl_frame.pack(fill="x", padx=10, pady=2)
player_money_label = CTkLabel(player_lbl_frame, text=f"Player Money: €{0:,.2f}", anchor="w")
player_profit_label = CTkLabel(player_lbl_frame, text=f"Player P/L: €{0:,.2f}", anchor="w")
player_money_label.pack(side="left", padx=5)
player_profit_label.pack(side="right", padx=5)

owned_frame = CTkFrame(right_panel)
owned_frame.pack(fill="both", expand=True, padx=10, pady=(10, 5))

owned_label = CTkLabel(owned_frame, text="Your Owned Stocks and Short positions", font=("Arial", 14, "bold"))
owned_label.pack(anchor="w")

owned_stocks_text = CTkTextbox(owned_frame, height=150)
owned_stocks_text.pack(fill="both", expand=True, pady=5)

player = {"money": 1000.0, "stocks": {}, "history": [1000.0], "shorts": {}}
ai_players = [{"money": 1000.0, "stocks": {}, "history": [1000.0], "shorts": {}} for _ in range(3)]

def update_owned_display():
    to_close = []
    lines = ["Owned Stocks:"]
    if player["stocks"]:
        for stock, qty in player["stocks"].items():
            price = companies[stock]["stock_price"]
            total_value = qty * price
            lines.append(f"  {stock}: {qty} shares (€{total_value:,.2f})")
    else:
        lines.append("  None")
    lines.append("\nShort Positions:")
    has_shorts = False
    for stock, positions in list(player["shorts"].items()):
        for pos in list(positions):
            elapsed_days = (time.time() - pos["start_time"]) / (24 * 3600)
            remaining_days = pos["duration_days"] - elapsed_days
            if remaining_days <= 0.0:
                to_close.append((pos["id"], stock))
            else:
                has_shorts = True
                price = companies[stock]["stock_price"]
                lines.append(
                    f"  {stock} | id:{pos['id'][:8]} qty:{pos['qty']} "
                    f"short_price:€{pos['short_price']:,.2f} current_price:€{price:,.2f} "
                    f"remaining_days:{remaining_days:.2f}"
                )
    for pos_id, stock in to_close:
        timeout(pos_id, stock, player, update_ui=False)
    if to_close:
        return update_owned_display()
    if not has_shorts:
        lines.append("  None")
    text = "\n".join(lines)
    owned_stocks_text.configure(state="normal")
    owned_stocks_text.delete("1.0", "end")
    owned_stocks_text.insert("1.0", text)
    owned_stocks_text.configure(state="disabled")

def generate_stock_price_series(start_price, mu=0.0005, sigma=0.02, days=365):
    shocks = np.random.normal(0, 1, days)
    prices = start_price * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) + sigma * shocks))
    return np.maximum(prices, 1.0).tolist()

def define_stock_market(num_companies):
    while len(companies) < num_companies:
        name = ''.join(random.choices(characters, k=3))
        if name not in companies:
            start_price = random.uniform(50, 200)
            history = generate_stock_price_series(start_price)
            companies[name] = {
                "name": name,
                "stock_amount": random.randint(1000, 10000),
                "stock_price": history[-1],
                "history": history
            }

def timeout(position_id, company_name, player_obj, update_ui=True):
    positions = player_obj["shorts"].get(company_name, [])
    pos = next((p for p in positions if p["id"] == position_id), None)
    if not pos:
        return
    current_price = companies[company_name]["stock_price"]
    qty = pos["qty"]
    hist = companies[company_name]["history"]
    slippage = 0.0
    if len(hist) >= 6:
        recent = np.array(hist[-6:])
        returns = np.diff(np.log(recent + 1e-9))
        vol = float(np.std(returns[-5:]))
        slippage = min(0.20, vol * 3.0)
    if pos.get("duration_days", 0.0) < 1.0:
        slippage += 0.05
    buyback_price = current_price * (1.0 + slippage)
    cost = buyback_price * qty
    player_obj["money"] -= cost
    positions.remove(pos)
    if not positions:
        del player_obj["shorts"][company_name]
    if update_ui:
        update_stock_display()

def set_timer_days(duration_days, position_id, company_name, player_obj):
    seconds = max(0.0, float(duration_days)) * 24.0 * 3600.0
    t = Timer(seconds, timeout, args=(position_id, company_name, player_obj))
    t.daemon = True
    t.start()

def trade(player_obj, company_name, action, time_of_closing: float, qty=1):
    if company_name not in companies or qty <= 0:
        return 0
    price = companies[company_name]["stock_price"]
    if action == "buy":
        max_qty = int(player_obj["money"] // price)
        qty = min(qty, max_qty)
        if qty <= 0:
            return 0
        player_obj["money"] -= qty * price
        player_obj["stocks"][company_name] = player_obj["stocks"].get(company_name, 0) + qty
        return qty
    elif action == "short":
        max_qty = int(player_obj["money"] // price)
        if max_qty == 0:
            return 0
        qty = min(qty, max_qty)
        if qty <= 0:
            return 0
        player_obj["money"] += qty * price
        pos_id = str(uuid.uuid4())
        position = {
            "id": pos_id,
            "qty": qty,
            "short_price": price,
            "duration_days": float(time_of_closing),
            "start_time": time.time()
        }
        player_obj["shorts"].setdefault(company_name, []).append(position)
        set_timer_days(time_of_closing, pos_id, company_name, player_obj)
        return qty
    elif action == "sell":
        owned = player_obj["stocks"].get(company_name, 0)
        qty = min(qty, owned)
        if qty <= 0:
            return 0
        hist = companies[company_name]["history"]
        if len(hist) >= 6:
            recent = np.array(hist[-6:])
            returns = np.diff(np.log(recent + 1e-9))
            vol = float(np.std(returns[-5:]))
            slippage = min(0.20, vol * 2.5)
        else:
            slippage = 0.0
        market_amount = companies[company_name].get("stock_amount", 1000)
        pressure = qty / max(1.0, market_amount)
        pressure_slippage = min(0.10, pressure * 2.0)
        total_slippage = min(0.35, slippage + pressure_slippage)
        realized_price = price * (1.0 - total_slippage)
        revenue = qty * realized_price
        player_obj["money"] += revenue
        player_obj["stocks"][company_name] = owned - qty
        if player_obj["stocks"][company_name] == 0:
            del player_obj["stocks"][company_name]
        return qty
    return 0

def ai_trade():
    for ai in ai_players:
        for name, comp in companies.items():
            hist = comp["history"]
            if len(hist) < 10:
                continue
            mean_price = np.mean(hist[-10:])
            current_price = comp["stock_price"]
            rnd = random.random()
            if current_price < mean_price * 0.98 and rnd > 0.25:
                trade(ai, name, "buy", 1)
            elif current_price > mean_price * 1.02 and rnd > 0.25:
                trade(ai, name, "sell", 1)
        portfolio_value = ai["money"] + sum(
            ai["stocks"].get(n, 0) * companies[n]["stock_price"] for n in ai["stocks"]
        )
        ai["history"].append(portfolio_value)

def update_stock_prices():
    mu, sigma = 0.0005, 0.02
    prices = np.array([c['stock_price'] for c in companies.values()])
    shocks = np.random.normal(size=prices.shape)
    prices = prices * np.exp((mu - 0.5 * sigma ** 2) + sigma * shocks)
    for c, p in zip(companies.values(), prices):
        c['stock_price'] = p
        c['history'].append(p)
        if len(c['history']) > 365:
            c['history'].pop(0)
    player_value = player["money"] + sum(
        player["stocks"].get(n, 0) * companies[n]["stock_price"] for n in player["stocks"]
    )
    player["history"].append(player_value)
    ai_trade()
    update_stock_display()
    root.after(1000, update_stock_prices)

center_frame = CTkFrame(left_panel)
center_frame.pack(fill="x", pady=10)

amount_entry = CTkEntry(center_frame, placeholder_text="Amount")
amount_entry.pack(fill="x", padx=10, pady=5)

def set_amount(n):
    try:
        n_int = int(float(n))
    except Exception:
        n_int = 0
    n_int = max(0, n_int)
    amount_entry.delete(0, "end")
    amount_entry.insert(0, str(n_int))

def compute_buy_max_for_selected():
    if not selected_company:
        return 0
    price = companies[selected_company]["stock_price"]
    return int(player["money"] // price) if price > 0 else 0

def compute_sell_max_for_selected():
    if not selected_company:
        return 0
    return player["stocks"].get(selected_company, 0)

def get_amount_from_entry():
    try:
        val = int(float(amount_entry.get()))
    except Exception:
        val = 0
    return max(0, val)

def change_amount_by(delta):
    amt = get_amount_from_entry()
    new = amt + int(delta)
    if selected_company:
        buy_max = compute_buy_max_for_selected()
        sell_max = compute_sell_max_for_selected()
        cap = buy_max if buy_max > 0 else sell_max
        new = min(new, cap)
    else:
        cash_cap = int(player["money"])
        new = min(new, cash_cap)
    new = max(0, new)
    set_amount(new)

def on_btn_min():
    set_amount(0)

def on_btn_minus100():
    change_amount_by(-100)

def on_btn_plus100():
    change_amount_by(100)

def on_btn_max():
    if not selected_company:
        set_amount(int(player["money"]))
        return
    comp = companies.get(selected_company)
    if not comp:
        set_amount(0)
        return
    price = comp["stock_price"]
    if price <= 0:
        set_amount(0)
        return
    buy_max = compute_buy_max_for_selected()
    sell_max = compute_sell_max_for_selected()
    if buy_max > 0:
        set_amount(buy_max)
    else:
        set_amount(sell_max)

btn_frame = CTkFrame(center_frame)
btn_frame.pack(fill="x", padx=5, pady=5)

labels = ["min", "-100", "+100", "max"]
commands = [on_btn_min, on_btn_minus100, on_btn_plus100, on_btn_max]
for i, lab in enumerate(labels):
    b = CTkButton(btn_frame, text=lab, command=commands[i])
    b.grid(row=0, column=i, sticky="nsew", padx=3, pady=3)
    btn_frame.grid_columnconfigure(i, weight=1)

def player_buy_action():
    if not selected_company:
        return
    qty = get_amount_from_entry()
    traded = trade(player, selected_company, "buy", 0, qty)
    if traded > 0:
        update_stock_display()

def player_sell_action():
    if not selected_company:
        return
    qty = get_amount_from_entry()
    traded = trade(player, selected_company, "sell", 0, qty)
    if traded > 0:
        update_stock_display()

def player_short_action():
    if not selected_company:
        return
    qty = get_amount_from_entry()
    shorted = trade(player, selected_company, "short", 10, qty)
    if shorted > 0:
        update_stock_display()

btn_1.configure(command=player_buy_action)
btn_2.configure(command=player_sell_action)
btn_3.configure(command=player_short_action)

def update_stock_display():
    if selected_company:
        comp = companies[selected_company]
        line_stock.set_data(range(len(comp["history"])), comp["history"])
        ax_stock.relim()
        ax_stock.autoscale_view()
        ax_stock.set_title(f"Stock: {comp['name']}", fontsize=8)
        ax_stock.tick_params(labelsize=6)
        try:
            fig_stock.tight_layout()
        except Exception:
            pass
        canvas_stock.draw_idle()
    else:
        line_stock.set_data([], [])
        canvas_stock.draw_idle()
    line_player.set_data(range(len(player["history"])), player["history"])
    player_money_ax.relim()
    player_money_ax.autoscale_view()
    player_money_ax.set_title("Player Portfolio (€)", fontsize=8)
    player_money_ax.tick_params(labelsize=6)
    try:
        player_money_fig.tight_layout()
    except Exception:
        pass
    player_money_canvas.draw_idle()
    for i, ai in enumerate(ai_players):
        if i < len(ai_lines):
            ai_lines[i].set_data(range(len(ai["history"])), ai["history"])
            ai_axes[i].relim()
            ai_axes[i].autoscale_view()
            ai_axes[i].set_title(f"AI{i+1}", fontsize=6)
            ai_axes[i].tick_params(labelsize=6)
            try:
                ai_figs[i].tight_layout()
            except Exception:
                pass
            ai_canvases[i].draw_idle()
        current_money = ai["money"]
        current_value = ai["history"][-1] if ai["history"] else current_money
        profit = current_value - 1000.0
        if i < len(ai_money_labels):
            ai_money_labels[i].configure(text=f"AI{i+1} Money: €{current_money:,.2f}")
            ai_profit_labels[i].configure(text=f"AI{i+1} P/L: €{profit:,.2f}")

    line_player.set_data(range(len(player["history"])), player["history"])
    player_money_ax.relim()
    player_money_ax.autoscale_view()
    player_money_ax.set_title("Player Portfolio (€)", fontsize=8)
    player_money_ax.tick_params(labelsize=6)
    try:
        player_money_fig.tight_layout()
    except Exception:
        pass
    player_money_canvas.draw_idle()

    player_current_money = player["money"]
    player_current_value = player["history"][-1] if player["history"] else player_current_money
    player_profit = player_current_value - 1000.0
    player_money_label.configure(text=f"Player Money: €{player_current_money:,.2f}")
    player_profit_label.configure(text=f"Player P/L: €{player_profit:,.2f}")

    update_overview()
    update_owned_display()
    amt = get_amount_from_entry()
    set_amount(amt)


def show_chart(name):
    global selected_company
    selected_company = name
    on_btn_max()
    update_stock_display()


def show_overview():
    global selected_company
    selected_company = None
    update_stock_display()


def update_overview():
    if not stock_labels:
        top10 = list(companies.keys())[:10]
        columns = 2
        for idx, name in enumerate(top10):
            lbl = CTkButton(
                overview_frame,
                text=name,
                command=lambda n=name: show_chart(n)
            )
            row = idx // columns
            col = idx % columns
            lbl.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            stock_labels.append(lbl)
        for c in range(columns):
            overview_frame.grid_columnconfigure(c, weight=1)

    for lbl in stock_labels:
        name = lbl.cget("text").split()[0]
        comp = companies[name]
        if len(comp["history"]) >= 2:
            arrow = "▲" if comp["history"][-1] >= comp["history"][-2] else "▼"
        else:
            arrow = ""
        color = "green" if arrow == "▲" else "red"
        lbl.configure(text=f"{name} {arrow}", fg_color=color)


def on_suggestion_click(name):
    show_chart(name)


def update_autofill(event=None):
    user_input = text_input.get("1.0", "end-1c").strip().upper()
    if user_input:
        matches = [n for n in companies if n.startswith(user_input)]
        items = matches[:MAX_SUGGESTIONS]
    else:
        items = list(companies.keys())[:MAX_SUGGESTIONS]

    for i, btn in enumerate(suggestion_buttons):
        if i < len(items):
            name = items[i]
            btn.configure(text=name, command=lambda n=name: on_suggestion_click(n))
        else:
            btn.configure(text="", command=lambda: None)


text_input.bind("<KeyRelease>", update_autofill)


if __name__ == "__main__":
    define_stock_market(300)
    update_overview()
    update_autofill()
    update_stock_display()
    update_stock_prices()
    root.mainloop()
