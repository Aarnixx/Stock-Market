Stock Market 

Tässä Python-projektissa simuloidaan fiktionaalisia osakemarkkinoita reaaliajassa.  
Pelaaja voi ostaa, myydä ja lyhyeksi myydä osakkeita ja seurata salkkunsa arvoa live-kaavioissa. 

OMINAISUUDET 
- Reaaliaikaiset osakekurssit satunnaisprosesseilla. 
- 3 tekoäly pelaajaa, jotka ostavat, myyvät ja myyvät lyhyeksi osakkeita. 
- lyhyeksi myynnit sulkeutuvat automaattisesti kymmenen pelin sisäisen päivän jälkeen. 
- Live-kaaviot kaikille pelaajille ja yksittäisille osakkeille 
- Omistusten ja P/L:n seuranta 

Tekoäly Pelaajat
-Tekoäly pelaajat arvioivat koko osakemarkkinoiden tilaa historiaan verrattuna. 
Jos osake markkinat ovat luultavasti alihinnoiteltu pelaaja ostaa,
ja jos taas osake on luultavasti yli arvioitu tekoäly myy osakkeensa ja lyhyeksi myy eniten ylihinnoiteltuja osakkeita. 

KÄYTTÖ 
- Valitse osake yleiskatsauksesta tai hakuehdotuksista 
- Syötä määrä ja käytä Osta, Myy tai Short -painiketta 
- Seuraa salkun arvoa ja lyhyeksi myynti asemaasi 

ASENNUS 
-Kloonaa repositorio: git clone https://github.com/Aarnixx/Stock-Market.git 
-Siirry projektin kansioon: cd Stock-Market 
-Asenna riippuvuudet, jotka ovat requirements.txt tiedostossa: pip install -r requirements.txt 
-Käynnistä: python main.py 
-Kaikki yhdellä kerralla: git clone https://github.com/Aarnixx/Stock-Market.git && cd Stock-Market && pip install -r requirements.txt && python main.py 
