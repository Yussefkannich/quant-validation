#!/usr/bin/env python3
"""
funding_monitor.py - Messwerkzeug fuer Funding-Rate-Carry auf Kraken Perpetuals.

KEIN HANDEL. Dieses Skript eroeffnet keine Positionen und braucht keine
API-Schluessel. Es misst nur, ob die Strategie sich ueberhaupt lohnen wuerde.

Die Strategie, um die es geht
-----------------------------
Spot long + Perpetual short in gleicher Groesse. Die Kursrichtung ist dann
egal (delta-neutral), verdient wird die Funding-Rate, die Long-Positionen
an Short-Positionen zahlen, solange sie positiv ist.

Warum erst messen
-----------------
Die Gebuehren fallen SOFORT an, der Ertrag tropft ueber Wochen herein:

    Einstieg : Spot kaufen (Taker ~0.26%) + Perp shorten (~0.05%)
    Ausstieg : Spot verkaufen (~0.26%)    + Perp schliessen (~0.05%)
    ---------------------------------------------------------------
    zusammen rund 0.62% vom Einsatz, unabhaengig von der Haltedauer

Bei 3% Funding im Jahr dauert es rund 75 Tage, bis das wieder drin ist.
Bei 11% rund 20 Tage. Faellt die Rate in der Zwischenzeit auf null, hast
du die Gebuehr bezahlt und nichts verdient. Dieses Skript rechnet genau
diesen Break-even aus - vor jedem Einsatz von Geld.

Was es tut
----------
  --now        aktuelle Funding-Raten aller beobachteten Kontrakte,
               annualisiert, mit Break-even-Haltedauer
  --history    historische Raten der letzten N Tage auswerten:
               Durchschnitt, Streuung, Anteil negativer Perioden,
               und was ein Carry ueber diesen Zeitraum NETTO gebracht haette
  --log        eine Zeile an funding_log.csv anhaengen (fuer cron/systemd,
               z.B. stuendlich - damit baust du dir eine eigene Messreihe)
  --selftest   Rechenlogik ohne Netz pruefen

Aufruf
------
    python3 funding_monitor.py --now
    python3 funding_monitor.py --history --days 90
    python3 funding_monitor.py --log
    python3 funding_monitor.py --selftest

Quelle: oeffentliche Kraken-Futures-API (kein Schluessel noetig).
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import urllib.request
from datetime import datetime, timezone, timedelta

FUTURES_BASE = "https://futures.kraken.com/derivatives/api/v3"
SPOT_BASE = "https://api.kraken.com/0/public"

# Beobachtete Perpetuals (Kraken-Notation: PF_ = linear perpetual)
KONTRAKTE = {
    "PF_XBTUSD": "BTC",
    "PF_ETHUSD": "ETH",
    "PF_SOLUSD": "SOL",
    "PF_XRPUSD": "XRP",
}

# Gebuehrenmodell - PRUEFE DAS GEGEN DEIN EIGENES KONTO.
# Kraken staffelt nach 30-Tage-Volumen; die Werte hier sind die
# ungünstigsten (kleinstes Volumen, Taker auf beiden Seiten).
SPOT_TAKER = 0.0026        # 0.26%
FUTURES_TAKER = 0.0005     # 0.05%
RUNDREISE = 2 * (SPOT_TAKER + FUTURES_TAKER)   # rein + raus, beide Beine

PERIODEN_PRO_TAG = 3       # Kraken zahlt Funding alle 8 Stunden
BASISORDNER = os.path.dirname(os.path.abspath(__file__))
LOGDATEI = os.path.join(BASISORDNER, "funding_log.csv")
ALARMLOG = os.path.join(BASISORDNER, "funding_alarms.log")
ALARMSTATUS = os.path.join(BASISORDNER, "funding_alarm_state.json")

# --- Schwellwert-Alarm ---------------------------------------------------
# Aus der Break-even-Rechnung: ab 0.01% je 8h liegt der Break-even bei rund
# 21 Tagen, darunter ueber 40. Das ist die Grenze, ab der es sich zu
# schauen lohnt - NICHT ein Signal zu handeln.
ALARM_SCHWELLE = 0.0001        # 0.01% je 8h
# Ein einzelner Ausschlag bedeutet nichts. Erst wenn mehrere Messungen
# hintereinander darueber liegen, ist es eine Phase und kein Zucken.
ALARM_BESTAETIGUNGEN = 3
# Danach Ruhe, damit ein laufender Zustand nicht stuendlich meldet.
ALARM_RUHE_STUNDEN = 12

log = logging.getLogger("funding")


# ----------------------------------------------------------------------
# Netzwerk
# ----------------------------------------------------------------------

def hole(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "funding-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def aktuelle_tickers():
    """Alle Futures-Tickers inkl. fundingRate und markPrice."""
    d = hole(f"{FUTURES_BASE}/tickers")
    if d.get("result") != "success":
        raise RuntimeError(f"Kraken-Antwort unerwartet: {d.get('error')}")
    return {t["symbol"]: t for t in d.get("tickers", [])}


def historische_raten(symbol, debug=False):
    """Historische Funding-Raten eines Kontrakts (oeffentlich, ohne Schluessel).

    Kraken hat diesen Endpunkt ueber die Versionen verschoben und ist bei
    der Gross-/Kleinschreibung des Symbols nicht konsistent. Statt einen
    Pfad zu raten, werden die bekannten Varianten der Reihe nach probiert
    und die erste erfolgreiche genommen.
    """
    kandidaten = []
    for basis in ("https://futures.kraken.com/derivatives/api/v4",
                  "https://futures.kraken.com/api/v4",
                  "https://futures.kraken.com/derivatives/api/v3",
                  "https://futures.kraken.com/api/v3"):
        for s in (symbol, symbol.lower()):
            kandidaten.append(f"{basis}/historicalfundingrates?symbol={s}")

    fehler = []
    for url in kandidaten:
        try:
            d = hole(url)
        except Exception as exc:
            fehler.append(f"{url} -> {exc}")
            continue
        raten = d.get("rates") or d.get("history") or []
        if raten:
            if debug:
                log.info("Funktionierender Endpunkt: %s", url)
            return raten
        fehler.append(f"{url} -> leere Antwort ({d.get('error', 'kein Fehler')})")

    raise RuntimeError("Kein Endpunkt lieferte Daten:\n  " + "\n  ".join(fehler))


# ----------------------------------------------------------------------
# Rechnen
# ----------------------------------------------------------------------

def annualisiert(rate_pro_periode, perioden_pro_tag=PERIODEN_PRO_TAG):
    """Funding-Rate je Periode -> Jahresrate (einfach, nicht verzinst)."""
    return rate_pro_periode * perioden_pro_tag * 365


def breakeven_tage(rate_pro_periode, rundreise=RUNDREISE,
                   perioden_pro_tag=PERIODEN_PRO_TAG):
    """Wie viele Tage muss die Position halten, bis die Gebuehr drin ist."""
    pro_tag = rate_pro_periode * perioden_pro_tag
    if pro_tag <= 0:
        return math.inf
    return rundreise / pro_tag


def carry_netto(raten, rundreise=RUNDREISE):
    """Was haette ein Carry ueber diese Perioden NETTO gebracht?

    raten: Liste von Funding-Raten je Periode (als Dezimalzahl).
    Ein Einstieg, ein Ausstieg, dazwischen alle Perioden kassiert.
    """
    if not raten:
        return 0.0
    brutto = sum(raten)
    return brutto - rundreise


def normalisiere_rate(eintrag):
    """Kraken liefert je nach Endpunkt 'relativeFundingRate' (absolut, in USD)
    oder 'fundingRate'. Fuer den Carry brauchen wir die RELATIVE Rate.

    Der Wert wird auf den Mark-Preis bezogen, falls nur die absolute
    Rate vorliegt. Gibt None zurueck, wenn sich nichts sinnvoll bilden laesst.
    """
    for schluessel in ("relativeFundingRate", "fundingRate"):
        v = eintrag.get(schluessel)
        if v is None:
            continue
        v = float(v)
        preis = eintrag.get("relativeFundingRatePrediction") and None
        # relativeFundingRate ist bereits relativ (z.B. 0.00003 = 0.003%)
        if schluessel == "relativeFundingRate":
            return v
        # fundingRate ist absolut in Quote-Waehrung -> durch Preis teilen
        p = eintrag.get("markPrice") or eintrag.get("indexPrice")
        if p:
            return v / float(p)
    return None


# ----------------------------------------------------------------------
# Befehle
# ----------------------------------------------------------------------

def zeige_jetzt():
    t = aktuelle_tickers()
    print(f"\nFunding-Raten  {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    print("=" * 74)
    print(f"{'Kontrakt':<12}{'je 8h':>10}{'annualisiert':>15}"
          f"{'Break-even':>14}{'Bewertung':>20}")
    print("-" * 74)
    for sym, name in KONTRAKTE.items():
        e = t.get(sym)
        if not e:
            print(f"{name:<12}{'nicht gefunden':>59}")
            continue
        r = normalisiere_rate(e)
        if r is None:
            print(f"{name:<12}{'keine Rate':>59}")
            continue
        jahr = annualisiert(r)
        be = breakeven_tage(r)
        if r <= 0:
            bew = "zahlt drauf"
        elif be > 120:
            bew = "lohnt nicht"
        elif be > 45:
            bew = "grenzwertig"
        else:
            bew = "interessant"
        be_txt = "nie" if be == math.inf else f"{be:.0f} Tage"
        print(f"{name:<12}{r*100:>9.4f}%{jahr:>14.2%}{be_txt:>14}{bew:>20}")
    print("-" * 74)
    print(f"Gebuehrenannahme: {RUNDREISE:.2%} Rundreise "
          f"(Spot {SPOT_TAKER:.2%} + Perp {FUTURES_TAKER:.2%}, je rein und raus)")
    print("Break-even = Haltedauer, ab der die Gebuehr wieder drin ist.\n")


def zeige_historie(tage, debug=False):
    print(f"\nHistorische Funding-Raten, letzte {tage} Tage")
    print("=" * 78)
    print(f"{'Kontrakt':<10}{'Schnitt/8h':>12}{'annual.':>10}{'Streuung':>11}"
          f"{'negativ':>10}{'Carry netto':>14}{'p.a.':>10}")
    print("-" * 78)
    grenze = datetime.now(timezone.utc) - timedelta(days=tage)
    for sym, name in KONTRAKTE.items():
        try:
            roh = historische_raten(sym, debug=debug)
        except Exception as exc:
            print(f"{name:<10}Fehler:")
            for zeile in str(exc).splitlines():
                print(f"   {zeile}")
            continue
        werte = []
        for e in roh:
            ts = e.get("timestamp")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt < grenze:
                continue
            r = normalisiere_rate(e)
            if r is not None:
                werte.append(r)
        if len(werte) < 3:
            print(f"{name:<10}{'zu wenig Daten':>30}")
            continue
        schnitt = sum(werte) / len(werte)
        var = sum((x - schnitt) ** 2 for x in werte) / (len(werte) - 1)
        streu = math.sqrt(var)
        neg = sum(1 for x in werte if x <= 0) / len(werte)
        netto = carry_netto(werte)
        realtage = len(werte) / PERIODEN_PRO_TAG
        netto_pa = netto * 365 / realtage if realtage > 0 else 0.0
        print(f"{name:<10}{schnitt*100:>11.4f}%{annualisiert(schnitt):>10.1%}"
              f"{streu*100:>10.4f}%{neg:>10.0%}{netto:>13.2%}{netto_pa:>10.1%}")
    print("-" * 78)
    print("Carry netto = was EIN Einstieg ueber den ganzen Zeitraum gebracht")
    print(f"haette, nach {RUNDREISE:.2%} Gebuehren. 'negativ' = Anteil der")
    print("Perioden, in denen du Funding ZAHLST statt bekommst.\n")
    print("Wichtig: Das ist eine Rueckschau auf eine einzige Marktphase.")
    print("Funding-Raten haengen an der Nachfrage nach Hebel - die kann")
    print("monatelang hoch bleiben und dann fuer Monate verschwinden.\n")


# ----------------------------------------------------------------------
# Schwellwert-Alarm
# ----------------------------------------------------------------------

def lade_alarmstatus():
    try:
        with open(ALARMSTATUS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def speichere_alarmstatus(status):
    try:
        with open(ALARMSTATUS, "w") as f:
            json.dump(status, f, indent=2)
    except OSError as exc:
        log.warning("Alarmstatus nicht speicherbar: %s", exc)


def letzte_messungen(name, anzahl):
    """Die letzten n geloggten Raten eines Kontrakts aus der CSV."""
    if not os.path.exists(LOGDATEI):
        return []
    werte = []
    try:
        with open(LOGDATEI, newline="") as f:
            for zeile in csv.DictReader(f):
                if zeile.get("kontrakt") == name:
                    try:
                        werte.append(float(zeile["rate_8h"]))
                    except (KeyError, ValueError):
                        pass
    except OSError as exc:
        log.warning("Logdatei nicht lesbar: %s", exc)
        return []
    return werte[-anzahl:]


def melde(text):
    """Alarm ausgeben: Datei, Konsole, und falls vorhanden Desktop-Hinweis."""
    zeit = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    zeile = f"[{zeit}] {text}"
    print(zeile)
    try:
        with open(ALARMLOG, "a") as f:
            f.write(zeile + "\n")
    except OSError as exc:
        log.warning("Alarmlog nicht schreibbar: %s", exc)
    # Desktop-Hinweis nur versuchen, nie daran scheitern.
    try:
        import shutil
        import subprocess
        if shutil.which("notify-send"):
            umgebung = dict(os.environ)
            umgebung.setdefault("DISPLAY", ":0")
            subprocess.run(["notify-send", "Funding-Alarm", text],
                           env=umgebung, timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def pruefe_alarm(gemessen, schwelle=ALARM_SCHWELLE,
                 bestaetigungen=ALARM_BESTAETIGUNGEN,
                 ruhe_stunden=ALARM_RUHE_STUNDEN, nur_pruefen=False):
    """Schlaegt an, wenn eine Rate mehrfach hintereinander ueber der
    Schwelle lag. gemessen: dict {Kontraktname: Rate je 8h}.

    Gibt die Liste der ausgeloesten Kontrakte zurueck.
    """
    status = lade_alarmstatus()
    jetzt = datetime.now(timezone.utc)
    ausgeloest = []

    for name, r in gemessen.items():
        eintrag = status.get(name, {})
        if r is None or r < schwelle:
            # Zustand zuruecksetzen, damit die naechste Phase wieder meldet
            if eintrag.get("aktiv"):
                eintrag["aktiv"] = False
                status[name] = eintrag
            continue

        letzte = letzte_messungen(name, bestaetigungen)
        genug = (len(letzte) >= bestaetigungen
                 and all(x >= schwelle for x in letzte))
        if not genug:
            continue

        zuletzt = eintrag.get("zuletzt")
        if zuletzt:
            try:
                vergangen = (jetzt - datetime.fromisoformat(zuletzt)).total_seconds() / 3600
            except ValueError:
                vergangen = ruhe_stunden + 1
            if eintrag.get("aktiv") and vergangen < ruhe_stunden:
                continue

        be = breakeven_tage(r)
        melde(f"{name}: {r*100:.4f}% je 8h ({annualisiert(r):.1%} p.a.), "
              f"Break-even {be:.0f} Tage - "
              f"{bestaetigungen} Messungen in Folge ueber {schwelle*100:.3f}%")
        ausgeloest.append(name)
        eintrag.update({"aktiv": True, "zuletzt": jetzt.isoformat(),
                        "rate": r})
        status[name] = eintrag

    if not nur_pruefen:
        speichere_alarmstatus(status)
    return ausgeloest


def schreibe_log(ohne_alarm=False):
    t = aktuelle_tickers()
    gemessen = {}
    jetzt = datetime.now(timezone.utc).isoformat()
    neu = not os.path.exists(LOGDATEI)
    with open(LOGDATEI, "a", newline="") as f:
        w = csv.writer(f)
        if neu:
            w.writerow(["zeit", "kontrakt", "rate_8h", "annualisiert",
                        "breakeven_tage", "mark_price"])
        for sym, name in KONTRAKTE.items():
            e = t.get(sym)
            if not e:
                continue
            r = normalisiere_rate(e)
            if r is None:
                continue
            be = breakeven_tage(r)
            gemessen[name] = r
            w.writerow([jetzt, name, f"{r:.8f}", f"{annualisiert(r):.6f}",
                        "" if be == math.inf else f"{be:.1f}",
                        e.get("markPrice", "")])
    log.info("An %s angehaengt", LOGDATEI)
    if not ohne_alarm:
        pruefe_alarm(gemessen)


# ----------------------------------------------------------------------

def selbsttest():
    print("Selbsttest\n" + "-" * 66)
    fehler = 0

    # 1) Annualisierung
    a = annualisiert(0.0001)          # 0.01% je 8h
    erwartet = 0.0001 * 3 * 365
    ok = abs(a - erwartet) < 1e-12
    print(f"[1] Annualisierung 0.01%/8h -> {a:.2%} p.a. {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # 2) Break-even gegen Handrechnung
    be = breakeven_tage(0.0001)       # 0.03%/Tag gegen 0.62% Gebuehr
    hand = RUNDREISE / (0.0001 * 3)
    ok = abs(be - hand) < 1e-9
    print(f"[2] Break-even bei 0.01%/8h: {be:.1f} Tage {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # 3) Negative Rate -> nie rentabel
    ok = breakeven_tage(-0.0001) == math.inf and breakeven_tage(0.0) == math.inf
    print(f"[3] Negative/Null-Rate ergibt 'nie': {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # 4) Carry muss Gebuehren abziehen
    raten = [0.0001] * 90             # 30 Tage
    netto = carry_netto(raten)
    ok = abs(netto - (0.009 - RUNDREISE)) < 1e-9 and netto < sum(raten)
    print(f"[4] Carry 30 Tage bei 0.01%/8h: brutto {sum(raten):.2%}, "
          f"netto {netto:.2%} {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # 5) Kurze Haltedauer muss negativ sein
    kurz = carry_netto([0.0001] * 9)  # 3 Tage
    ok = kurz < 0
    print(f"[5] Carry ueber nur 3 Tage: {kurz:.2%} (muss negativ sein) "
          f"{'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # 6) Raten-Normalisierung beider Formate
    r1 = normalisiere_rate({"relativeFundingRate": 0.00003})
    r2 = normalisiere_rate({"fundingRate": 3.0, "markPrice": 100000.0})
    ok = abs(r1 - 0.00003) < 1e-12 and abs(r2 - 0.00003) < 1e-12
    print(f"[6] Beide Rate-Formate ergeben {r1:.5f} / {r2:.5f} "
          f"{'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # 7) Realistisches Szenario zur Einordnung
    print("\n    Einordnung mit echten Groessenordnungen:")
    for name, r in [("Maerz 2026 (Kraken, 0.003%/8h)", 0.00003),
                    ("Juni 2026 (0.01%/8h)", 0.0001),
                    ("Hochphase (0.05%/8h)", 0.0005)]:
        be = breakeven_tage(r)
        print(f"      {name:<34}{annualisiert(r):>7.1%} p.a., "
              f"Break-even {be:>5.0f} Tage")

    # 8) Alarmlogik gegen eine kuenstliche Messreihe
    import tempfile
    global LOGDATEI, ALARMSTATUS, ALARMLOG
    sicher = (LOGDATEI, ALARMSTATUS, ALARMLOG)
    with tempfile.TemporaryDirectory() as tmp:
        LOGDATEI = os.path.join(tmp, "log.csv")
        ALARMSTATUS = os.path.join(tmp, "state.json")
        ALARMLOG = os.path.join(tmp, "alarm.log")
        with open(LOGDATEI, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["zeit", "kontrakt", "rate_8h", "annualisiert",
                        "breakeven_tage", "mark_price"])
            for x in [0.00002, 0.00002, 0.00002]:
                w.writerow(["t", "BTC", f"{x:.8f}", "", "", ""])
        leise = pruefe_alarm({"BTC": 0.00002})
        ok = leise == []
        print(f"\n[7] Niedrige Rate loest nichts aus: {'OK' if ok else 'FEHLER'}")
        fehler += 0 if ok else 1

        with open(LOGDATEI, "a", newline="") as f:
            w = csv.writer(f)
            for x in [0.00015, 0.00015, 0.00015]:
                w.writerow(["t", "BTC", f"{x:.8f}", "", "", ""])
        laut = pruefe_alarm({"BTC": 0.00015})
        ok = laut == ["BTC"]
        print(f"[8] Drei hohe Messungen loesen aus: {'OK' if ok else 'FEHLER'}")
        fehler += 0 if ok else 1

        nochmal = pruefe_alarm({"BTC": 0.00015})
        ok = nochmal == []
        print(f"[9] Ruhezeit verhindert Wiederholung: {'OK' if ok else 'FEHLER'}")
        fehler += 0 if ok else 1

        with open(LOGDATEI, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "BTC", "0.00015000", "", "", ""])
            w.writerow(["t", "BTC", "0.00002000", "", "", ""])
        einzeln = pruefe_alarm({"BTC": 0.00002})
        ok = einzeln == []
        print(f"[10] Einzelner Ausschlag reicht nicht: {'OK' if ok else 'FEHLER'}")
        fehler += 0 if ok else 1
    LOGDATEI, ALARMSTATUS, ALARMLOG = sicher

    print("-" * 66)
    print("Selbsttest bestanden." if fehler == 0
          else f"Selbsttest FEHLGESCHLAGEN ({fehler} Punkte).")
    return 0 if fehler == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--now", action="store_true", help="aktuelle Raten anzeigen")
    p.add_argument("--history", action="store_true", help="Historie auswerten")
    p.add_argument("--days", type=int, default=90, help="Zeitraum fuer --history")
    p.add_argument("--log", action="store_true", help="Zeile an funding_log.csv")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="zeigt, welcher API-Pfad funktioniert hat")
    p.add_argument("--no-alarm", action="store_true",
                   help="beim Loggen keinen Schwellwert-Alarm pruefen")
    p.add_argument("--test-alarm", action="store_true",
                   help="Alarmweg einmal testen (schreibt Testmeldung)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.selftest:
        return selbsttest()
    if args.test_alarm:
        melde("TEST - so sieht ein Alarm aus. Kein echter Messwert.")
        print(f"\nGeschrieben nach: {ALARMLOG}")
        print("Kam kein Desktop-Hinweis, fehlt notify-send oder DISPLAY -")
        print("die Datei wird trotzdem immer geschrieben.")
        return 0
    if args.log:
        schreibe_log(ohne_alarm=args.no_alarm)
        return 0
    if args.history:
        zeige_historie(args.days, debug=args.debug)
        return 0
    zeige_jetzt()
    return 0


if __name__ == "__main__":
    sys.exit(main())
