#!/usr/bin/env python3
"""
r3_stress.py - Belastungstest fuer die R3-Strategie.

Ausgangslage nach r3_backtest.py
--------------------------------
    +0.502% je Trade gegen 0.074% beim Zufallseinstieg, t=5.64
    17 von 20 Werten positiv, alle neun Parameterzellen positiv,
    nach 2010 staerker als davor.

Das ist das erste Ergebnis der Reihe, das die erste Pruefung ueberlebt.
Bevor daraus irgendetwas folgt, muessen fuenf Fragen beantwortet sein.

  1) SIGNIFIKANZ MIT CLUSTERN
     Die 1565 Trades sind NICHT unabhaengig. SPY, QQQ, DIA und IWM
     bilden weitgehend denselben Markt ab. Bricht der Markt ein, feuert
     R3 bei zehn Werten am selben Tag - das ist EIN Ereignis, nicht
     zehn Beobachtungen. Ein t-Wert, der Unabhaengigkeit unterstellt,
     ist dann zu hoch. Dieser Test gruppiert nach Handelstag, Woche und
     Monat und rechnet auf den Gruppenmitteln.

  2) PORTFOLIO STATT DURCHSCHNITT
     +0.5% je Trade ist keine Jahresrendite. Mit begrenzter Zahl
     gleichzeitiger Positionen, echtem Kapitaleinsatz und Leerlauf
     zwischen den Signalen entsteht eine Kurve, die man mit Buy & Hold
     vergleichen kann.

  3) VERLUSTVERTEILUNG
     Kein Stop-Loss. Der schlechteste Trade lag bei -30%. Wie sieht der
     linke Rand aus, und was kostet er im Mittel?

  4) GEBUEHRENSTAFFEL
     Bei welchen Kosten kippt das Ergebnis?

  5) OHNE DIE GEWINNER
     AAPL, NVDA, V und MSFT trugen ueberdurchschnittlich bei. Bleibt
     ohne sie etwas uebrig?

Vorab festgelegte Huerde
------------------------
    Nach Cluster-Korrektur muss t ueber 3 bleiben, und ohne die vier
    Gewinner muss der Vorsprung ueber 0.25% je Trade liegen.

Aufruf
------
    python3 r3_stress.py --selftest
    python3 r3_stress.py
    python3 r3_stress.py --slots 10 --start 2000-01-01
"""

import argparse
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

# r3_backtest kann im selben Ordner oder in ../strategien liegen
_hier = os.path.dirname(os.path.abspath(__file__))
for _p in (_hier, os.path.join(os.path.dirname(_hier), "strategien")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import r3_backtest as r3          # noqa: E402

GEWINNER = ["AAPL", "NVDA", "V", "MSFT"]
BENCHMARK = "SPY"

log = logging.getLogger("r3_stress")


# ----------------------------------------------------------------------

def basis_namespace(args, **aenderungen):
    ns = argparse.Namespace(
        richtung=args.richtung, rsi_periode=2, ein_long=10, start_long=60,
        ein_short=90, start_short=40, aus_long=70, aus_short=30,
        sma=200, max_halten=20, kosten=args.kosten,
        seed=args.seed, sims=1, split="2010-01-01")
    for k, v in aenderungen.items():
        setattr(ns, k, v)
    return ns


def alle_trades(daten, ns, nur=None):
    """Alle Trades mit Datum. Liste von dicts."""
    raus = []
    for tk, df in daten.items():
        if nur is not None and tk not in nur:
            continue
        for (ein, dauer, rend, seite) in r3.handele(df, ns.richtung, ns):
            raus.append({"ticker": tk, "ein_idx": ein, "dauer": dauer,
                         "rendite": rend, "seite": seite,
                         "ein_datum": df.index[ein],
                         "aus_datum": df.index[min(len(df) - 1, ein + dauer)]})
    raus.sort(key=lambda x: x["ein_datum"])
    return raus


# ----------------------------------------------------------------------
# 1) Cluster-korrigierte Signifikanz
# ----------------------------------------------------------------------

def cluster_t(trades, freq=None):
    """t-Wert auf Gruppenmitteln. freq=None -> jeder Trade einzeln."""
    s = pd.Series([t["rendite"] for t in trades],
                  index=pd.DatetimeIndex([t["ein_datum"] for t in trades]))
    if freq is not None:
        s = s.groupby(pd.Grouper(freq=freq)).mean().dropna()
    if len(s) < 10:
        return float("nan"), float("nan"), len(s)
    se = s.std(ddof=1) / math.sqrt(len(s))
    t = s.mean() / se if se > 0 else float("nan")
    return float(t), float(s.mean()), len(s)


def test_cluster(trades, args):
    print("\n" + "=" * 76)
    print("1) SIGNIFIKANZ MIT CLUSTERN")
    print("=" * 76)
    print(f"{'Gruppierung':<26}{'Einheiten':>11}{'Schnitt':>11}{'t':>9}{'p':>9}")
    print("-" * 76)
    for name, freq in (("keine (jeder Trade)", None), ("Handelstag", "D"),
                       ("Woche", "W"), ("Monat", "ME"), ("Quartal", "QE")):
        t, m, n = cluster_t(trades, freq)
        if math.isnan(t):
            continue
        p = math.erfc(abs(t) / math.sqrt(2))
        print(f"{name:<26}{n:>11}{m:>10.3%}{t:>9.2f}{p:>9.3f}")
    print("-" * 76)
    t_tag, _, _ = cluster_t(trades, "D")
    t_roh, _, _ = cluster_t(trades, None)
    print(f"Der ungruppierte t-Wert ({t_roh:.2f}) ueberschaetzt die Sicherheit,")
    print("weil gleichzeitige Trades in korrelierten Werten mitgezaehlt werden.")
    if not math.isnan(t_tag):
        if abs(t_tag) >= 3:
            print(f"Nach Tagesgruppierung bleibt t={t_tag:.2f} - Huerde (3) gehalten.")
        else:
            print(f"Nach Tagesgruppierung nur t={t_tag:.2f} - Huerde (3) gerissen.")
    return t_tag


# ----------------------------------------------------------------------
# 2) Portfolio
# ----------------------------------------------------------------------

def tagesrenditen(df):
    """Rendite von Eroeffnung t zu Eroeffnung t+1, indiziert auf t."""
    o = df["Open"]
    return (o.shift(-1) / o - 1.0).fillna(0.0)


def portfolio(daten, trades, slots, kosten):
    """Kapitalgewichtete Simulation mit begrenzten Positionsplaetzen."""
    kalender = sorted(set().union(*[set(df.index) for df in daten.values()]))
    ret = {tk: tagesrenditen(df) for tk, df in daten.items()}
    nach_datum = {}
    for t in trades:
        nach_datum.setdefault(t["ein_datum"], []).append(t)

    kapital, bar = 1.0, 1.0
    offen = []
    kurve, belegung, uebergangen = [], [], 0

    for d in kalender:
        # Ausstiege
        noch = []
        for p in offen:
            if p["aus"] <= d:
                bar += p["wert"] * (1 - kosten)
            else:
                noch.append(p)
        offen = noch
        # Einstiege
        for t in nach_datum.get(d, []):
            if len(offen) >= slots:
                uebergangen += 1
                continue
            einsatz = kapital / slots
            if einsatz > bar:
                uebergangen += 1
                continue
            bar -= einsatz
            offen.append({"ticker": t["ticker"], "aus": t["aus_datum"],
                          "wert": einsatz * (1 - kosten), "seite": t["seite"]})
        # Tagesbewegung
        for p in offen:
            r = ret[p["ticker"]].get(d, 0.0)
            p["wert"] *= (1 + r * p["seite"])
        kapital = bar + sum(p["wert"] for p in offen)
        kurve.append(kapital)
        belegung.append(len(offen) / slots)

    s = pd.Series(kurve, index=pd.DatetimeIndex(kalender))
    return s, float(np.mean(belegung)), uebergangen


def kennzahlen(kurve, pro_jahr=252):
    r = kurve.pct_change().dropna()
    jahre = len(r) / pro_jahr
    cagr = kurve.iloc[-1] ** (1 / jahre) - 1 if jahre > 0 else float("nan")
    vol = r.std(ddof=1) * math.sqrt(pro_jahr)
    return {"CAGR": float(cagr), "Vol": float(vol),
            "Sharpe": float(r.mean() * pro_jahr / vol) if vol > 0 else float("nan"),
            "MaxDD": float((kurve / kurve.cummax() - 1).min())}


def test_portfolio(daten, trades, args):
    print("\n" + "=" * 76)
    print(f"2) PORTFOLIO ({args.slots} gleichzeitige Positionen)")
    print("=" * 76)
    kurve, belegung, uebergangen = portfolio(daten, trades, args.slots, args.kosten)
    k = kennzahlen(kurve)

    bh = None
    if BENCHMARK in daten:
        o = daten[BENCHMARK]["Open"]
        bh = (o / o.iloc[0]).reindex(kurve.index).ffill().bfill()

    print(f"{'':<20}{'R3-Portfolio':>16}{'Buy & Hold SPY':>18}")
    print("-" * 76)
    if bh is not None:
        kb = kennzahlen(bh)
        for feld in ("CAGR", "Vol", "Sharpe", "MaxDD"):
            a, b = k[feld], kb[feld]
            if feld == "Sharpe":
                print(f"{feld:<20}{a:>16.2f}{b:>18.2f}")
            else:
                print(f"{feld:<20}{a:>16.2%}{b:>18.2%}")
    else:
        for feld in ("CAGR", "Vol", "Sharpe", "MaxDD"):
            print(f"{feld:<20}{k[feld]:>16.2%}")

    print(f"\nMittlere Kapitalauslastung: {belegung:.1%}")
    print(f"Signale mangels freiem Platz uebergangen: {uebergangen}")
    print("-" * 76)
    if belegung < 0.5:
        print("Das Kapital liegt die meiste Zeit brach. Ein Vergleich der")
        print("Gesamtrendite mit Buy & Hold ist deshalb unfair zu R3 - aber")
        print("es ist die Rendite, die du tatsaechlich bekommst.")
    return k


# ----------------------------------------------------------------------
# 3) Verlustverteilung
# ----------------------------------------------------------------------

def test_verluste(trades):
    print("\n" + "=" * 76)
    print("3) VERLUSTVERTEILUNG (kein Stop-Loss)")
    print("=" * 76)
    a = np.array([t["rendite"] for t in trades])
    verluste = a[a < 0]
    gewinne = a[a > 0]
    print(f"Trades gesamt            {len(a)}")
    print(f"davon im Plus            {len(gewinne)} ({len(gewinne)/len(a):.1%})")
    print(f"Mittlerer Gewinn         {gewinne.mean():+.3%}")
    print(f"Mittlerer Verlust        {verluste.mean():+.3%}")
    print(f"Verhaeltnis              {abs(gewinne.mean()/verluste.mean()):.2f}"
          f"  (unter 1 heisst: Verluste sind groesser als Gewinne)")
    print("\nQuantile der Trade-Rendite:")
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"   {q:>3}%   {np.percentile(a, q):>+8.2%}")
    schlimmste = np.sort(a)[:max(1, len(a) // 20)]
    print(f"\nMittelwert der schlechtesten 5% der Trades: {schlimmste.mean():+.2%}")
    print(f"Schlechtester Einzeltrade: {a.min():+.2%}")
    print("-" * 76)
    print("Ohne Stop-Loss ist der linke Rand offen. Ein einzelner Ausreisser")
    print("kann mehrere Monate durchschnittlicher Ertraege loeschen.")


# ----------------------------------------------------------------------
# 4) Gebuehrenstaffel
# ----------------------------------------------------------------------

def test_gebuehren(daten, args):
    print("\n" + "=" * 76)
    print("4) GEBUEHRENSTAFFEL")
    print("=" * 76)
    print(f"{'je Seite':<14}{'Trades':>9}{'Schnitt':>11}{'t (Tage)':>11}{'Befund':>22}")
    print("-" * 76)
    for k in (0.0005, 0.0010, 0.0015, 0.0020, 0.0030):
        ns = basis_namespace(args, kosten=k)
        tr = alle_trades(daten, ns)
        if not tr:
            continue
        t_tag, m, _ = cluster_t(tr, "D")
        befund = "traegt" if (not math.isnan(t_tag) and t_tag >= 3 and m > 0) else "kippt"
        print(f"{k:<14.2%}{len(tr):>9}{m:>10.3%}{t_tag:>11.2f}{befund:>22}")
    print("-" * 76)
    print("Bei rund 5 Tagen Haltedauer schlagen Gebuehren hart durch - jeder")
    print("Trade zahlt sie voll, egal wie kurz er lief.")


# ----------------------------------------------------------------------
# 5) Ohne die Gewinner
# ----------------------------------------------------------------------

def test_ausschluss(daten, args):
    print("\n" + "=" * 76)
    print("5) OHNE DIE GROSSEN GEWINNER")
    print("=" * 76)
    ns = basis_namespace(args)
    voll = alle_trades(daten, ns)
    t_v, m_v, _ = cluster_t(voll, "D")
    print(f"{'Volles Universum':<34}{m_v:>10.3%}   t={t_v:.2f}")

    rest = [tk for tk in daten if tk not in GEWINNER]
    ohne = alle_trades(daten, ns, nur=set(rest))
    t_o, m_o, _ = cluster_t(ohne, "D")
    da = [g for g in GEWINNER if g in daten]
    print(f"{'Ohne ' + ', '.join(da):<34}{m_o:>10.3%}   t={t_o:.2f}")
    print("-" * 76)
    if m_v != 0:
        print(f"Vom Vorsprung bleiben {m_o/m_v:.0%}.")
    if m_o >= 0.0025 and not math.isnan(t_o) and t_o >= 3:
        print("Huerde (0.25% je Trade) gehalten.")
    else:
        print("Huerde (0.25% je Trade) gerissen.")

    print("\nEinzelausschluss:")
    for g in da:
        teil = [tk for tk in daten if tk != g]
        tr = alle_trades(daten, ns, nur=set(teil))
        t_e, m_e, _ = cluster_t(tr, "D")
        marke = "  <- traegt viel" if m_e < m_v * 0.8 else ""
        print(f"   ohne {g:<6}{m_e:>9.3%}   t={t_e:>5.2f}{marke}")
    return m_o, t_o


# ----------------------------------------------------------------------

def selbsttest():
    print("Selbsttest\n" + "-" * 70)
    fehler = 0
    rng = np.random.default_rng(11)

    # Kuenstlich korrelierte Trades: Cluster-t MUSS kleiner sein als roh
    tage = pd.date_range("2010-01-01", periods=200, freq="B")
    trades = []
    for d in tage:
        gemeinsam = rng.normal(0.005, 0.02)
        for _ in range(8):          # acht "Werte" mit fast identischem Ergebnis
            trades.append({"ein_datum": d, "rendite": gemeinsam + rng.normal(0, 0.001)})
    t_roh, _, n_roh = cluster_t(trades, None)
    t_tag, _, n_tag = cluster_t(trades, "D")
    ok = abs(t_tag) < abs(t_roh)
    print(f"[1] Cluster senkt den t-Wert ({t_roh:.2f} -> {t_tag:.2f}, "
          f"{n_roh} -> {n_tag} Einheiten): {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # Unkorrelierte Trades: Cluster darf kaum etwas aendern
    trades2 = []
    for d in tage:
        for _ in range(8):
            trades2.append({"ein_datum": d, "rendite": rng.normal(0.005, 0.02)})
    t_roh2, _, _ = cluster_t(trades2, None)
    t_tag2, _, _ = cluster_t(trades2, "D")
    ok = abs(t_tag2 - t_roh2) < abs(t_roh - t_tag)
    print(f"[2] Ohne Korrelation kaum Aenderung ({t_roh2:.2f} -> {t_tag2:.2f}): "
          f"{'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # Portfolio: dauerhaft investierter Einzelwert muss Buy & Hold ergeben
    n = 600
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    c = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)), index=idx)
    df = pd.DataFrame({"Close": c, "Open": c})
    daten = {"AAA": df}
    ein, aus = 10, n - 5
    tr = [{"ticker": "AAA", "ein_datum": idx[ein], "aus_datum": idx[aus],
           "rendite": 0.0, "seite": 1, "ein_idx": ein, "dauer": aus - ein}]
    kurve, bel, ueber = portfolio(daten, tr, slots=1, kosten=0.0)
    erwartet = c.iloc[aus] / c.iloc[ein]
    ok = abs(kurve.iloc[-1] - erwartet) / erwartet < 0.01
    print(f"[3] Portfolio trifft Buy & Hold ({kurve.iloc[-1]:.4f} gegen "
          f"{erwartet:.4f}): {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # Kosten muessen die Kurve senken
    k2, _, _ = portfolio(daten, tr, slots=1, kosten=0.01)
    ok = k2.iloc[-1] < kurve.iloc[-1]
    print(f"[4] Kosten senken die Endkurve: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # Positionsplaetze begrenzen wirklich
    viele = []
    for i in range(20):
        viele.append({"ticker": "AAA", "ein_datum": idx[20],
                      "aus_datum": idx[40], "rendite": 0.0, "seite": 1,
                      "ein_idx": 20, "dauer": 20})
    _, _, ueber2 = portfolio(daten, viele, slots=3, kosten=0.0)
    ok = ueber2 == 17
    print(f"[5] Platzbegrenzung greift ({ueber2} von 20 uebergangen): "
          f"{'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    print("-" * 70)
    print("Selbsttest bestanden." if fehler == 0
          else f"Selbsttest FEHLGESCHLAGEN ({fehler} Punkte).")
    return 0 if fehler == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--tickers", default=",".join(r3.STANDARD_TICKER))
    p.add_argument("--start", default=r3.START)
    p.add_argument("--end", default=None)
    p.add_argument("--richtung", default="long", choices=["long", "short", "beide"])
    p.add_argument("--kosten", type=float, default=0.0010)
    p.add_argument("--slots", type=int, default=5,
                   help="gleichzeitig offene Positionen")
    p.add_argument("--seed", type=int, default=31)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.selftest:
        return selbsttest()

    tickers = [x.strip() for x in args.tickers.split(",") if x.strip()]
    log.info("Lade %d Werte ...", len(tickers))
    daten = r3.lade(tickers, args.start, args.end)
    if not daten:
        log.error("Keine Daten")
        return 1
    log.info("%d Werte geladen", len(daten))

    ns = basis_namespace(args)
    trades = alle_trades(daten, ns)
    log.info("%d Trades", len(trades))

    t_tag = test_cluster(trades, args)
    test_portfolio(daten, trades, args)
    test_verluste(trades)
    test_gebuehren(daten, args)
    m_o, t_o = test_ausschluss(daten, args)

    print("\n" + "=" * 76)
    print("GESAMTBILD")
    print("=" * 76)
    h1 = (not math.isnan(t_tag)) and abs(t_tag) >= 3
    h2 = m_o >= 0.0025 and (not math.isnan(t_o)) and t_o >= 3
    print(f"Huerde 1 - t nach Tagesgruppierung ueber 3:      "
          f"{'gehalten' if h1 else 'gerissen'} (t={t_tag:.2f})")
    print(f"Huerde 2 - ohne die Gewinner ueber 0.25%/Trade:  "
          f"{'gehalten' if h2 else 'gerissen'} ({m_o:.3%})")
    if h1 and h2:
        print("\nBeide Huerden gehalten. Das ist noch kein Live-Bot, aber der")
        print("erste Ansatz der Reihe, der eine ernsthafte Pruefung uebersteht.")
        print("Naechster Schritt waere eine Out-of-Sample-Periode, die in")
        print("keiner der bisherigen Auswertungen vorkam.")
    else:
        print("\nMindestens eine Huerde gerissen. Vorab festgelegt heisst:")
        print("nicht nachverhandeln.")
    print("=" * 76 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
