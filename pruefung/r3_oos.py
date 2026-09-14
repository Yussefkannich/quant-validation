#!/usr/bin/env python3
"""
r3_oos.py - Out-of-Sample-Test der R3-Strategie.

Warum dieser Test noetig ist
----------------------------
Alles bisher Gemessene stammt aus EINER Datenmenge: 20 US-Werte ab 2000.
Auf genau diesen Daten wurden neun Parameterzellen, fuenf Stresstests und
mehrere Universumsvarianten durchprobiert. Selbst wenn jeder einzelne
Test ehrlich war, gilt: Wer oft genug hinschaut, findet irgendwo etwas.

Ein Out-of-Sample-Test beantwortet die einzige Frage, die zaehlt:
Gilt der Effekt auch dort, wo noch nie jemand gesucht hat?

Was hier anders ist
-------------------
  1. ANDERE MAERKTE. Deutschland, Frankreich, Niederlande, Grossbritannien,
     Japan, Kanada, Australien. Andere Handelszeiten, andere Anleger,
     andere Aufsicht. Kein einziger dieser Werte kam bisher vor.
  2. ANDERE ANLAGEKLASSE. Krypto, rund um die Uhr gehandelt, ohne
     Eroeffnungsluecken - eine voellig andere Mikrostruktur.
  3. FRISCHER ZEITRAUM. Zusaetzlich nur die Jahre ab dem Holdout-Datum,
     auf allen Universen.

  4. KEINE PARAMETERSUCHE. Die Werte stehen fest: RSI(2), Einstieg unter
     10, Ausstieg ueber 70, 200-Tage-Filter, Startwert unter 60. Es gibt
     bewusst KEIN Parameter-Gitter in diesem Skript. Wer out-of-sample
     noch optimiert, hat den Test verstanden, aber nicht begriffen.

Vorab festgelegte Huerde
------------------------
    Ein Universum gilt als bestanden, wenn der Schnitt ueber 0.25% je
    Trade liegt UND der nach Handelstag gruppierte t-Wert ueber 2 liegt.
    Die Schwelle ist niedriger als im Haupttest, weil die Stichproben
    kleiner sind.
    Insgesamt bestanden heisst: mindestens zwei Drittel der Universen
    bestehen, und der Gesamtschnitt liegt ueber 0.25%.

Aufruf
------
    python3 r3_oos.py --selftest
    python3 r3_oos.py
    python3 r3_oos.py --holdout 2020-01-01 --kosten 0.0015
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
import r3_stress as rst           # noqa: E402

# Festgeschriebene Parameter - hier wird NICHT optimiert.
FEST = {"rsi_periode": 2, "ein_long": 10, "start_long": 60, "aus_long": 70,
        "ein_short": 90, "start_short": 40, "aus_short": 30,
        "sma": 200, "max_halten": 20}

UNIVERSEN = {
    "Deutschland": ["SAP.DE", "SIE.DE", "ALV.DE", "BAS.DE", "BMW.DE",
                    "MBG.DE", "DTE.DE", "BAYN.DE", "MUV2.DE", "RWE.DE"],
    "Frankreich/NL": ["AIR.PA", "MC.PA", "OR.PA", "SAN.PA", "BNP.PA",
                      "ASML.AS", "AD.AS", "INGA.AS", "HEIA.AS", "PHIA.AS"],
    "Grossbritannien": ["HSBA.L", "SHEL.L", "AZN.L", "ULVR.L", "BP.L",
                        "GSK.L", "RIO.L", "BATS.L", "DGE.L", "LLOY.L"],
    "Japan": ["7203.T", "6758.T", "9984.T", "8306.T", "6861.T",
              "9432.T", "4063.T", "6098.T", "8058.T", "7974.T"],
    "Kanada/Australien": ["RY.TO", "TD.TO", "ENB.TO", "CNR.TO", "BNS.TO",
                          "BHP.AX", "CBA.AX", "CSL.AX", "WES.AX", "NAB.AX"],
    "Krypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD",
               "LTC-USD", "DOGE-USD", "LINK-USD", "BCH-USD", "XLM-USD"],
}

log = logging.getLogger("r3_oos")


def namespace(args, **extra):
    ns = argparse.Namespace(richtung=args.richtung, kosten=args.kosten,
                            seed=args.seed, sims=1, split="2010-01-01")
    for k, v in FEST.items():
        setattr(ns, k, v)
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns


def auswerten_universum(name, tickers, args, ab=None):
    """Laedt ein Universum und wertet es aus. Gibt ein dict zurueck."""
    try:
        daten = r3.lade(tickers, args.start, args.end)
    except Exception as exc:
        log.warning("%s: Laden fehlgeschlagen (%s)", name, exc)
        return None
    if not daten:
        log.warning("%s: keine verwertbaren Daten", name)
        return None

    ns = namespace(args)
    trades = rst.alle_trades(daten, ns)
    if ab is not None:
        grenze = pd.Timestamp(ab)
        trades = [t for t in trades if t["ein_datum"] >= grenze]
    if len(trades) < 20:
        log.warning("%s: nur %d Trades, zu wenig", name, len(trades))
        return None

    renditen = [t["rendite"] for t in trades]
    t_tag, m_tag, n_tag = rst.cluster_t(trades, "D")

    # Kontrollgruppe: Zufallseinstieg gleicher Haltedauer
    rng = np.random.default_rng(args.seed)
    zufall = []
    nach_ticker = {}
    for t in trades:
        nach_ticker.setdefault(t["ticker"], []).append(
            (t["ein_idx"], t["dauer"], t["rendite"], t["seite"]))
    for tk, tr in nach_ticker.items():
        for _ in range(args.sims):
            zufall.extend(r3.zufallstrades(daten[tk], tr, rng, args.kosten))

    return {
        "name": name, "werte": len(daten), "trades": len(trades),
        "mittel": float(np.mean(renditen)),
        "quote": float(np.mean([x > 0 for x in renditen])),
        "t": t_tag, "tage": n_tag,
        "zufall": float(np.mean(zufall)) if zufall else float("nan"),
        "zufall_quote": float(np.mean([x > 0 for x in zufall])) if zufall else float("nan"),
        "bestanden": (np.mean(renditen) >= args.huerde
                      and not math.isnan(t_tag) and t_tag >= args.t_huerde),
        "renditen": renditen,
    }


def tabelle(zeilen, titel, args):
    print("\n" + "=" * 82)
    print(titel)
    print("=" * 82)
    print(f"{'Universum':<20}{'Werte':>6}{'Trades':>8}{'Schnitt':>10}"
          f"{'Quote':>8}{'Zufall':>9}{'t':>7}{'':>6}")
    print("-" * 82)
    for z in zeilen:
        marke = "  ok" if z["bestanden"] else "  --"
        print(f"{z['name']:<20}{z['werte']:>6}{z['trades']:>8}"
              f"{z['mittel']:>9.3%}{z['quote']:>8.1%}{z['zufall']:>9.3%}"
              f"{z['t']:>7.2f}{marke:>6}")
    alle = [x for z in zeilen for x in z["renditen"]]
    if alle:
        print("-" * 82)
        bestanden = sum(1 for z in zeilen if z["bestanden"])
        print(f"{'GESAMT':<20}{'':>6}{len(alle):>8}{np.mean(alle):>9.3%}"
              f"{np.mean([x > 0 for x in alle]):>8.1%}")
        print(f"\nBestanden: {bestanden} von {len(zeilen)} Universen "
              f"(Huerde: {args.huerde:.2%} je Trade und t>{args.t_huerde})")
    return zeilen


# ----------------------------------------------------------------------

def selbsttest():
    print("Selbsttest\n" + "-" * 70)
    fehler = 0

    # [1] Parameter sind festgeschrieben, nicht aus args ableitbar
    args = argparse.Namespace(richtung="long", kosten=0.001, seed=1, sims=5,
                              huerde=0.0025, t_huerde=2.0)
    ns = namespace(args)
    ok = all(getattr(ns, k) == v for k, v in FEST.items())
    print(f"[1] Parameter festgeschrieben: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [2] Kein Parameter-Gitter im Skript: es darf keine Schleife geben,
    # die Einstiegs- oder Ausstiegsschwellen variiert. Die Suchbegriffe
    # werden zur Laufzeit zusammengesetzt, sonst findet der Test sich
    # selbst im eigenen Quelltext.
    quelltext = open(os.path.abspath(__file__)).read()
    verdaechtig = [f"for {w} in (" for w in ("ein", "aus", "schwelle")]
    verdaechtig += [f".{w}_{s} =" for w in ("ein", "aus")
                    for s in ("long", "short")]
    gefunden = [v for v in verdaechtig if v in quelltext]
    ok = not gefunden
    print(f"[2] Keine Parametersuche enthalten: "
          f"{'OK' if ok else 'FEHLER ' + str(gefunden)}")
    fehler += 0 if ok else 1

    # [3] Universen ueberschneiden sich nicht mit dem Haupttest
    haupt = set(r3.STANDARD_TICKER)
    alle = set(t for v in UNIVERSEN.values() for t in v)
    doppelt = haupt & alle
    ok = len(doppelt) == 0
    print(f"[3] Keine Ueberschneidung mit dem Haupttest: "
          f"{'OK' if ok else 'FEHLER ' + str(doppelt)}")
    fehler += 0 if ok else 1

    # [4] Huerdenlogik: knapp darunter faellt durch, knapp darueber besteht
    rng = np.random.default_rng(3)
    tage = pd.date_range("2015-01-01", periods=300, freq="B")
    schwach = [{"ein_datum": d, "rendite": rng.normal(0.0005, 0.02)} for d in tage]
    stark = [{"ein_datum": d, "rendite": rng.normal(0.008, 0.02)} for d in tage]
    t_s, m_s, _ = rst.cluster_t(schwach, "D")
    t_k, m_k, _ = rst.cluster_t(stark, "D")
    ok = (m_s < 0.0025 or t_s < 2) and (m_k >= 0.0025 and t_k >= 2)
    print(f"[4] Huerdenlogik trennt ({m_s:+.3%}/t={t_s:.1f} gegen "
          f"{m_k:+.3%}/t={t_k:.1f}): {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [5] Auswertung laeuft auf synthetischen Daten durch
    daten = {}
    for tk in ["A", "B", "C"]:
        n = 1500
        idx = pd.date_range("2012-01-01", periods=n, freq="B")
        c = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.013, n)), index=idx)
        daten[tk] = pd.DataFrame({"Close": c, "Open": c.shift(1).bfill()})
    tr = rst.alle_trades(daten, ns)
    ok = len(tr) > 10 and all("ein_datum" in t for t in tr)
    print(f"[5] Auswertungspfad laeuft ({len(tr)} Trades): "
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
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--holdout", default="2022-01-01",
                   help="zusaetzliche Auswertung nur ab diesem Datum")
    p.add_argument("--richtung", default="long", choices=["long", "short", "beide"])
    p.add_argument("--kosten", type=float, default=0.0010)
    p.add_argument("--huerde", type=float, default=0.0025)
    p.add_argument("--t-huerde", type=float, default=2.0)
    p.add_argument("--sims", type=int, default=10)
    p.add_argument("--seed", type=int, default=41)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.selftest:
        return selbsttest()

    print("\nParameter stehen fest und werden NICHT angepasst:")
    print("   " + ", ".join(f"{k}={v}" for k, v in FEST.items()))

    zeilen = []
    for name, tickers in UNIVERSEN.items():
        log.info("Lade %s (%d Werte) ...", name, len(tickers))
        z = auswerten_universum(name, tickers, args)
        if z:
            zeilen.append(z)
    if not zeilen:
        log.error("Kein Universum auswertbar")
        return 1

    tabelle(zeilen, "A) NEUE MAERKTE, VOLLER ZEITRAUM", args)

    zeilen_h = []
    for name, tickers in UNIVERSEN.items():
        z = auswerten_universum(name, tickers, args, ab=args.holdout)
        if z:
            zeilen_h.append(z)
    if zeilen_h:
        tabelle(zeilen_h, f"B) NUR AB {args.holdout}", args)

    alle = [x for z in zeilen for x in z["renditen"]]
    bestanden = sum(1 for z in zeilen if z["bestanden"])
    gesamt_ok = (bestanden >= math.ceil(len(zeilen) * 2 / 3)
                 and np.mean(alle) >= args.huerde)

    print("\n" + "=" * 82)
    print("URTEIL")
    print("=" * 82)
    print(f"Universen bestanden : {bestanden} von {len(zeilen)} "
          f"(noetig: {math.ceil(len(zeilen)*2/3)})")
    print(f"Gesamtschnitt       : {np.mean(alle):.3%} "
          f"(noetig: {args.huerde:.2%})")
    print()
    if gesamt_ok:
        print("BESTANDEN. Der Effekt zeigt sich auch in Maerkten, in denen")
        print("nie gesucht wurde. Das ist das staerkste verfuegbare Argument")
        print("dafuer, dass es sich nicht um Kurvenanpassung handelt.")
        print("\nWas damit NICHT bewiesen ist: dass er in Zukunft anhaelt, dass")
        print("er nach Slippage und Steuern uebrig bleibt, oder dass die")
        print("Kapitalauslastung ihn lohnend macht.")
    else:
        print("NICHT BESTANDEN. Der Effekt haengt an den Daten, auf denen er")
        print("gefunden wurde. Vorab festgelegt heisst: nicht nachverhandeln.")
    print("=" * 82 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
