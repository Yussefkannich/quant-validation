#!/usr/bin/env python3
"""
ema_oos.py - Out-of-Sample-Test des EMA-Crossovers in Krypto.

Ausgangslage
------------
ema_crossover_backtest.py auf BTC/ETH/SOL (Kosten 0.26%):
  +7.04% p.a. gegen Buy & Hold, Perzentil 96.8 gegen Zufallstiming,
  5 von 6 Gitterzellen positiv.
Aber: nur drei stark korrelierte Coins, t-Wert der Tagesdifferenz
negativ, Vorsprung fast nur ab 2022. Das kann Zufall sein.

Die Frage dieses Skripts
------------------------
Gilt der Effekt auch auf Coins, auf denen der EMA-Crossover nie
angesehen wurde?

Was festgeschrieben ist
-----------------------
  * EMA 12/26, Kosten 0.26% je Wechsel, Start 2016, Teilung 2022-01-01
  * zehn neue Coins, keiner davon war im Haupttest
  * KEIN Parameter-Gitter. Wer out-of-sample noch optimiert, hat keinen
    Out-of-Sample-Test mehr.

Vorab festgelegte Huerde (15.09.2026, vor dem ersten Lauf)
----------------------------------------------------------
  1. Mittleres Perzentil gegen Zufallstiming mindestens 90
  2. Mindestens 70% der Coins vor Buy & Hold (bei 10 Coins: 7)
  3. Vorsprung im Mittel in BEIDEN Teilperioden positiv
Alle drei muessen erfuellt sein. Die Werte stehen als Konstanten im
Code und sind absichtlich NICHT per Kommandozeile aenderbar.

Zusaetzlich (nur Information, keine Huerde): t-Wert auf Log-Renditen.
Anders als der t-Wert auf einfachen Tagesrenditen hat er dasselbe
Vorzeichen wie die Differenz der Jahresrenditen.

Hinweis zur Verzerrung
----------------------
Yahoo fuehrt nur Coins, die heute noch existieren. Ausgestorbene Coins
fehlen. Das beschoenigt vor allem Buy & Hold - der Test ist in diesem
Punkt eher streng gegen die Strategie.

Aufruf
------
    python3 ema_oos.py --selftest
    python3 ema_oos.py
"""

import argparse
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

_hier = os.path.dirname(os.path.abspath(__file__))
for _p in (_hier, os.path.join(os.path.dirname(_hier), "strategien")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ema_crossover_backtest as eb   # noqa: E402

# ---- Festgeschrieben - hier wird NICHT optimiert -----------------------
FAST, SLOW = 12, 26
KOSTEN = 0.0026
START = "2016-01-01"
TEILUNG = "2022-01-01"
HAUPTTEST = ["BTC-USD", "ETH-USD", "SOL-USD"]
NEUE_COINS = ["ADA-USD", "XRP-USD", "DOGE-USD", "LTC-USD", "BCH-USD",
              "LINK-USD", "BNB-USD", "TRX-USD", "AVAX-USD", "DOT-USD"]
HUERDE_PERZENTIL = 90.0
HUERDE_ANTEIL_VORN = 0.70
# ------------------------------------------------------------------------

log = logging.getLogger("ema_oos")


def log_t(r_s, r_b):
    """t-Wert der Differenz der Log-Renditen (Vorzeichen passt zur CAGR)."""
    d = (np.log1p(r_s) - np.log1p(r_b)).dropna()
    if len(d) < 30 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))


def coin_auswerten(df, sims, rng):
    pos = eb.signal(df, FAST, SLOW)
    r = eb.rendite_serie(df, pos, KOSTEN).iloc[SLOW:]
    bh = df["Close"].pct_change(fill_method=None).shift(-1).reindex(r.index).dropna()
    r = r.reindex(bh.index)
    mz = float(pos.reindex(r.index).mean())
    n_handel = int(pos.diff().abs().sum())
    werte = []
    for _ in range(sims):
        zp = eb.zufallstiming(df, mz, max(2, n_handel), rng)
        zr = eb.rendite_serie(df, zp, KOSTEN).reindex(r.index).dropna()
        werte.append(eb.cagr(zr))
    v = np.array([x for x in werte if not np.isnan(x)])
    return {
        "r": r, "bh": bh, "mz": mz, "wechsel": n_handel,
        "diff": eb.cagr(r) - eb.cagr(bh),
        "perzentil": float((v < eb.cagr(r)).mean() * 100),
        "zufall": float(v.mean()),
        "t_log": log_t(r, bh),
        "start": r.index[0],
    }


def teilperioden(ergebnisse):
    s = pd.concat([e["r"].rename(k) for k, e in ergebnisse.items()], axis=1, sort=True).mean(axis=1).dropna()
    b = pd.concat([e["bh"].rename(k) for k, e in ergebnisse.items()], axis=1, sort=True).mean(axis=1).dropna()
    grenze = pd.Timestamp(TEILUNG)
    aus = {}
    for name, maske in [(f"bis {TEILUNG}", s.index < grenze),
                        (f"ab {TEILUNG}", s.index >= grenze)]:
        st = s[maske]
        bt = b.reindex(st.index).dropna()
        st = st.reindex(bt.index)
        if len(st) < 60:
            aus[name] = (float("nan"), float("nan"), float("nan"), float("nan"))
            continue
        aus[name] = (eb.cagr(st), eb.cagr(bt), eb.cagr(st) - eb.cagr(bt), log_t(st, bt))
    return s, b, aus


def urteil(ergebnisse, teil):
    n = len(ergebnisse)
    mp = float(np.mean([e["perzentil"] for e in ergebnisse.values()]))
    vorn = sum(1 for e in ergebnisse.values() if e["diff"] > 0)
    noetig = math.ceil(n * HUERDE_ANTEIL_VORN)
    diffs = [v[2] for v in teil.values()]
    beide = all(not math.isnan(x) and x > 0 for x in diffs)
    k1, k2, k3 = mp >= HUERDE_PERZENTIL, vorn >= noetig, beide
    return {"mp": mp, "vorn": vorn, "noetig": noetig, "k1": k1, "k2": k2,
            "k3": k3, "bestanden": k1 and k2 and k3}


def ausgabe(daten, args):
    print("\nFestgeschrieben: EMA %d/%d, Kosten %.2f%%, Start %s, Teilung %s"
          % (FAST, SLOW, KOSTEN * 100, START, TEILUNG))
    rng = np.random.default_rng(args.seed)
    ergebnisse = {tk: coin_auswerten(df, args.sims, rng) for tk, df in daten.items()}

    print("\n" + "=" * 92)
    print(f"A) NEUE COINS ({args.sims} Zufallsdurchlaeufe je Coin)")
    print("=" * 92)
    print(f"{'Coin':<10}{'ab':>12}{'EMA-Cross':>11}{'Buy&Hold':>11}{'Diff':>10}"
          f"{'t (log)':>9}{'Zufall':>10}{'Perz.':>8}{'Markt':>8}{'Wechsel':>9}")
    print("-" * 92)
    for tk, e in ergebnisse.items():
        print(f"{tk:<10}{e['start'].strftime('%Y-%m'):>12}{eb.cagr(e['r']):>10.2%}"
              f"{eb.cagr(e['bh']):>11.2%}{e['diff']:>+10.2%}{e['t_log']:>9.2f}"
              f"{e['zufall']:>10.2%}{e['perzentil']:>8.1f}{e['mz']:>8.0%}"
              f"{e['wechsel']:>9d}")

    s, b, teil = teilperioden(ergebnisse)
    print("-" * 92)
    print(f"{'MITTEL':<10}{'':>12}{eb.cagr(s):>10.2%}{eb.cagr(b):>11.2%}"
          f"{eb.cagr(s)-eb.cagr(b):>+10.2%}{log_t(s, b):>9.2f}")
    print(f"Max Drawdown   EMA-Cross {eb.max_dd(s):.1%}   Buy&Hold {eb.max_dd(b):.1%}")

    print("\n" + "=" * 92)
    print("B) TEILPERIODEN (Mittel ueber alle Coins)")
    print("=" * 92)
    for name, (cs, cb, d, t) in teil.items():
        print(f"{name:<18}EMA-Cross {cs:>8.2%}   Buy&Hold {cb:>8.2%}"
              f"   Diff {d:>+8.2%}   t(log)={t:>5.2f}")

    u = urteil(ergebnisse, teil)
    print("\n" + "=" * 92)
    print("URTEIL (Huerde vorab festgelegt)")
    print("=" * 92)
    haken = lambda ok: "erfuellt" if ok else "NICHT erfuellt"
    print(f"1. Mittleres Perzentil  {u['mp']:6.1f}  (noetig {HUERDE_PERZENTIL:.0f})"
          f"      {haken(u['k1'])}")
    print(f"2. Coins vor Buy&Hold   {u['vorn']:>3} von {len(ergebnisse)}  "
          f"(noetig {u['noetig']})       {haken(u['k2'])}")
    print(f"3. Beide Teilperioden positiv                  {haken(u['k3'])}")
    print()
    if u["bestanden"]:
        print("BESTANDEN. Der Effekt zeigt sich auch auf Coins, auf denen nie")
        print("gesucht wurde. Naechster Schritt waere Paper-Trading, kein Echtgeld.")
        print("\nNicht bewiesen: dass er anhaelt, dass er nach Slippage, Steuern")
        print("und Funding (bei Short) bleibt, und dass die Coins unabhaengig")
        print("sind - Krypto faellt und steigt weitgehend gemeinsam.")
    else:
        print("NICHT BESTANDEN. Der Vorsprung auf BTC/ETH/SOL traegt nicht")
        print("ueber die Coins hinaus, auf denen er gefunden wurde.")
        print("Vorab festgelegt heisst: nicht nachverhandeln.")
    print("=" * 92 + "\n")
    return u


# ----------------------------------------------------------------------

def selbsttest():
    print("Selbsttest\n" + "-" * 70)
    fehler = 0

    # [1] Keine Ueberschneidung mit dem Haupttest
    doppelt = set(NEUE_COINS) & (set(HAUPTTEST) | set(eb.STANDARD_TICKER))
    ok = not doppelt
    print(f"[1] Keine Ueberschneidung mit dem Haupttest: "
          f"{'OK' if ok else 'FEHLER ' + str(doppelt)}")
    fehler += 0 if ok else 1

    # [2] Parameter und Huerden sind nicht per Kommandozeile aenderbar.
    # Suchbegriffe zur Laufzeit zusammengesetzt, sonst findet der Test
    # sich selbst.
    quelltext = open(os.path.abspath(__file__)).read()
    verboten = ["--" + w for w in ("fast", "slow", "kosten", "huerde", "split")]
    verboten += ["for " + w + " in" for w in ("fa, sl", "fast", "slow")]
    gefunden = [v for v in verboten if v in quelltext]
    ok = not gefunden
    print(f"[2] Keine Parametersuche, Huerde nicht aenderbar: "
          f"{'OK' if ok else 'FEHLER ' + str(gefunden)}")
    fehler += 0 if ok else 1

    # [3] Log-t hat das Vorzeichen der CAGR-Differenz, auch wenn der
    # einfache Mittelwert das Gegenteil sagt (Schwankungsverlust).
    rng = np.random.default_rng(7)
    n = 3000
    idx = pd.date_range("2016-01-01", periods=n, freq="D")
    # Buy & Hold: abwechselnd +4% / -3.5%  -> einfach +0.25%, geometrisch ~+0.18%
    # Strategie:  gleichmaessig +0.20%     -> einfach +0.20%, geometrisch  +0.20%
    rb = pd.Series(np.where(np.arange(n) % 2 == 0, 0.04, -0.035), index=idx)
    rs = pd.Series(0.002 + rng.normal(0, 0.0001, n), index=idx)
    arith = (rs - rb).mean()
    cdiff = eb.cagr(rs) - eb.cagr(rb)
    t = log_t(rs, rb)
    ok = arith < 0 and cdiff > 0 and t > 0
    print(f"[3] Log-t folgt der CAGR (einfach {arith:+.4%}, CAGR {cdiff:+.1%}, "
          f"t_log {t:+.2f}): {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [4] Huerdenlogik
    def fake(diffs, perz):
        return {f"C{i}": {"diff": d, "perzentil": p} for i, (d, p) in enumerate(zip(diffs, perz))}
    gut = urteil(fake([0.05] * 8 + [-0.01] * 2, [95] * 10),
                 {"a": (0, 0, 0.02, 1), "b": (0, 0, 0.03, 1)})
    zu_wenig = urteil(fake([0.05] * 6 + [-0.01] * 4, [95] * 10),
                      {"a": (0, 0, 0.02, 1), "b": (0, 0, 0.03, 1)})
    perz_knapp = urteil(fake([0.05] * 10, [89.9] * 10),
                        {"a": (0, 0, 0.02, 1), "b": (0, 0, 0.03, 1)})
    eine_periode = urteil(fake([0.05] * 10, [95] * 10),
                          {"a": (0, 0, -0.001, 1), "b": (0, 0, 0.10, 1)})
    ok = (gut["bestanden"] and not zu_wenig["bestanden"]
          and not perz_knapp["bestanden"] and not eine_periode["bestanden"]
          and gut["noetig"] == 7)
    print(f"[4] Huerdenlogik (8/10 ok, 6/10 nein, Perz. 89.9 nein, "
          f"eine Periode negativ nein): {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [5] Gesamter Pfad auf Kunstdaten, mit unterschiedlichen Startdaten
    daten = {}
    for i, tk in enumerate(["X", "Y", "Z"]):
        m = 2500 - i * 600
        ix = pd.date_range(pd.Timestamp("2016-01-01") + pd.Timedelta(days=i * 600),
                           periods=m, freq="D")
        c = pd.Series(10 * np.cumprod(1 + rng.normal(0.001, 0.04, m)), index=ix)
        daten[tk] = pd.DataFrame({"Close": c})
    import io
    import contextlib
    puffer = io.StringIO()
    with contextlib.redirect_stdout(puffer):
        u = ausgabe(daten, argparse.Namespace(sims=20, seed=1))
    ok = "URTEIL" in puffer.getvalue() and isinstance(u["bestanden"], bool)
    print(f"[5] Auswertungspfad laeuft (Kunstdaten: "
          f"{'bestanden' if u['bestanden'] else 'nicht bestanden'}): "
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
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.selftest:
        return selbsttest()

    log.info("Lade %d neue Coins ...", len(NEUE_COINS))
    daten = eb.lade(NEUE_COINS, START)
    fehlend = [c for c in NEUE_COINS if c not in daten]
    if fehlend:
        log.warning("Nicht geladen: %s", ", ".join(fehlend))
    if len(daten) < 6:
        log.error("Nur %d Coins geladen - zu wenig fuer ein Urteil", len(daten))
        return 1
    ausgabe(daten, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
