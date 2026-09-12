#!/usr/bin/env python3
"""
rs_validate.py - Belastbarkeitspruefung fuer den Relative-Strength-Backtest.

Das erste Skript hat gesagt: +9.6% p.a. gegen SPY.
Dieses Skript prueft, ob davon etwas uebrig bleibt.

Vier Tests:

  A) FAIRE VERGLEICHSBASIS
     Gegen SPY zu vergleichen ist unzulaessig, weil das Universum nur aus
     heutigen Index-Mitgliedern besteht - die Pleiten und Absteiger der
     letzten zwoelf Jahre fehlen. Ein gleichgewichtetes Halten DERSELBEN
     100 Werte hat denselben Vorteil. Nur der Abstand dazu stammt
     wirklich aus dem Ranking.

  B) SIGNIFIKANZ
     t-Test auf den monatlichen Renditedifferenzen gegen die faire Basis.
     Unter |t| = 2 ist das Ergebnis nicht von Zufall zu unterscheiden.

  C) PARAMETER-ROBUSTHEIT
     Haelt der Vorsprung ueber verschiedene Fensterlaengen und
     Positionszahlen? Wenn nur eine Kombination funktioniert, hast du
     eine Zufallszelle gefunden, keinen Effekt.

  D) ZEITLICHE STABILITAET
     Jahresweise Aufschluesselung. Kommt der gesamte Vorsprung aus zwei
     Jahren, ist es kein Edge, sondern eine Episode.

  E) KONZENTRATION
     Welche Werte wurden am haeufigsten gehalten? Wenn das die bekannten
     Gewinner der Periode sind, misst der Backtest die Ticker-Auswahl,
     nicht die Strategie.

Aufruf:
    python3 rs_validate.py
    python3 rs_validate.py --selftest      # ohne Netz
    python3 rs_validate.py --start 2014-01-01
"""

import argparse
import logging
import math
import sys
from collections import Counter

import numpy as np
import pandas as pd

BENCHMARK = "SPY"
START = "2014-01-01"
COST_PER_SIDE = 0.0005

UNIVERSE = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX",
    "GD", "GE", "GILD", "GM", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
    "INTU", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
    "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL",
    "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS",
    "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT",
    "XOM",
]

log = logging.getLogger("rs_validate")


# ----------------------------------------------------------------------
# Daten
# ----------------------------------------------------------------------

def load_prices(tickers, start, end=None):
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance lieferte keine Daten")

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))
        if "Close" in lvl0:
            px = raw["Close"].copy()
        elif "Close" in lvl1:
            px = raw.xs("Close", axis=1, level=1).copy()
        else:
            raise RuntimeError("Keine Close-Spalte gefunden")
    else:
        px = raw[["Close"]].copy()
        px.columns = [tickers[0]]

    px = px.sort_index()
    leer = [c for c in px.columns if px[c].notna().sum() == 0]
    if leer:
        log.warning("Ohne Daten, ignoriert: %s", ", ".join(leer))
        px = px.drop(columns=leer)
    return px


# ----------------------------------------------------------------------
# Strategie und Kontrollen
# ----------------------------------------------------------------------

def momentum_scores(monthly, lookback, skip):
    return monthly.shift(skip) / monthly.shift(skip + lookback) - 1.0


def backtest(monthly, top_n, lookback, skip, cost_per_side):
    """Gibt (Renditen, gehaltene Werte je Monat, Umschlag) zurueck."""
    mom = momentum_scores(monthly, lookback, skip)
    fwd = monthly.pct_change(fill_method=None).shift(-1)

    dates, rets, picks_hist, turn = [], [], [], []
    prev = set()
    for t in mom.index:
        s = mom.loc[t].dropna()
        s = s[fwd.loc[t, s.index].notna()]
        if len(s) < top_n:
            continue
        picks = list(s.sort_values(ascending=False).head(top_n).index)
        brutto = float(fwd.loc[t, picks].mean())
        u = len(set(picks).symmetric_difference(prev)) / (2 * top_n)
        prev = set(picks)
        dates.append(t)
        rets.append(brutto - u * 2 * cost_per_side)
        picks_hist.append(picks)
        turn.append(u)
    return pd.Series(rets, index=pd.DatetimeIndex(dates)), picks_hist, turn


def equal_weight_universe(monthly):
    """Faire Kontrolle: gleichgewichtetes Halten aller Werte des Universums,
    monatlich zurueckgesetzt. Traegt denselben Survivorship Bias."""
    fwd = monthly.pct_change(fill_method=None).shift(-1)
    return fwd.mean(axis=1, skipna=True)


# ----------------------------------------------------------------------
# Statistik
# ----------------------------------------------------------------------

def cagr(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).prod() ** (12 / len(r)) - 1)


def norm_p(t):
    """Zweiseitiger p-Wert ueber Normalapproximation (kein scipy noetig)."""
    return math.erfc(abs(t) / math.sqrt(2))


def ttest(a, b):
    """Gepaarter t-Test auf monatlichen Differenzen."""
    idx = a.index.intersection(b.index)
    d = (a.reindex(idx) - b.reindex(idx)).dropna()
    n = len(d)
    if n < 12:
        return np.nan, np.nan, np.nan, n
    se = d.std(ddof=1) / math.sqrt(n)
    t = d.mean() / se if se > 0 else np.nan
    return float(d.mean() * 12), float(t), norm_p(t), n


def max_dd(r):
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


# ----------------------------------------------------------------------
# Ausgabe
# ----------------------------------------------------------------------

def report(monthly_uni, bench_m, args):
    strat, picks_hist, turn = backtest(
        monthly_uni, args.top_n, args.lookback, args.skip, args.costs)
    ew = equal_weight_universe(monthly_uni).reindex(strat.index).dropna()
    spy = None
    if bench_m is not None:
        spy = bench_m.pct_change(fill_method=None).shift(-1).reindex(strat.index).dropna()

    print("\n" + "=" * 66)
    print("A) FAIRE VERGLEICHSBASIS")
    print("=" * 66)
    print(f"{'Strategie (Top ' + str(args.top_n) + ')':<38}{cagr(strat):>10.2%} p.a.")
    print(f"{'Gleichgewicht, dasselbe Universum':<38}{cagr(ew):>10.2%} p.a.")
    if spy is not None:
        print(f"{'SPY (unverzerrter Index)':<38}{cagr(spy):>10.2%} p.a.")
    print("-" * 66)
    if spy is not None:
        print(f"{'Universum-Effekt (EW minus SPY)':<38}{cagr(ew)-cagr(spy):>10.2%} p.a.")
        print("   -> dieser Teil kommt allein aus der Ticker-Liste, nicht aus")
        print("      der Strategie. Reiner Survivorship Bias.")
    print(f"{'Ranking-Effekt (Strategie minus EW)':<38}{cagr(strat)-cagr(ew):>10.2%} p.a.")
    print("   -> NUR das ist der Beitrag der Strategie.")

    print("\n" + "=" * 66)
    print("B) SIGNIFIKANZ (gegen die faire Basis)")
    print("=" * 66)
    alpha, t, p, n = ttest(strat, ew)
    print(f"Mittlerer Vorsprung   {alpha:+.2%} p.a. ueber {n} Monate")
    print(f"t-Wert                {t:.2f}")
    print(f"p-Wert                {p:.3f}")
    if not np.isnan(t):
        if abs(t) < 2:
            print("BEWERTUNG: nicht signifikant. Mit dieser Datenmenge waere")
            print("           dieses Ergebnis auch ohne echten Edge moeglich.")
        elif abs(t) < 3:
            print("BEWERTUNG: grenzwertig signifikant. Beachte: du hast bereits")
            print("           mehrere Strategien getestet - bei genug Versuchen")
            print("           erreicht irgendeine davon t=2 rein zufaellig.")
        else:
            print("BEWERTUNG: statistisch deutlich. Trotzdem C-E pruefen.")
    print(f"\nMax Drawdown   Strategie {max_dd(strat):.2%}   Gleichgewicht {max_dd(ew):.2%}")
    print(f"Umschlag pro Monat: {np.mean(turn):.1%}")

    print("\n" + "=" * 66)
    print("C) PARAMETER-ROBUSTHEIT (Ranking-Effekt in % p.a. gegen EW)")
    print("=" * 66)
    lookbacks = [3, 6, 9, 12]
    tops = [5, 10, 20, 30]
    print(f"{'Fenster\\Top':<14}" + "".join(f"{'Top ' + str(x):>12}" for x in tops))
    positive = total = 0
    for lb in lookbacks:
        zeile = f"{str(lb) + ' Monate':<14}"
        for tn in tops:
            s, _, _ = backtest(monthly_uni, tn, lb, args.skip, args.costs)
            e = equal_weight_universe(monthly_uni).reindex(s.index).dropna()
            d = cagr(s) - cagr(e)
            total += 1
            positive += 1 if d > 0 else 0
            zeile += f"{d:>11.2%} "
        print(zeile)
    print("-" * 66)
    print(f"Positiv in {positive} von {total} Kombinationen.")
    if positive < total * 0.7:
        print("   -> Der Effekt haengt an der Parameterwahl. Das ist typisch")
        print("      fuer Kurvenanpassung, nicht fuer einen echten Edge.")
    else:
        print("   -> Breit positiv. Das spricht fuer einen echten Effekt.")

    print("\n" + "=" * 66)
    print("D) ZEITLICHE STABILITAET (Jahresrenditen)")
    print("=" * 66)
    df = pd.DataFrame({"Strategie": strat, "Gleichgew.": ew})
    if spy is not None:
        df["SPY"] = spy
    jahre = df.groupby(df.index.year).apply(
        lambda g: (1 + g).prod() - 1, include_groups=False)
    jahre["Diff zu EW"] = jahre["Strategie"] - jahre["Gleichgew."]
    with pd.option_context("display.float_format", lambda v: f"{v:>8.2%}"):
        print(jahre.to_string())
    gew = (jahre["Diff zu EW"] > 0).sum()
    print("-" * 66)
    print(f"Besser als die faire Basis in {gew} von {len(jahre)} Jahren.")
    top2 = jahre["Diff zu EW"].nlargest(2).sum()
    ges = jahre["Diff zu EW"].sum()
    if ges != 0:
        print(f"Anteil der zwei besten Jahre am Gesamtvorsprung: {top2/ges:.0%}")
        if top2 / ges > 0.7:
            print("   -> Der Vorsprung steckt fast komplett in zwei Jahren.")
            print("      Das ist eine Episode, keine Systematik.")

    print("\n" + "=" * 66)
    print("E) KONZENTRATION - haeufigste Positionen")
    print("=" * 66)
    c = Counter(x for p in picks_hist for x in p)
    monate = len(picks_hist)
    for name, k in c.most_common(12):
        print(f"   {name:<8}{k:>4} von {monate} Monaten   ({k/monate:>5.1%})")
    print("\nWenn das ueberwiegend die bekannten Gewinner der Periode sind,")
    print("misst der Backtest deine Ticker-Liste, nicht die Strategie.")
    print()

    return strat, ew, spy


# ----------------------------------------------------------------------

def selftest():
    print("Selbsttest\n" + "-" * 66)
    rng = np.random.default_rng(7)
    n_m, n_a = 200, 50
    idx = pd.date_range("2008-01-31", periods=n_m, freq="ME")
    cols = [f"A{i:02d}" for i in range(n_a)]

    zuf = pd.DataFrame(rng.normal(0.006, 0.055, (n_m, n_a)), index=idx, columns=cols)
    px = 100 * (1 + zuf).cumprod()
    s, _, _ = backtest(px, 10, 6, 1, 0.0)
    e = equal_weight_universe(px).reindex(s.index).dropna()
    a, t, p, n = ttest(s, e)
    print(f"[1] Zufallsdaten: Vorsprung {a:+.2%} p.a., t={t:.2f}, p={p:.3f}")
    print(f"    (t sollte betragsmaessig meist unter 2 liegen)")

    drift = rng.normal(0.0, 0.012, n_a)
    mit = pd.DataFrame(rng.normal(0.004, 0.045, (n_m, n_a)) + drift, index=idx, columns=cols)
    px2 = 100 * (1 + mit).cumprod()
    s2, _, _ = backtest(px2, 10, 6, 1, 0.0)
    e2 = equal_weight_universe(px2).reindex(s2.index).dropna()
    a2, t2, p2, _ = ttest(s2, e2)
    print(f"[2] Eingebautes Momentum: Vorsprung {a2:+.2%} p.a., t={t2:.2f}, p={p2:.3f}")

    halb = n_m // 2
    man = px2.copy()
    man.iloc[halb:] *= np.linspace(1.0, 3.0, n_m - halb)[:, None]
    gleich = np.allclose(
        momentum_scores(px2, 6, 1).iloc[:halb].fillna(0).values,
        momentum_scores(man, 6, 1).iloc[:halb].fillna(0).values)
    print(f"[3] Kein Look-Ahead: {'OK' if gleich else 'FEHLER'}")

    ok = gleich and t2 > t and a2 > a
    print("-" * 66)
    print("Selbsttest bestanden." if ok else "Selbsttest FEHLGESCHLAGEN.")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--lookback", type=int, default=6)
    p.add_argument("--skip", type=int, default=1)
    p.add_argument("--costs", type=float, default=COST_PER_SIDE)
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.selftest:
        return selftest()

    log.info("Lade Kurse ...")
    px = load_prices(UNIVERSE + [BENCHMARK], args.start, args.end)
    monthly = px.resample("ME").last()
    bench = monthly[BENCHMARK] if BENCHMARK in monthly.columns else None
    uni = monthly.drop(columns=[BENCHMARK], errors="ignore")
    log.info("%s bis %s, %d Werte, %d Monate",
             monthly.index[0].date(), monthly.index[-1].date(),
             uni.shape[1], len(monthly))

    strat, ew, spy = report(uni, bench, args)

    if args.csv:
        out = pd.DataFrame({"strategie": strat, "gleichgewicht": ew})
        if spy is not None:
            out["spy"] = spy
        out.to_csv(args.csv)
        log.info("Geschrieben: %s", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
