#!/usr/bin/env python3
"""
rs_stress.py - Belastungstest fuer den Relative-Strength-Effekt.

Ausgangslage nach rs_validate.py:
    Ranking-Effekt +7.7% p.a., t=1.96 (grenzwertig),
    68% des Vorsprungs aus zwei Jahren,
    NVDA in 54% aller Monate gehalten, AMD in 46%.

Verdacht: Der Effekt ist keine Systematik, sondern die Halbleiter- und
Big-Tech-Hausse 2016-2020, eingefangen durch eine Ticker-Liste, die
diese Gewinner rueckblickend enthaelt.

Fuenf Tests, die diesen Verdacht pruefen:

  1) AUSSCHLUSS DER TREIBER
     Universum ohne NVDA, AMD, TSLA, AVGO, NFLX. Ueberlebt der Effekt?
     Zusaetzlich: Einzelausschluss jedes haeufig gehaltenen Werts
     (leave-one-out) - haengt alles an einem Namen?

  2) TEILPERIODEN
     2014-2020 gegen 2021-2026 getrennt. Eine Systematik wirkt in
     beiden Haelften, eine Episode nur in einer.

  3) LONG-SHORT (marktneutral)
     Top N long, Bottom N short. Wenn das Ranking Information traegt,
     muss auch die Differenz positiv sein. Bleibt nur die Long-Seite
     positiv, hattest du schlicht mehr Marktrisiko.

  4) BETA-ADJUSTIERUNG
     Regression der Strategie auf das gleichgewichtete Universum.
     Der Achsenabschnitt ist der Teil, der NICHT aus hoeherem
     Marktrisiko stammt. Deine Vola lag bei 18.7% gegen 14.8% -
     ein Teil des Vorsprungs ist schlicht dafuer bezahlt.

  5) ZUFALLSPORTFOLIOS
     500 Portfolios, die monatlich N Werte REIN ZUFAELLIG ziehen.
     Wo liegt die echte Strategie in dieser Verteilung? Bei einem
     echten Edge deutlich oben. Landet sie im Mittelfeld, misst der
     Backtest nur die Konzentration auf wenige Werte.

Aufruf:
    python3 rs_stress.py
    python3 rs_stress.py --selftest
"""

import argparse
import logging
import math
import sys

import numpy as np
import pandas as pd

BENCHMARK = "SPY"
START = "2014-01-01"
COST_PER_SIDE = 0.0005
TREIBER = ["NVDA", "AMD", "TSLA", "AVGO", "NFLX"]

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

log = logging.getLogger("rs_stress")


# ----------------------------------------------------------------------

def load_prices(tickers, start, end=None):
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance lieferte keine Daten")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in set(raw.columns.get_level_values(0)):
            px = raw["Close"].copy()
        elif "Close" in set(raw.columns.get_level_values(1)):
            px = raw.xs("Close", axis=1, level=1).copy()
        else:
            raise RuntimeError("Keine Close-Spalte")
    else:
        px = raw[["Close"]].copy()
        px.columns = [tickers[0]]
    px = px.sort_index()
    leer = [c for c in px.columns if px[c].notna().sum() == 0]
    if leer:
        log.warning("Ohne Daten, ignoriert: %s", ", ".join(leer))
        px = px.drop(columns=leer)
    return px


def momentum_scores(monthly, lookback, skip):
    return monthly.shift(skip) / monthly.shift(skip + lookback) - 1.0


def backtest(monthly, top_n, lookback, skip, cost, seite="long"):
    """seite: 'long' = Top N, 'short' = Bottom N, 'ls' = Top minus Bottom."""
    mom = momentum_scores(monthly, lookback, skip)
    fwd = monthly.pct_change(fill_method=None).shift(-1)
    dates, rets = [], []
    prev = set()
    for t in mom.index:
        s = mom.loc[t].dropna()
        s = s[fwd.loc[t, s.index].notna()]
        if len(s) < 2 * top_n:
            continue
        rang = s.sort_values(ascending=False)
        top = list(rang.head(top_n).index)
        bot = list(rang.tail(top_n).index)
        r_top = float(fwd.loc[t, top].mean())
        r_bot = float(fwd.loc[t, bot].mean())
        if seite == "long":
            r, aktuell = r_top, set(top)
        elif seite == "short":
            r, aktuell = -r_bot, set(bot)
        else:
            r, aktuell = r_top - r_bot, set(top) | set(bot)
        u = len(aktuell.symmetric_difference(prev)) / (2 * len(aktuell))
        prev = aktuell
        dates.append(t)
        rets.append(r - u * 2 * cost)
    return pd.Series(rets, index=pd.DatetimeIndex(dates))


def equal_weight(monthly):
    return monthly.pct_change(fill_method=None).shift(-1).mean(axis=1, skipna=True)


def cagr(r):
    r = r.dropna()
    return float((1 + r).prod() ** (12 / len(r)) - 1) if len(r) else np.nan


def norm_p(t):
    return math.erfc(abs(t) / math.sqrt(2))


def ttest(a, b):
    idx = a.index.intersection(b.index)
    d = (a.reindex(idx) - b.reindex(idx)).dropna()
    if len(d) < 12:
        return np.nan, np.nan, np.nan, len(d)
    se = d.std(ddof=1) / math.sqrt(len(d))
    t = d.mean() / se if se > 0 else np.nan
    return float(d.mean() * 12), float(t), norm_p(t), len(d)


def ols_alpha(y, x):
    """y = a + b*x. Gibt (alpha p.a., beta, t-Wert alpha) zurueck."""
    idx = y.index.intersection(x.index)
    yy = y.reindex(idx).dropna()
    xx = x.reindex(yy.index).dropna()
    yy = yy.reindex(xx.index)
    n = len(yy)
    if n < 24:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(n), xx.values])
    beta, *_ = np.linalg.lstsq(X, yy.values, rcond=None)
    resid = yy.values - X @ beta
    s2 = resid @ resid / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    t_a = beta[0] / math.sqrt(cov[0, 0])
    return float(beta[0] * 12), float(beta[1]), float(t_a)


# ----------------------------------------------------------------------

def test_ausschluss(monthly, args):
    print("\n" + "=" * 68)
    print("1) AUSSCHLUSS DER TREIBER")
    print("=" * 68)
    voll = backtest(monthly, args.top_n, args.lookback, args.skip, args.costs)
    ew_voll = equal_weight(monthly).reindex(voll.index).dropna()
    basis = cagr(voll) - cagr(ew_voll)
    print(f"{'Volles Universum':<44}{basis:>10.2%} p.a.")

    da = [t for t in TREIBER if t in monthly.columns]
    ohne = monthly.drop(columns=da)
    s = backtest(ohne, args.top_n, args.lookback, args.skip, args.costs)
    e = equal_weight(ohne).reindex(s.index).dropna()
    rest = cagr(s) - cagr(e)
    a, t, p, _ = ttest(s, e)
    print(f"{'Ohne ' + ', '.join(da):<44}{rest:>10.2%} p.a.")
    print(f"{'  davon t-Wert / p-Wert':<44}{t:>10.2f} / {p:.3f}")
    print("-" * 68)
    if basis != 0:
        print(f"Vom urspruenglichen Effekt bleiben {rest/basis:.0%} uebrig.")
    if rest <= 0.02 or (not np.isnan(t) and abs(t) < 1.5):
        print("BEFUND: Der Effekt haengt an diesen fuenf Werten. Das ist")
        print("        keine Strategie, das ist eine Sektorwette, die man")
        print("        erst im Rueckblick treffen konnte.")
    else:
        print("BEFUND: Etwas ueberlebt den Ausschluss. Weiter mit Test 2-5.")

    print("\nEinzelausschluss (leave-one-out), Ranking-Effekt danach:")
    kandidaten = [c for c in ["NVDA", "AMD", "TSLA", "AVGO", "NFLX",
                              "LLY", "META", "AMZN", "AAPL", "GE"]
                  if c in monthly.columns]
    for k in kandidaten:
        m2 = monthly.drop(columns=[k])
        s2 = backtest(m2, args.top_n, args.lookback, args.skip, args.costs)
        e2 = equal_weight(m2).reindex(s2.index).dropna()
        d = cagr(s2) - cagr(e2)
        marke = "  <-- traegt den Effekt" if d < basis * 0.7 else ""
        print(f"   ohne {k:<6}{d:>9.2%}{marke}")


def test_teilperioden(monthly, args):
    print("\n" + "=" * 68)
    print("2) TEILPERIODEN")
    print("=" * 68)
    s = backtest(monthly, args.top_n, args.lookback, args.skip, args.costs)
    e = equal_weight(monthly).reindex(s.index).dropna()
    grenze = pd.Timestamp(args.split)
    for name, maske in [("bis " + args.split, s.index < grenze),
                        ("ab " + args.split, s.index >= grenze)]:
        s_t = s[maske]
        e_t = e.reindex(s_t.index).dropna()
        if len(s_t) < 12:
            continue
        a, t, p, n = ttest(s_t, e_t)
        print(f"{name:<16}Strategie {cagr(s_t):>8.2%}   Gleichgew. {cagr(e_t):>8.2%}"
              f"   Diff {cagr(s_t)-cagr(e_t):>+8.2%}   t={t:>5.2f}  ({n} Mon.)")
    print("-" * 68)
    print("Eine Systematik wirkt in beiden Haelften. Eine Episode nur in einer.")


def test_longshort(monthly, args):
    print("\n" + "=" * 68)
    print("3) LONG-SHORT (marktneutral)")
    print("=" * 68)
    lo = backtest(monthly, args.top_n, args.lookback, args.skip, args.costs, "long")
    sh = backtest(monthly, args.top_n, args.lookback, args.skip, args.costs, "short")
    ls = backtest(monthly, args.top_n, args.lookback, args.skip, args.costs, "ls")
    e = equal_weight(monthly).reindex(lo.index).dropna()
    print(f"{'Long Top ' + str(args.top_n):<30}{cagr(lo):>10.2%} p.a.")
    print(f"{'Short Bottom ' + str(args.top_n):<30}{cagr(sh):>10.2%} p.a.")
    print(f"{'Long-Short':<30}{cagr(ls):>10.2%} p.a.")
    print(f"{'Gleichgewicht (Referenz)':<30}{cagr(e):>10.2%} p.a.")
    vol_ls = ls.std(ddof=1) * math.sqrt(12)
    sr = (ls.mean() * 12) / vol_ls if vol_ls > 0 else np.nan
    t_ls = ls.mean() / (ls.std(ddof=1) / math.sqrt(len(ls))) if len(ls) > 12 else np.nan
    print(f"\nLong-Short: Vola {vol_ls:.2%}  Sharpe {sr:.2f}  t={t_ls:.2f} "
          f"p={norm_p(t_ls):.3f}")
    print("-" * 68)
    if cagr(ls) <= 0:
        print("BEFUND: Die Rangordnung selbst traegt keine Information. Der")
        print("        Vorsprung der Long-Seite kam aus Marktrichtung, nicht")
        print("        aus dem Ranking.")
    elif not np.isnan(t_ls) and abs(t_ls) < 2:
        print("BEFUND: Long-Short positiv, aber nicht signifikant.")
    else:
        print("BEFUND: Auch marktneutral positiv. Das ist das staerkste")
        print("        Argument fuer einen echten Ranking-Effekt.")


def test_beta(monthly, args):
    print("\n" + "=" * 68)
    print("4) BETA-ADJUSTIERUNG")
    print("=" * 68)
    s = backtest(monthly, args.top_n, args.lookback, args.skip, args.costs)
    e = equal_weight(monthly).reindex(s.index).dropna()
    s = s.reindex(e.index)
    alpha, beta, t_a = ols_alpha(s, e)
    roh = cagr(s) - cagr(e)
    print(f"Roher Vorsprung                {roh:>+9.2%} p.a.")
    print(f"Beta gegen das Universum       {beta:>9.2f}")
    print(f"Alpha nach Beta-Korrektur      {alpha:>+9.2%} p.a.   t={t_a:.2f}"
          f"  p={norm_p(t_a):.3f}")
    print("-" * 68)
    if beta > 1.1:
        mehr = (beta - 1) * cagr(e)
        print(f"Beta ueber 1 heisst: rund {mehr:.2%} p.a. des Vorsprungs sind")
        print("schlicht die Bezahlung fuer hoeheres Marktrisiko. Dafuer")
        print("braucht man keine Strategie, dafuer reicht ein Hebel.")
    if not np.isnan(t_a) and abs(t_a) < 2:
        print("Das risikobereinigte Alpha ist nicht signifikant.")


def test_zufall(monthly, args, n_sim=500, seed=11):
    print("\n" + "=" * 68)
    print(f"5) ZUFALLSPORTFOLIOS ({n_sim} Durchlaeufe)")
    print("=" * 68)
    s = backtest(monthly, args.top_n, args.lookback, args.skip, args.costs)
    e = equal_weight(monthly).reindex(s.index).dropna()
    echt = cagr(s) - cagr(e)

    fwd = monthly.pct_change(fill_method=None).shift(-1)
    rng = np.random.default_rng(seed)
    verteilung = []
    termine = list(s.index)
    for _ in range(n_sim):
        rr = []
        for t in termine:
            verf = fwd.loc[t].dropna().index
            if len(verf) < args.top_n:
                continue
            pick = rng.choice(len(verf), args.top_n, replace=False)
            rr.append(float(fwd.loc[t, verf[pick]].mean()))
        r = pd.Series(rr)
        verteilung.append(cagr(r) - cagr(e.iloc[:len(r)]))
    v = np.array(verteilung)
    perzentil = float((v < echt).mean() * 100)
    print(f"Echte Strategie          {echt:>+9.2%} p.a.")
    print(f"Zufallsportfolios        Mittel {v.mean():>+7.2%}   "
          f"Streuung {v.std(ddof=1):.2%}")
    print(f"                         5%-95% Band "
          f"{np.percentile(v,5):+.2%} bis {np.percentile(v,95):+.2%}")
    print(f"\nDie echte Strategie liegt im {perzentil:.1f}. Perzentil.")
    print("-" * 68)
    if perzentil < 90:
        print("BEFUND: Zufaellig gezogene Portfolios aus demselben Universum")
        print("        erreichen das ebenfalls. Das Ranking leistet nichts,")
        print("        was Wuerfeln nicht auch schafft.")
    elif perzentil < 97.5:
        print("BEFUND: Oberes Ende, aber innerhalb dessen, was Konzentration")
        print("        allein erklaeren kann.")
    else:
        print("BEFUND: Deutlich ausserhalb der Zufallsverteilung.")


# ----------------------------------------------------------------------

def selftest():
    print("Selbsttest\n" + "-" * 68)
    rng = np.random.default_rng(5)
    n_m, n_a = 180, 60
    idx = pd.date_range("2010-01-31", periods=n_m, freq="ME")
    cols = [f"A{i:02d}" for i in range(n_a)]

    zuf = pd.DataFrame(rng.normal(0.006, 0.05, (n_m, n_a)), index=idx, columns=cols)
    px = 100 * (1 + zuf).cumprod()
    ls = backtest(px, 10, 6, 1, 0.0, "ls")
    print(f"[1] Long-Short auf Zufallsdaten: {cagr(ls):+.2%} p.a. (soll nahe 0)")

    drift = rng.normal(0.0, 0.012, n_a)
    mit = pd.DataFrame(rng.normal(0.004, 0.04, (n_m, n_a)) + drift, index=idx, columns=cols)
    px2 = 100 * (1 + mit).cumprod()
    ls2 = backtest(px2, 10, 6, 1, 0.0, "ls")
    print(f"[2] Long-Short mit echtem Effekt: {cagr(ls2):+.2%} p.a. (soll > 0)")

    a, b, t = ols_alpha(backtest(px2, 10, 6, 1, 0.0), equal_weight(px2))
    print(f"[3] Regression laeuft: alpha {a:+.2%}, beta {b:.2f}, t={t:.2f}")

    halb = n_m // 2
    man = px2.copy()
    man.iloc[halb:] *= np.linspace(1.0, 3.0, n_m - halb)[:, None]
    kein_lookahead = np.allclose(
        momentum_scores(px2, 6, 1).iloc[:halb].fillna(0).values,
        momentum_scores(man, 6, 1).iloc[:halb].fillna(0).values)
    print(f"[4] Kein Look-Ahead: {'OK' if kein_lookahead else 'FEHLER'}")

    ok = kein_lookahead and cagr(ls2) > cagr(ls)
    print("-" * 68)
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
    p.add_argument("--split", default="2021-01-01",
                   help="Trennpunkt fuer die Teilperioden")
    p.add_argument("--sims", type=int, default=500)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.selftest:
        return selftest()

    log.info("Lade Kurse ...")
    px = load_prices(UNIVERSE + [BENCHMARK], args.start)
    monthly = px.resample("ME").last()
    uni = monthly.drop(columns=[BENCHMARK], errors="ignore")
    log.info("%s bis %s, %d Werte", monthly.index[0].date(),
             monthly.index[-1].date(), uni.shape[1])

    test_ausschluss(uni, args)
    test_teilperioden(uni, args)
    test_longshort(uni, args)
    test_beta(uni, args)
    test_zufall(uni, args, n_sim=args.sims)

    print("\n" + "=" * 68)
    print("So liest du das Gesamtbild:")
    print("  Ueberlebt der Effekt Test 1 UND 2 UND 3, ist er ernst zu nehmen.")
    print("  Faellt er in einem davon um, gibt es keinen Live-Bot.")
    print("=" * 68 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
