#!/usr/bin/env python3
"""
relative_strength_backtest.py

Cross-Sectional-Relative-Strength-Backtest auf Monatsbasis.

Idee
----
Keine Prognose pro Einzelwert, sondern ein Ranking:
Aus einem Universum von ~100 Aktien werden monatlich die TOP_N Werte
mit der hoechsten Rendite der letzten LOOKBACK_MONTHS Monate gekauft -
wobei die letzten SKIP_MONTHS Monate bewusst uebersprungen werden
(kurzfristige Umkehr). Gleichgewichtet, Haltedauer ein Monat.

Verglichen wird gegen Buy & Hold auf den Benchmark (SPY).

WICHTIGE EINSCHRAENKUNG (Survivorship Bias)
-------------------------------------------
UNIVERSE enthaelt die HEUTIGEN Index-Mitglieder. Wer heute im S&P 100
steht, hat die letzten zehn Jahre ueberlebt - Pleiten und Absteiger
fehlen komplett. Das schoenigt JEDES Ergebnis, auch das des Benchmarks,
aber die Strategie profitiert staerker davon.
Ein Ergebnis, das hier nur knapp vor Buy & Hold liegt, ist in
Wirklichkeit vermutlich schlechter. Erst ein deutlicher Abstand
ist ueberhaupt ein Signal.

Aufrufe
-------
    python3 relative_strength_backtest.py                 # echter Lauf (braucht Netz)
    python3 relative_strength_backtest.py --selftest      # Logikpruefung ohne Netz
    python3 relative_strength_backtest.py --top-n 15 --lookback 12 --costs 0.001
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------

BENCHMARK = "SPY"
START = "2014-01-01"
END = None  # None = bis heute

LOOKBACK_MONTHS = 6      # Rendite-Fenster fuer das Ranking
SKIP_MONTHS = 1          # juengster Monat wird uebersprungen
TOP_N = 10               # Anzahl gehaltener Positionen
COST_PER_SIDE = 0.0005   # 0.05% je Kauf/Verkauf (Alpaca: 0, Puffer fuer Spread)

# S&P-100-naehe Auswahl (Stand 2026) - bewusst als feste Liste, damit der
# Lauf reproduzierbar ist und nicht von einer Web-Abfrage abhaengt.
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

log = logging.getLogger("rs_backtest")


# ----------------------------------------------------------------------
# Datenbeschaffung
# ----------------------------------------------------------------------

def load_prices(tickers, start, end):
    """Taegliche adjustierte Schlusskurse als DataFrame (Spalte = Ticker).

    Robuste Spaltenbehandlung: yfinance liefert je nach Version und Anzahl
    der Ticker mal einen MultiIndex (Feld/Ticker), mal (Ticker/Feld), mal
    flache Spalten. Genau daran ist stock_predictor.py damals gescheitert.
    """
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
    )
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
            raise RuntimeError(f"Keine Close-Spalte gefunden: {raw.columns[:5]}")
    else:
        if "Close" not in raw.columns:
            raise RuntimeError(f"Keine Close-Spalte gefunden: {list(raw.columns)}")
        px = raw[["Close"]].copy()
        px.columns = [tickers[0] if isinstance(tickers, list) else tickers]

    px = px.sort_index()
    leer = [c for c in px.columns if px[c].notna().sum() == 0]
    if leer:
        log.warning("Ohne Daten, werden ignoriert: %s", ", ".join(leer))
        px = px.drop(columns=leer)
    return px


def to_monthly(px):
    """Letzter Handelstag je Monat."""
    return px.resample("ME").last()


# ----------------------------------------------------------------------
# Kern der Strategie
# ----------------------------------------------------------------------

def momentum_scores(monthly, lookback, skip):
    """Rendite von t-(skip+lookback) bis t-skip, je Monatsende t.

    Durch die Verschiebung um `skip` fliesst NUR Information ein, die am
    Rebalancing-Tag t bereits vorlag - kein Blick in die Zukunft.
    """
    alt = monthly.shift(skip + lookback)
    neu = monthly.shift(skip)
    return neu / alt - 1.0


def run_backtest(monthly_px, bench_px, top_n, lookback, skip, cost_per_side):
    """Monatliches Rebalancing. Gibt Zeitreihe der Portfoliorenditen zurueck."""
    mom = momentum_scores(monthly_px, lookback, skip)
    fwd = monthly_px.pct_change(fill_method=None).shift(-1)  # Rendite von t nach t+1

    dates, rets, holdings_hist, turnover_hist = [], [], [], []
    prev_holdings = set()

    for t in mom.index:
        scores = mom.loc[t].dropna()
        # nur Werte, fuer die auch die Folgemonats-Rendite existiert
        scores = scores[scores.index.isin(fwd.columns)]
        scores = scores[fwd.loc[t, scores.index].notna()]
        if len(scores) < top_n:
            continue

        picks = list(scores.sort_values(ascending=False).head(top_n).index)
        brutto = float(fwd.loc[t, picks].mean())

        neu = set(picks)
        wechsel = len(neu.symmetric_difference(prev_holdings))
        turnover = wechsel / (2 * top_n)
        kosten = turnover * 2 * cost_per_side
        prev_holdings = neu

        dates.append(t)
        rets.append(brutto - kosten)
        holdings_hist.append(picks)
        turnover_hist.append(turnover)

    strat = pd.Series(rets, index=pd.DatetimeIndex(dates), name="strategie")

    bench = None
    if bench_px is not None:
        b = bench_px.pct_change(fill_method=None).shift(-1)
        bench = b.reindex(strat.index).dropna()
        bench.name = "benchmark"

    return strat, bench, holdings_hist, turnover_hist


# ----------------------------------------------------------------------
# Kennzahlen
# ----------------------------------------------------------------------

def metrics(r):
    """Kennzahlen aus einer Reihe von Monatsrenditen."""
    r = r.dropna()
    if len(r) == 0:
        return {}
    equity = (1 + r).cumprod()
    jahre = len(r) / 12.0
    cagr = equity.iloc[-1] ** (1 / jahre) - 1 if jahre > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol if vol > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    return {
        "Monate": len(r),
        "Gesamt": equity.iloc[-1] - 1,
        "CAGR": cagr,
        "Vol p.a.": vol,
        "Sharpe": sharpe,
        "Max Drawdown": dd,
        "Gewinnmonate": (r > 0).mean(),
        "Bester Monat": r.max(),
        "Schlechtester Monat": r.min(),
    }


def print_report(strat, bench, turnover_hist, holdings_hist):
    m_s = metrics(strat)
    m_b = metrics(bench) if bench is not None and len(bench) else {}

    print("\n" + "=" * 62)
    print("ERGEBNIS")
    print("=" * 62)
    kopf = f"{'':<22}{'Strategie':>18}{'Benchmark':>18}"
    print(kopf)
    print("-" * 62)
    for k in m_s:
        v_s = m_s[k]
        v_b = m_b.get(k, np.nan)
        if k == "Monate":
            print(f"{k:<22}{v_s:>18d}{(int(v_b) if pd.notna(v_b) else 0):>18d}")
        else:
            fs = f"{v_s:>17.2%}" if abs(v_s) < 100 else f"{v_s:>17.2f}"
            fb = f"{v_b:>17.2%}" if pd.notna(v_b) else f"{'-':>17}"
            if k == "Sharpe":
                fs, fb = f"{v_s:>17.2f}", (f"{v_b:>17.2f}" if pd.notna(v_b) else f"{'-':>17}")
            print(f"{k:<22}{fs} {fb} ")

    if turnover_hist:
        print(f"\nDurchschnittlicher Umschlag pro Monat: {np.mean(turnover_hist):.1%}")
    if holdings_hist:
        print(f"Letzte Auswahl: {', '.join(holdings_hist[-1])}")

    if m_b:
        diff = m_s["CAGR"] - m_b["CAGR"]
        print("\nEINORDNUNG")
        print("-" * 62)
        print(f"CAGR-Differenz zum Benchmark: {diff:+.2%} p.a.")
        if diff <= 0:
            print("Kein Vorteil. Strategie verwerfen, nicht nachjustieren.")
        elif diff < 0.02:
            print("Vorsprung kleiner als der Survivorship-Bias-Effekt (s. Docstring).")
            print("Das ist KEIN belastbarer Edge.")
        else:
            print("Vorsprung vorhanden. Naechster Schritt: anderes Universum,")
            print("anderer Zeitraum, andere Parameter - haelt er da auch?")
        n = m_s["Monate"]
        se = strat.std(ddof=1) * np.sqrt(12) / np.sqrt(n / 12) if n > 12 else np.nan
        if pd.notna(se):
            print(f"Standardfehler der CAGR-Schaetzung: rund {se:.2%} - alles,")
            print("was kleiner ist, ist Rauschen.")
    print()


# ----------------------------------------------------------------------
# Selbsttest ohne Netzwerk
# ----------------------------------------------------------------------

def selftest():
    """Prueft die Mechanik gegen synthetische Daten mit bekannter Antwort."""
    print("Selbsttest\n" + "-" * 62)
    rng = np.random.default_rng(42)
    n_assets, n_months = 40, 240
    idx = pd.date_range("2006-01-31", periods=n_months, freq="ME")
    cols = [f"A{i:02d}" for i in range(n_assets)]

    # 1) Kein Momentum im Datensatz -> im Mittel darf kein Vorteil entstehen.
    #    Ueber viele Durchlaeufe, weil ein einzelner Lauf reines Rauschen ist.
    #    Die Streuung hier IST die Messgrenze: alles, was im echten Backtest
    #    kleiner ausfaellt, ist nicht von Zufall zu unterscheiden.
    einzel = []
    for seed in range(30):
        r = np.random.default_rng(1000 + seed)
        zuf = pd.DataFrame(r.normal(0.007, 0.06, (n_months, n_assets)),
                           index=idx, columns=cols)
        px_z = 100 * (1 + zuf).cumprod()
        s, _, _, _ = run_backtest(px_z, None, TOP_N, LOOKBACK_MONTHS, SKIP_MONTHS, 0.0)
        g = zuf.mean(axis=1).shift(-1).reindex(s.index).dropna()
        einzel.append((s.reindex(g.index).mean() - g.mean()) * 12)
    diff1, streu1 = float(np.mean(einzel)), float(np.std(einzel, ddof=1))
    print(f"[1] Ohne echtes Momentum: Vorteil {diff1:+.2%} p.a. im Mittel "
          f"(Streuung {streu1:.2%})")
    print(f"    -> Rauschgrenze: ein Vorsprung unter rund {2*streu1:.1%} p.a. "
          f"beweist nichts.")

    # 2) Kuenstlich eingebautes Momentum -> Strategie MUSS es finden
    drift = rng.normal(0.0, 0.010, n_assets)          # dauerhafter Qualitaetsunterschied
    mit_mom = pd.DataFrame(rng.normal(0.005, 0.05, (n_months, n_assets)) + drift,
                           index=idx, columns=cols)
    px_mom = 100 * (1 + mit_mom).cumprod()
    s2, _, _, _ = run_backtest(px_mom, None, TOP_N, LOOKBACK_MONTHS, SKIP_MONTHS, 0.0)
    gleich2 = mit_mom.mean(axis=1).shift(-1).reindex(s2.index).dropna()
    diff2 = s2.reindex(gleich2.index).mean() - gleich2.mean()
    print(f"[2] Mit eingebautem Momentum: Vorteil {diff2*12:+.2%} p.a. (soll klar > 0 sein)")

    # 3) Kein Blick in die Zukunft: Kurse NACH einem Stichtag veraendern
    #    darf die Auswahl an diesem Stichtag nicht beeinflussen.
    halb = n_months // 2
    manipuliert = px_mom.copy()
    manipuliert.iloc[halb:] *= np.linspace(1.0, 3.0, n_months - halb)[:, None]
    m_orig = momentum_scores(px_mom, LOOKBACK_MONTHS, SKIP_MONTHS).iloc[:halb]
    m_neu = momentum_scores(manipuliert, LOOKBACK_MONTHS, SKIP_MONTHS).iloc[:halb]
    identisch = np.allclose(m_orig.fillna(0).values, m_neu.fillna(0).values)
    print(f"[3] Kein Look-Ahead in den Scores: {'OK' if identisch else 'FEHLER'}")

    # 4) Kosten muessen die Rendite senken, nicht erhoehen
    s3, _, _, _ = run_backtest(px_mom, None, TOP_N, LOOKBACK_MONTHS, SKIP_MONTHS, 0.002)
    teurer = s3.mean() < s2.mean()
    print(f"[4] Kosten senken die Rendite: {'OK' if teurer else 'FEHLER'}")

    ok = identisch and teurer and diff2 > diff1
    print("-" * 62)
    print("Selbsttest bestanden." if ok else "Selbsttest FEHLGESCHLAGEN.")
    return 0 if ok else 1


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true", help="Logikpruefung ohne Netzwerk")
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--lookback", type=int, default=LOOKBACK_MONTHS)
    p.add_argument("--skip", type=int, default=SKIP_MONTHS)
    p.add_argument("--costs", type=float, default=COST_PER_SIDE,
                   help="Kosten je Seite, z.B. 0.0005 = 0.05%%")
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--csv", default=None, help="Monatsrenditen hierhin schreiben")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.selftest:
        return selftest()

    log.info("Lade Kurse fuer %d Werte plus Benchmark ...", len(UNIVERSE))
    px = load_prices(UNIVERSE + [BENCHMARK], args.start, args.end)
    monthly = to_monthly(px)

    bench_m = monthly[BENCHMARK] if BENCHMARK in monthly.columns else None
    universum = monthly.drop(columns=[BENCHMARK], errors="ignore")
    log.info("Daten: %s bis %s, %d Werte, %d Monate",
             monthly.index[0].date(), monthly.index[-1].date(),
             universum.shape[1], len(monthly))

    strat, bench, holdings, turnover = run_backtest(
        universum, bench_m, args.top_n, args.lookback, args.skip, args.costs
    )
    if len(strat) == 0:
        log.error("Keine Rebalancing-Termine - Zeitraum zu kurz fuer das Fenster?")
        return 1

    print_report(strat, bench, turnover, holdings)

    if args.csv:
        out = pd.DataFrame({"strategie": strat})
        if bench is not None:
            out["benchmark"] = bench
        out.to_csv(args.csv)
        log.info("Geschrieben: %s", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
