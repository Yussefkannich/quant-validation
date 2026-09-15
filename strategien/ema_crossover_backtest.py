#!/usr/bin/env python3
"""
ema_crossover_backtest.py - EMA-Crossover, durch dieselbe Pruefkette
wie Ichimoku und Relative Strength.

Die Regeln
----------
alpha      = 2 / (N + 1)
EMA_t      = Kurs_t * alpha + EMA_(t-1) * (1 - alpha)
             (pandas: close.ewm(span=N, adjust=False).mean())

Long-Signal:  EMA(schnell) > EMA(langsam)   -> investiert
sonst:                                      -> nicht investiert
Standard: 12/26 (dieselben Perioden wie im MACD).

Die Signalentscheidung faellt mit dem Schlusskurs von Tag t und gilt fuer
die Periode t -> t+1. Wer am selben Schlusskurs entscheidet UND handelt,
rechnet sich einen Tag Vorsprung ein, den es in der Praxis nicht gibt.

Warum dieser Test, obwohl SMA-Crossover schon verworfen ist
-----------------------------------------------------------
Der EMA reagiert schneller als der SMA. Das beliebte Argument lautet:
"Der SMA ist zu traege, mit dem EMA klappt es." Teil C vergleicht
deshalb EMA und SMA mit denselben Perioden auf denselben Daten.
Liegen beide gleichauf, war die Traegheit nicht das Problem.

Die Pruefkette
--------------
  A) Gegen Buy & Hold je Wert und im Mittel
  B) Gegen ZUFALLSTIMING mit gleicher Marktzeit und Handelszahl
     (der entscheidende Test fuer jeden Trendfilter)
  C) Parameter-Gitter, jeweils EMA und SMA nebeneinander
  D) Teilperioden

Aufruf
------
    python3 ema_crossover_backtest.py --selftest
    python3 ema_crossover_backtest.py
    python3 ema_crossover_backtest.py --fast 20 --slow 50
    python3 ema_crossover_backtest.py --tickers BTC-USD,ETH-USD --kosten 0.0026
"""

import argparse
import logging
import math
import sys

import numpy as np
import pandas as pd

STANDARD_TICKER = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "AAPL", "MSFT",
                   "NVDA", "XOM", "JPM", "KO"]
START = "2016-01-01"
KOSTEN = 0.0010          # 0.10% je Positionswechsel (Ein- oder Ausstieg)
FAST, SLOW = 12, 26

log = logging.getLogger("ema")


# ----------------------------------------------------------------------

def lade(tickers, start, end=None):
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, threads=True)
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance lieferte keine Daten")
    daten = {}
    if isinstance(raw.columns, pd.MultiIndex):
        felder = set(raw.columns.get_level_values(0))
        for tk in tickers:
            try:
                if "Close" in felder:
                    s = raw["Close"][tk]
                else:
                    s = raw[tk]["Close"]
            except KeyError:
                log.warning("Keine Daten fuer %s", tk)
                continue
            df = pd.DataFrame({"Close": s}).dropna()
            if len(df) > 300:
                daten[tk] = df
            else:
                log.warning("Zu wenig Daten fuer %s", tk)
    else:
        daten[tickers[0]] = raw[["Close"]].dropna()
    return daten


# ----------------------------------------------------------------------
# Indikator und Signal
# ----------------------------------------------------------------------

def ema(close, n):
    """Exponentiell gleitender Durchschnitt nach der Lehrbuchformel.
    adjust=False = rekursiv: EMA_t = a*Kurs_t + (1-a)*EMA_(t-1)."""
    return close.ewm(span=n, adjust=False).mean()


def sma(close, n):
    return close.rolling(n).mean()


def signal(df, fast=FAST, slow=SLOW, art="ema"):
    """1 = investiert, 0 = nicht investiert. Nutzt nur Daten bis einschliesslich t."""
    c = df["Close"]
    f = ema if art == "ema" else sma
    schnell, langsam = f(c, fast), f(c, slow)
    pos = (schnell > langsam).astype(float)
    # Einschwingphase: der EMA startet am ersten Kurs und braucht einige
    # Perioden, bis er aussagekraeftig ist. Die ersten 'slow' Tage zaehlen
    # nicht - fuer EMA und SMA gleich, damit der Vergleich fair bleibt.
    pos.iloc[:slow] = 0.0
    return pos


def rendite_serie(df, pos, kosten=KOSTEN):
    """Rendite der Strategie. Position an t gilt fuer die Periode t -> t+1."""
    fwd = df["Close"].pct_change(fill_method=None).shift(-1)
    wechsel = pos.diff().abs().fillna(pos.abs())
    r = pos * fwd - wechsel * kosten
    return r.dropna()


# ----------------------------------------------------------------------
# Kennzahlen
# ----------------------------------------------------------------------

def cagr(r, pro_jahr=252):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).prod() ** (pro_jahr / len(r)) - 1)


def sharpe(r, pro_jahr=252):
    r = r.dropna()
    v = r.std(ddof=1) * math.sqrt(pro_jahr)
    return float(r.mean() * pro_jahr / v) if v > 0 else np.nan


def max_dd(r):
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def norm_p(t):
    return math.erfc(abs(t) / math.sqrt(2))


# ----------------------------------------------------------------------
# Zufallstiming mit gleicher Marktzeit
# ----------------------------------------------------------------------

def zufallstiming(df, marktzeit, n_handel, rng):
    """Positionsreihe mit derselben Marktzeit und aehnlich vielen Wechseln,
    aber zufaellig verteilt - die faire Nullhypothese fuer Trendfilter."""
    n = len(df)
    pos = np.zeros(n)
    bloecke = max(1, int(n_handel / 2))
    ziel = int(round(marktzeit * n))
    if ziel <= 0:
        return pd.Series(pos, index=df.index)
    laenge = max(1, ziel // bloecke)
    gesetzt = 0
    versuche = 0
    while gesetzt < ziel and versuche < bloecke * 20:
        versuche += 1
        start = rng.integers(0, max(1, n - laenge))
        ende = min(n, start + laenge)
        neu = int((pos[start:ende] == 0).sum())
        if neu == 0:
            continue
        pos[start:ende] = 1.0
        gesetzt += neu
    return pd.Series(pos, index=df.index)


# ----------------------------------------------------------------------
# Auswertung
# ----------------------------------------------------------------------

def mittel_gegen_bh(daten, fast, slow, art, kosten):
    rs, bs, mzs = [], [], []
    for tk, df in daten.items():
        p = signal(df, fast, slow, art)
        rr = rendite_serie(df, p, kosten).iloc[slow:]
        bb = df["Close"].pct_change(fill_method=None).shift(-1).reindex(rr.index).dropna()
        rs.append(rr.reindex(bb.index).rename(tk))
        bs.append(bb.rename(tk))
        mzs.append(float(p.reindex(rr.index).mean()))
    sm = pd.concat(rs, axis=1).mean(axis=1).dropna()
    bm = pd.concat(bs, axis=1).mean(axis=1).dropna()
    return sm, bm, float(np.mean(mzs))


def auswerten(daten, args):
    name = f"EMA {args.fast}/{args.slow}"
    print("\n" + "=" * 76)
    print(f"A) {name} GEGEN BUY & HOLD   (Kosten {args.kosten:.2%} je Wechsel)")
    print("=" * 76)
    print(f"{'Wert':<10}{'EMA-Cross':>12}{'Buy&Hold':>12}{'Differenz':>12}"
          f"{'Sharpe E':>10}{'Sharpe BH':>11}{'Marktzeit':>11}{'Wechsel':>9}")
    print("-" * 87)

    strat_alle, bh_alle, zeilen = [], [], []
    for tk, df in daten.items():
        pos = signal(df, args.fast, args.slow)
        r = rendite_serie(df, pos, args.kosten).iloc[args.slow:]
        bh = df["Close"].pct_change(fill_method=None).shift(-1).reindex(r.index).dropna()
        r = r.reindex(bh.index)
        mz = float(pos.reindex(r.index).mean())
        n_handel = int(pos.diff().abs().sum())
        zeilen.append((tk, r, mz, n_handel))
        strat_alle.append(r.rename(tk))
        bh_alle.append(bh.rename(tk))
        print(f"{tk:<10}{cagr(r):>11.2%}{cagr(bh):>12.2%}"
              f"{cagr(r)-cagr(bh):>+12.2%}{sharpe(r):>10.2f}"
              f"{sharpe(bh):>11.2f}{mz:>11.1%}{n_handel:>9d}")

    s_mittel = pd.concat(strat_alle, axis=1).mean(axis=1).dropna()
    b_mittel = pd.concat(bh_alle, axis=1).mean(axis=1).dropna()
    print("-" * 87)
    print(f"{'MITTEL':<10}{cagr(s_mittel):>11.2%}{cagr(b_mittel):>12.2%}"
          f"{cagr(s_mittel)-cagr(b_mittel):>+12.2%}{sharpe(s_mittel):>10.2f}"
          f"{sharpe(b_mittel):>11.2f}")
    d = (s_mittel - b_mittel).dropna()
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 30 else np.nan
    print(f"\nt-Wert der Differenz: {t:.2f}   p={norm_p(t):.3f}")
    print(f"Max Drawdown   EMA-Cross {max_dd(s_mittel):.1%}   "
          f"Buy&Hold {max_dd(b_mittel):.1%}")
    print("Hinweis: ein kleinerer Drawdown allein ist kein Verdienst - weniger")
    print("Marktzeit senkt ihn automatisch. Das prueft Teil B.")

    print("\n" + "=" * 76)
    print(f"B) GEGEN ZUFALLSTIMING ({args.sims} Durchlaeufe, gleiche Marktzeit)")
    print("=" * 76)
    rng = np.random.default_rng(args.seed)
    print(f"{'Wert':<10}{'EMA-Cross':>12}{'Zufall Mittel':>15}"
          f"{'Zufall 95%':>13}{'Perzentil':>12}")
    print("-" * 76)
    perzentile = []
    for tk, r, mz, n_handel in zeilen:
        df = daten[tk]
        werte = []
        for _ in range(args.sims):
            zp = zufallstiming(df, mz, max(2, n_handel), rng)
            zr = rendite_serie(df, zp, args.kosten).reindex(r.index).dropna()
            werte.append(cagr(zr))
        v = np.array([x for x in werte if not np.isnan(x)])
        pz = float((v < cagr(r)).mean() * 100)
        perzentile.append(pz)
        print(f"{tk:<10}{cagr(r):>11.2%}{v.mean():>15.2%}"
              f"{np.percentile(v, 95):>13.2%}{pz:>11.1f}.")
    print("-" * 76)
    mp = float(np.mean(perzentile))
    print(f"Mittleres Perzentil ueber alle Werte: {mp:.1f}   (Huerde: 90)")
    if mp < 75:
        print("BEFUND: Zufaelliges Ein- und Aussteigen mit derselben Marktzeit")
        print("        erreicht dasselbe. Das Signal leistet nichts.")
    elif mp < 90:
        print("BEFUND: Leicht ueber dem Zufall, aber innerhalb der Streuung.")
    else:
        print("BEFUND: Deutlich ueber Zufallstiming. Weiter mit C und D.")

    print("\n" + "=" * 76)
    print("C) PARAMETER-GITTER: EMA GEGEN SMA (Differenz zu Buy & Hold p.a.)")
    print("=" * 76)
    varianten = [(5, 20), (9, 21), (12, 26), (20, 50), (50, 200), (10, 100)]
    print(f"{'Schnell/Langsam':<18}{'EMA':>12}{'SMA':>12}{'EMA-SMA':>12}"
          f"{'Marktzeit EMA':>16}")
    for fa, sl in varianten:
        se, be, mze = mittel_gegen_bh(daten, fa, sl, "ema", args.kosten)
        ss, bs, _ = mittel_gegen_bh(daten, fa, sl, "sma", args.kosten)
        de, ds = cagr(se) - cagr(be), cagr(ss) - cagr(bs)
        marke = "  <- gewaehlt" if (fa, sl) == (args.fast, args.slow) else ""
        print(f"{f'{fa}/{sl}':<18}{de:>+11.2%}{ds:>+12.2%}{de-ds:>+12.2%}"
              f"{mze:>16.1%}{marke}")
    print("-" * 76)
    print("Liegen EMA und SMA gleichauf, war die Traegheit des SMA nicht")
    print("das Problem. Ist nur eine Zeile positiv, ist das Glueck, kein Signal.")

    print("\n" + "=" * 76)
    print("D) TEILPERIODEN")
    print("=" * 76)
    grenze = pd.Timestamp(args.split)
    for label, maske in [(f"bis {args.split}", s_mittel.index < grenze),
                         (f"ab {args.split}", s_mittel.index >= grenze)]:
        st = s_mittel[maske]
        bt = b_mittel.reindex(st.index).dropna()
        st = st.reindex(bt.index)
        if len(st) < 60:
            continue
        dd = (st - bt).dropna()
        tt = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd)))
        print(f"{label:<18}EMA-Cross {cagr(st):>8.2%}   Buy&Hold {cagr(bt):>8.2%}"
              f"   Diff {cagr(st)-cagr(bt):>+8.2%}   t={tt:>5.2f}")
    print()


# ----------------------------------------------------------------------

def selbsttest():
    print("Selbsttest\n" + "-" * 70)
    fehler = 0

    # [1] Formel gegen Handrechnung (Beispiel aus der Erklaerung, N=3)
    beispiel = pd.Series([10.0, 12.0, 14.0, 11.0])
    ok = np.allclose(ema(beispiel, 3).values, [10.0, 11.0, 12.5, 11.75])
    print(f"[1] EMA stimmt mit Handrechnung (10/11/12.5/11.75): {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [2] Formel gegen eigene Rekursion auf Zufallsdaten
    rng = np.random.default_rng(4)
    n = 1500
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    c = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n)), index=idx)
    df = pd.DataFrame({"Close": c})
    a = 2 / (FAST + 1)
    hand = [c.iloc[0]]
    for x in c.iloc[1:]:
        hand.append(a * x + (1 - a) * hand[-1])
    ok = np.allclose(ema(c, FAST).values, hand)
    print(f"[2] EMA stimmt mit Rekursion (alpha={a:.4f}): {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    pos = signal(df)
    ok = set(pos.unique()) <= {0.0, 1.0} and pos.iloc[:SLOW].sum() == 0
    print(f"[3] Signal nur 0/1, Einschwingphase gesperrt: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    r_ohne = rendite_serie(df, pos, 0.0)
    r_mit = rendite_serie(df, pos, 0.01)
    ok = r_mit.sum() < r_ohne.sum()
    print(f"[4] Kosten senken die Rendite: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [5] Kein Blick in die Zukunft: Kurse NACH dem Stichtag veraendern
    # darf Signale VOR dem Stichtag nicht beruehren.
    halb = n // 2
    man = df.copy()
    man.iloc[halb:, 0] = man["Close"].iloc[halb:].values * np.linspace(1.0, 4.0, n - halb)
    ok = np.array_equal(signal(df).iloc[:halb].values, signal(man).iloc[:halb].values)
    print(f"[5] Kein Blick in die Zukunft: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [6] Gegenprobe: absichtlich falscher EMA (rueckwaerts gerechnet,
    # sieht also die Zukunft) MUSS vom selben Test erwischt werden.
    def falsch(d):
        cc = d["Close"][::-1]
        s = cc.ewm(span=FAST, adjust=False).mean()[::-1]
        l = cc.ewm(span=SLOW, adjust=False).mean()[::-1]
        return (s > l).astype(float)
    ok = not np.array_equal(falsch(df).iloc[:halb].values, falsch(man).iloc[:halb].values)
    print(f"[6] Falsche Variante wird erkannt: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [7] Timing: die Position von Tag t darf die Rendite t-1 -> t nicht sehen.
    # Signal "heute gestiegen" muss auf Zufallsdaten ~0 bringen,
    # die verbotene Variante (gleicher Tag) dagegen klar positiv.
    heute_hoch = (c.pct_change() > 0).astype(float)
    r_ehrlich = rendite_serie(df, heute_hoch, 0.0)
    r_betrug = heute_hoch * c.pct_change()
    ok = abs(r_ehrlich.mean()) < 0.001 and r_betrug.mean() > 0.004
    print(f"[7] Handel erst nach dem Signal ({r_ehrlich.mean():+.4%} gegen "
          f"verbotene {r_betrug.mean():+.4%} je Tag): {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    mz = float(pos.mean())
    zp = zufallstiming(df, mz, 20, rng)
    ok = abs(float(zp.mean()) - mz) < 0.15
    print(f"[8] Zufallstiming trifft Marktzeit ({float(zp.mean()):.1%} "
          f"gegen {mz:.1%}): {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    bh = df["Close"].pct_change(fill_method=None).shift(-1).reindex(r_ohne.index).dropna()
    print(f"\n    Auf Zufallsdaten: EMA-Cross {cagr(r_ohne.reindex(bh.index)):+.2%} "
          f"gegen Buy&Hold {cagr(bh):+.2%}, Marktzeit {mz:.0%}, "
          f"{int(pos.diff().abs().sum())} Wechsel")
    print("    (Ein Unterschied hier ist Rauschen, kein Befund.)")

    print("-" * 70)
    print("Selbsttest bestanden." if fehler == 0
          else f"Selbsttest FEHLGESCHLAGEN ({fehler} Punkte).")
    return 0 if fehler == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--tickers", default=",".join(STANDARD_TICKER))
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=None)
    p.add_argument("--fast", type=int, default=FAST)
    p.add_argument("--slow", type=int, default=SLOW)
    p.add_argument("--kosten", type=float, default=KOSTEN)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--split", default="2022-01-01")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.selftest:
        return selbsttest()
    if args.fast >= args.slow:
        log.error("--fast muss kleiner als --slow sein")
        return 1

    tickers = [x.strip() for x in args.tickers.split(",") if x.strip()]
    log.info("Lade %d Werte ...", len(tickers))
    daten = lade(tickers, args.start, args.end)
    if not daten:
        log.error("Keine verwertbaren Daten")
        return 1
    log.info("%d Werte geladen", len(daten))
    auswerten(daten, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
