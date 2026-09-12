#!/usr/bin/env python3
"""
ichimoku_backtest.py - Ichimoku Kinko Hyo, durch dieselbe Pruefkette
wie Relative Strength.

Die Regeln
----------
Tenkan-sen  = (Hoch_9  + Tief_9 ) / 2
Kijun-sen   = (Hoch_26 + Tief_26) / 2
Senkou A    = (Tenkan + Kijun) / 2, um 26 Perioden VORGEZEICHNET
Senkou B    = (Hoch_52 + Tief_52) / 2, um 26 Perioden VORGEZEICHNET
Chikou      = Schlusskurs, um 26 Perioden ZURUECKgezeichnet
Kumo (Wolke) = Bereich zwischen Senkou A und B

Long-Signal (Standardlesart):
    Kurs ueber der Wolke  UND  Tenkan ueber Kijun  UND
    Kurs ueber dem Kurs von vor 26 Perioden (Chikou-Bestaetigung)

Die Falle, auf die dieses Skript besonders achtet
-------------------------------------------------
"26 Perioden vorgezeichnet" klingt nach Zukunft, ist aber nur Darstellung:
Die Wolke an Position t stammt aus Daten von t-26. Wer das falsch
implementiert - etwa shift(-26) statt shift(+26) - baut sich einen
Blick in die Zukunft ein und bekommt traumhafte Ergebnisse, die
niemals handelbar waren. Test [4] im Selbsttest prueft genau das.

Die Pruefkette
--------------
  A) Gegen Buy & Hold je Wert und im Mittel
  B) Gegen ZUFALLSTIMING mit gleicher Marktzeit und Handelszahl.
     Das ist der entscheidende Test: Ein Trendfilter ist zeitweise
     nicht investiert. In einem steigenden Markt sieht schon reines
     Wuerfeln mit derselben Marktzeit gut aus. Nur wer das schlaegt,
     hat ein Signal.
  C) Parameter-Gitter - haelt der Effekt jenseits von 9/26/52?
     (Diese Zahlen stammen aus der japanischen Sechs-Tage-Woche der
     1930er Jahre und haben heute keine Begruendung ausser Tradition.)
  D) Teilperioden

Aufruf
------
    python3 ichimoku_backtest.py --selftest
    python3 ichimoku_backtest.py
    python3 ichimoku_backtest.py --tickers BTC-USD,ETH-USD --start 2018-01-01
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
TENKAN, KIJUN, SENKOU_B = 9, 26, 52

log = logging.getLogger("ichimoku")


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
                    df = pd.DataFrame({
                        "High": raw["High"][tk], "Low": raw["Low"][tk],
                        "Close": raw["Close"][tk]})
                else:
                    df = raw[tk][["High", "Low", "Close"]].copy()
            except KeyError:
                log.warning("Keine Daten fuer %s", tk)
                continue
            df = df.dropna()
            if len(df) > 200:
                daten[tk] = df
            else:
                log.warning("Zu wenig Daten fuer %s", tk)
    else:
        df = raw[["High", "Low", "Close"]].dropna()
        daten[tickers[0]] = df
    return daten


# ----------------------------------------------------------------------
# Indikator und Signal
# ----------------------------------------------------------------------

def ichimoku(df, tenkan=TENKAN, kijun=KIJUN, senkou_b=SENKOU_B):
    h, l, c = df["High"], df["Low"], df["Close"]
    t = (h.rolling(tenkan).max() + l.rolling(tenkan).min()) / 2
    k = (h.rolling(kijun).max() + l.rolling(kijun).min()) / 2
    # shift(+kijun) verschiebt NACH VORNE: an Position t stehen Werte,
    # die aus Daten von t-kijun gebildet wurden. Kein Blick nach vorn.
    span_a = ((t + k) / 2).shift(kijun)
    span_b = ((h.rolling(senkou_b).max() + l.rolling(senkou_b).min()) / 2).shift(kijun)
    return pd.DataFrame({"tenkan": t, "kijun": k,
                         "span_a": span_a, "span_b": span_b}, index=df.index)


def signal(df, tenkan=TENKAN, kijun=KIJUN, senkou_b=SENKOU_B):
    """1 = investiert, 0 = nicht investiert. Nutzt nur Daten bis einschliesslich t."""
    ind = ichimoku(df, tenkan, kijun, senkou_b)
    c = df["Close"]
    wolke_oben = ind[["span_a", "span_b"]].max(axis=1)
    ueber_wolke = c > wolke_oben
    tk_kreuz = ind["tenkan"] > ind["kijun"]
    chikou = c > c.shift(kijun)
    return (ueber_wolke & tk_kreuz & chikou).astype(float)


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

def zufallstiming(df, marktzeit, n_handel, rng, kosten=KOSTEN):
    """Positionsreihe mit derselben Marktzeit und aehnlich vielen Wechseln,
    aber zufaellig verteilt. Das ist die faire Nullhypothese fuer jeden
    Trendfilter: Wer nur zeitweise investiert ist, sieht in einem
    steigenden Markt automatisch anders aus als Buy & Hold."""
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

def auswerten(daten, args):
    print("\n" + "=" * 76)
    print("A) ICHIMOKU GEGEN BUY & HOLD")
    print("=" * 76)
    print(f"{'Wert':<10}{'Ichimoku':>12}{'Buy&Hold':>12}{'Differenz':>12}"
          f"{'Sharpe I':>10}{'Sharpe BH':>11}{'Marktzeit':>11}")
    print("-" * 76)

    strat_alle, bh_alle, zeilen = [], [], []
    for tk, df in daten.items():
        pos = signal(df, args.tenkan, args.kijun, args.senkou_b)
        r = rendite_serie(df, pos, args.kosten)
        bh = df["Close"].pct_change(fill_method=None).shift(-1).reindex(r.index).dropna()
        r = r.reindex(bh.index)
        mz = float(pos.reindex(r.index).mean())
        n_handel = int(pos.diff().abs().sum())
        zeilen.append((tk, r, bh, mz, n_handel))
        strat_alle.append(r.rename(tk))
        bh_alle.append(bh.rename(tk))
        print(f"{tk:<10}{cagr(r):>11.2%}{cagr(bh):>12.2%}"
              f"{cagr(r)-cagr(bh):>+12.2%}{sharpe(r):>10.2f}"
              f"{sharpe(bh):>11.2f}{mz:>11.1%}")

    s_mittel = pd.concat(strat_alle, axis=1).mean(axis=1).dropna()
    b_mittel = pd.concat(bh_alle, axis=1).mean(axis=1).dropna()
    print("-" * 76)
    print(f"{'MITTEL':<10}{cagr(s_mittel):>11.2%}{cagr(b_mittel):>12.2%}"
          f"{cagr(s_mittel)-cagr(b_mittel):>+12.2%}{sharpe(s_mittel):>10.2f}"
          f"{sharpe(b_mittel):>11.2f}")
    d = (s_mittel - b_mittel).dropna()
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 30 else np.nan
    print(f"\nt-Wert der Differenz: {t:.2f}   p={norm_p(t):.3f}")
    print(f"Max Drawdown   Ichimoku {max_dd(s_mittel):.1%}   "
          f"Buy&Hold {max_dd(b_mittel):.1%}")

    print("\n" + "=" * 76)
    print(f"B) GEGEN ZUFALLSTIMING ({args.sims} Durchlaeufe, gleiche Marktzeit)")
    print("=" * 76)
    rng = np.random.default_rng(args.seed)
    print(f"{'Wert':<10}{'Ichimoku':>12}{'Zufall Mittel':>15}"
          f"{'Zufall 95%':>13}{'Perzentil':>12}")
    print("-" * 76)
    perzentile = []
    for tk, r, bh, mz, n_handel in zeilen:
        df = daten[tk]
        werte = []
        for _ in range(args.sims):
            zp = zufallstiming(df, mz, max(2, n_handel), rng, args.kosten)
            zr = rendite_serie(df, zp, args.kosten).reindex(r.index).dropna()
            werte.append(cagr(zr))
        v = np.array([x for x in werte if not np.isnan(x)])
        pz = float((v < cagr(r)).mean() * 100)
        perzentile.append(pz)
        print(f"{tk:<10}{cagr(r):>11.2%}{v.mean():>15.2%}"
              f"{np.percentile(v, 95):>13.2%}{pz:>11.1f}.")
    print("-" * 76)
    print(f"Mittleres Perzentil ueber alle Werte: {np.mean(perzentile):.1f}")
    if np.mean(perzentile) < 75:
        print("BEFUND: Zufaelliges Ein- und Aussteigen mit derselben Marktzeit")
        print("        erreicht dasselbe. Das Signal leistet nichts.")
    elif np.mean(perzentile) < 90:
        print("BEFUND: Leicht ueber dem Zufall, aber innerhalb der Streuung.")
    else:
        print("BEFUND: Deutlich ueber Zufallstiming. Weiter mit C und D.")

    print("\n" + "=" * 76)
    print("C) PARAMETER-GITTER (Differenz zu Buy & Hold, im Mittel p.a.)")
    print("=" * 76)
    varianten = [(9, 26, 52), (7, 22, 44), (10, 30, 60), (12, 24, 48),
                 (5, 15, 30), (20, 60, 120)]
    print(f"{'Tenkan/Kijun/SpanB':<24}{'Differenz':>14}{'Marktzeit':>14}")
    for tn, kj, sb in varianten:
        rs, bs, mzs = [], [], []
        for tk, df in daten.items():
            p = signal(df, tn, kj, sb)
            rr = rendite_serie(df, p, args.kosten)
            bb = df["Close"].pct_change(fill_method=None).shift(-1).reindex(rr.index).dropna()
            rs.append(rr.reindex(bb.index).rename(tk))
            bs.append(bb.rename(tk))
            mzs.append(float(p.reindex(rr.index).mean()))
        sm = pd.concat(rs, axis=1).mean(axis=1).dropna()
        bm = pd.concat(bs, axis=1).mean(axis=1).dropna()
        marke = "  <- Standard" if (tn, kj, sb) == (9, 26, 52) else ""
        print(f"{f'{tn}/{kj}/{sb}':<24}{cagr(sm)-cagr(bm):>+13.2%}"
              f"{np.mean(mzs):>14.1%}{marke}")
    print("-" * 76)
    print("Sind die Standardwerte 9/26/52 nicht besonders, stammt ihr Ruf")
    print("aus Tradition, nicht aus Ueberlegenheit.")

    print("\n" + "=" * 76)
    print("D) TEILPERIODEN")
    print("=" * 76)
    grenze = pd.Timestamp(args.split)
    for name, maske in [(f"bis {args.split}", s_mittel.index < grenze),
                        (f"ab {args.split}", s_mittel.index >= grenze)]:
        st = s_mittel[maske]
        bt = b_mittel.reindex(st.index).dropna()
        st = st.reindex(bt.index)
        if len(st) < 60:
            continue
        dd = (st - bt).dropna()
        tt = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd)))
        print(f"{name:<18}Ichimoku {cagr(st):>8.2%}   Buy&Hold {cagr(bt):>8.2%}"
              f"   Diff {cagr(st)-cagr(bt):>+8.2%}   t={tt:>5.2f}")
    print()


# ----------------------------------------------------------------------

def selbsttest():
    print("Selbsttest\n" + "-" * 70)
    fehler = 0
    rng = np.random.default_rng(4)
    n = 1500
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    c_werte = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n))
    c = pd.Series(c_werte, index=idx)
    df = pd.DataFrame({"Close": c,
                       "High": c * (1 + abs(rng.normal(0, 0.004, n))),
                       "Low": c * (1 - abs(rng.normal(0, 0.004, n)))})

    ind = ichimoku(df)
    ok = ind["span_a"].iloc[:KIJUN].isna().all()
    print(f"[1] Wolke am Anfang undefiniert: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    pos = signal(df)
    ok = set(pos.dropna().unique()) <= {0.0, 1.0}
    print(f"[2] Signal ist nur 0 oder 1: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    r_ohne = rendite_serie(df, pos, 0.0)
    r_mit = rendite_serie(df, pos, 0.01)
    ok = r_mit.sum() < r_ohne.sum()
    print(f"[3] Kosten senken die Rendite: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # Der wichtigste Test: Kurse NACH einem Stichtag veraendern darf die
    # Signale VOR dem Stichtag nicht beruehren.
    halb = n // 2
    man = df.copy()
    faktor = np.linspace(1.0, 4.0, n - halb)
    for sp in ("Close", "High", "Low"):
        man.iloc[halb:, man.columns.get_loc(sp)] = man[sp].iloc[halb:].values * faktor
    s_orig = signal(df).iloc[:halb].fillna(0).values
    s_man = signal(man).iloc[:halb].fillna(0).values
    ok = np.array_equal(s_orig, s_man)
    print(f"[4] Kein Blick in die Zukunft: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # Gegenprobe: eine absichtlich falsche Implementierung MUSS auffliegen
    def falsch(d, kijun=KIJUN):
        h, l, c2 = d["High"], d["Low"], d["Close"]
        t = (h.rolling(TENKAN).max() + l.rolling(TENKAN).min()) / 2
        k = (h.rolling(kijun).max() + l.rolling(kijun).min()) / 2
        a = ((t + k) / 2).shift(-kijun)      # falsches Vorzeichen
        b = ((h.rolling(SENKOU_B).max() + l.rolling(SENKOU_B).min()) / 2).shift(-kijun)
        return (c2 > pd.concat([a, b], axis=1).max(axis=1)).astype(float)
    ok = not np.array_equal(falsch(df).iloc[:halb].fillna(0).values,
                            falsch(man).iloc[:halb].fillna(0).values)
    print(f"[5] Falsche Variante wird erkannt: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    mz = float(pos.mean())
    zp = zufallstiming(df, mz, 20, rng)
    ok = abs(float(zp.mean()) - mz) < 0.15
    print(f"[6] Zufallstiming trifft Marktzeit ({float(zp.mean()):.1%} "
          f"gegen {mz:.1%}): {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    bh = df["Close"].pct_change(fill_method=None).shift(-1).reindex(r_ohne.index).dropna()
    print(f"\n    Auf Zufallsdaten: Ichimoku {cagr(r_ohne.reindex(bh.index)):+.2%} "
          f"gegen Buy&Hold {cagr(bh):+.2%}, Marktzeit {mz:.0%}")
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
    p.add_argument("--tenkan", type=int, default=TENKAN)
    p.add_argument("--kijun", type=int, default=KIJUN)
    p.add_argument("--senkou-b", type=int, default=SENKOU_B)
    p.add_argument("--kosten", type=float, default=KOSTEN)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--split", default="2022-01-01")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.selftest:
        return selbsttest()

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
