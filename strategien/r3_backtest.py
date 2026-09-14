#!/usr/bin/env python3
"""
r3_backtest.py - Die R3-Strategie von Larry Connors, sauber geprueft.

Die Regeln (aus "High Probability ETF Trading", 2009)
----------------------------------------------------
Kaufsignal, wenn alle vier Bedingungen gelten:
  1. Schlusskurs ueber dem gleitenden 200-Tage-Durchschnitt
  2. Der 2-Perioden-RSI faellt drei Tage in Folge
  3. Am ersten Tag dieses Rueckgangs lag der RSI bereits unter 60
  4. Der RSI schliesst heute unter 10
Ausstieg: RSI schliesst ueber 70. Kein Stop-Loss, kein Gewinnziel.

Leerverkauf spiegelbildlich: Kurs unter der 200-Tage-Linie, RSI steigt
drei Tage, am ersten Tag ueber 40, heute ueber 90; Ausstieg unter 30.

Was an der ueblichen Darstellung nicht stimmt
---------------------------------------------
Wer fuer R3 wirbt, zeigt die Trefferquote: 9 von 12 Signalen, 12 von 17.
Das klingt nach Koennen und ist keines - aus zwei Gruenden.

ERSTENS ist die Trefferquote in die Ausstiegsregel eingebaut. Ausstieg
bei RSI ueber 70 schneidet Gewinne frueh ab; ohne Stop-Loss duerfen
Verluste dagegen laufen. Viele kleine Gewinne, seltene grosse Verluste -
die Quote sagt nichts ueber das Ergebnis.

ZWEITENS fehlt die Vergleichsbasis. Wie oft ist ein Einstieg an einem
BELIEBIGEN Tag mit derselben Haltedauer im Plus? In einem steigenden
Markt fast genauso oft. Genau das rechnet dieses Skript aus.

Die Kontrollgruppe
------------------
Zu jedem echten Trade wird ein Zufallstrade erzeugt: gleicher Wert,
gleiche Haltedauer, gleiche Richtung, aber zufaelliger Einstiegstag.
Gleiche Anzahl, gleiche Verteilung der Haltedauern. Der einzige
Unterschied ist, ob das R3-Setup vorlag.

Zeitliche Sauberkeit
--------------------
Das Signal steht mit dem Schlusskurs von Tag t fest. Ein- und Ausstieg
erfolgen deshalb zur EROEFFNUNG des Folgetags, nicht zum Schluss des
Signaltags. Test [4] im Selbsttest prueft, dass keine spaetere
Kursbewegung in die Signalbildung zurueckwirkt.

Aufruf
------
    python3 r3_backtest.py --selftest
    python3 r3_backtest.py
    python3 r3_backtest.py --richtung beide --split 2010-01-01
"""

import argparse
import logging
import math
import sys

import numpy as np
import pandas as pd

STANDARD_TICKER = [
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "XLF", "XLE", "XLK", "XLV",
    "AAPL", "MSFT", "JNJ", "PG", "KO", "JPM", "XOM", "WMT", "V", "NVDA",
]
START = "2000-01-01"

SMA_LANG = 200
RSI_PERIODE = 2
RSI_EINSTIEG_LONG = 10      # heute darunter
RSI_START_LONG = 60         # am ersten Tag des Rueckgangs darunter
RSI_AUSSTIEG_LONG = 70
RSI_EINSTIEG_SHORT = 90
RSI_START_SHORT = 40
RSI_AUSSTIEG_SHORT = 30
MAX_HALTEN = 20             # Notbremse, falls der Ausstieg nie kommt
KOSTEN = 0.0010             # 0.10% je Seite

log = logging.getLogger("r3")


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
                    df = pd.DataFrame({"Open": raw["Open"][tk],
                                       "Close": raw["Close"][tk]})
                else:
                    df = raw[tk][["Open", "Close"]].copy()
            except KeyError:
                log.warning("Keine Daten fuer %s", tk)
                continue
            df = df.dropna()
            if len(df) > SMA_LANG + 100:
                daten[tk] = df
            else:
                log.warning("Zu wenig Daten fuer %s", tk)
    else:
        daten[tickers[0]] = raw[["Open", "Close"]].dropna()
    return daten


# ----------------------------------------------------------------------
# Indikatoren
# ----------------------------------------------------------------------

def rsi(close, periode=RSI_PERIODE):
    """RSI nach Wilder. Bei nur Aufwaertsbewegungen 100, bei nur
    Abwaertsbewegungen 0."""
    delta = close.diff()
    gewinn = delta.clip(lower=0)
    verlust = -delta.clip(upper=0)
    mg = gewinn.ewm(alpha=1.0 / periode, adjust=False).mean()
    mv = verlust.ewm(alpha=1.0 / periode, adjust=False).mean()
    werte = np.where(mv == 0, 100.0,
                     np.where(mg == 0, 0.0, 100.0 - 100.0 / (1.0 + mg / mv)))
    s = pd.Series(werte, index=close.index)
    s.iloc[:periode] = np.nan
    return s


def signale(df, richtung="long", rsi_periode=RSI_PERIODE,
            ein_long=RSI_EINSTIEG_LONG, start_long=RSI_START_LONG,
            ein_short=RSI_EINSTIEG_SHORT, start_short=RSI_START_SHORT,
            sma_lang=SMA_LANG):
    """Boolesche Reihe: an Tag t liegt ein Einstiegssignal vor.

    Nutzt ausschliesslich Daten bis einschliesslich t.
    """
    c = df["Close"]
    r = rsi(c, rsi_periode)
    sma = c.rolling(sma_lang).mean()

    faellt3 = (r < r.shift(1)) & (r.shift(1) < r.shift(2)) & (r.shift(2) < r.shift(3))
    steigt3 = (r > r.shift(1)) & (r.shift(1) > r.shift(2)) & (r.shift(2) > r.shift(3))

    long_sig = (c > sma) & faellt3 & (r.shift(3) < start_long) & (r < ein_long)
    short_sig = (c < sma) & steigt3 & (r.shift(3) > start_short) & (r > ein_short)

    if richtung == "long":
        return long_sig.fillna(False), pd.Series(False, index=c.index), r
    if richtung == "short":
        return pd.Series(False, index=c.index), short_sig.fillna(False), r
    return long_sig.fillna(False), short_sig.fillna(False), r


# ----------------------------------------------------------------------
# Handel
# ----------------------------------------------------------------------

def handele(df, richtung, args):
    """Fuehrt die Strategie aus. Gibt eine Liste von Trades zurueck:
    (einstiegs_index, dauer_in_tagen, rendite_nach_kosten, richtungszahl)"""
    long_sig, short_sig, r = signale(
        df, richtung, args.rsi_periode, args.ein_long, args.start_long,
        args.ein_short, args.start_short, args.sma)
    o = df["Open"].values
    n = len(df)
    trades = []
    i = 0
    while i < n - 2:
        ist_long = bool(long_sig.iloc[i])
        ist_short = bool(short_sig.iloc[i])
        if not (ist_long or ist_short):
            i += 1
            continue
        seite = 1 if ist_long else -1
        ein_idx = i + 1                      # Eroeffnung des Folgetags
        if ein_idx >= n:
            break
        einstieg = o[ein_idx]

        # Ausstieg suchen: erster Schlusskurs, der die Bedingung erfuellt
        aus_idx = None
        for j in range(ein_idx, min(n - 1, ein_idx + args.max_halten)):
            rv = r.iloc[j]
            if np.isnan(rv):
                continue
            if seite == 1 and rv > args.aus_long:
                aus_idx = j + 1
                break
            if seite == -1 and rv < args.aus_short:
                aus_idx = j + 1
                break
        if aus_idx is None:
            aus_idx = min(n - 1, ein_idx + args.max_halten)
        if aus_idx >= n:
            break

        roh = (o[aus_idx] - einstieg) / einstieg * seite
        trades.append((ein_idx, aus_idx - ein_idx, roh - 2 * args.kosten, seite))
        i = aus_idx
    return trades


def zufallstrades(df, trades, rng, kosten):
    """Zu jedem echten Trade einer mit gleicher Dauer und Richtung,
    aber zufaelligem Einstiegstag."""
    o = df["Open"].values
    n = len(df)
    ergebnis = []
    for _, dauer, _, seite in trades:
        if dauer < 1 or n - dauer - 2 <= args_min_index:
            continue
        ein = int(rng.integers(args_min_index, n - dauer - 1))
        aus = ein + dauer
        roh = (o[aus] - o[ein]) / o[ein] * seite
        ergebnis.append(roh - 2 * kosten)
    return ergebnis


args_min_index = SMA_LANG + 5    # vor der 200-Tage-Linie gibt es keine Signale


# ----------------------------------------------------------------------

def stat(werte):
    if not werte:
        return {}
    a = np.array(werte)
    se = a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else float("nan")
    t = a.mean() / se if se and se > 0 else float("nan")
    return {"n": len(a), "mittel": float(a.mean()),
            "quote": float((a > 0).mean()), "t": float(t),
            "bester": float(a.max()), "schlechtester": float(a.min())}


def binomial_p(treffer, gesamt):
    """Wahrscheinlichkeit, mindestens so viele Treffer bei fairer Muenze."""
    if gesamt == 0:
        return float("nan")
    s = sum(math.comb(gesamt, k) for k in range(treffer, gesamt + 1))
    return s / (2 ** gesamt)


# ----------------------------------------------------------------------

def auswerten(daten, args):
    rng = np.random.default_rng(args.seed)

    print("\n" + "=" * 80)
    print("A) TRADES JE WERT - R3 gegen Zufallseinstieg gleicher Haltedauer")
    print("=" * 80)
    print(f"{'Wert':<8}{'Trades':>8}{'R3 Schnitt':>12}{'Quote':>8}"
          f"{'Zufall':>11}{'Quote':>8}{'Diff':>10}{'Tage':>7}")
    print("-" * 80)

    alle_r3, alle_zufall, zeilen = [], [], []
    for tk, df in daten.items():
        trades = handele(df, args.richtung, args)
        if len(trades) < 5:
            continue
        renditen = [t[2] for t in trades]
        dauern = [t[1] for t in trades]

        z_alle = []
        for _ in range(args.sims):
            z_alle.extend(zufallstrades(df, trades, rng, args.kosten))

        s_r, s_z = stat(renditen), stat(z_alle)
        if not s_r or not s_z:
            continue
        print(f"{tk:<8}{s_r['n']:>8}{s_r['mittel']:>11.3%}{s_r['quote']:>8.1%}"
              f"{s_z['mittel']:>10.3%}{s_z['quote']:>8.1%}"
              f"{s_r['mittel']-s_z['mittel']:>+10.3%}{np.mean(dauern):>7.1f}")
        alle_r3.extend(renditen)
        alle_zufall.extend(z_alle)
        zeilen.append((tk, trades, renditen, z_alle))

    if not zeilen:
        print("\nZu wenige Signale fuer eine Auswertung.")
        return

    g_r, g_z = stat(alle_r3), stat(alle_zufall)
    print("-" * 80)
    print(f"{'GESAMT':<8}{g_r['n']:>8}{g_r['mittel']:>11.3%}{g_r['quote']:>8.1%}"
          f"{g_z['mittel']:>10.3%}{g_z['quote']:>8.1%}"
          f"{g_r['mittel']-g_z['mittel']:>+10.3%}")

    a, b = np.array(alle_r3), np.array(alle_zufall)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t_diff = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    p_diff = math.erfc(abs(t_diff) / math.sqrt(2))

    print(f"\nR3 gegen Zufallseinstieg:  t={t_diff:.2f}   p={p_diff:.3f}")
    print(f"R3 gegen null:             t={g_r['t']:.2f}   "
          f"p={math.erfc(abs(g_r['t'])/math.sqrt(2)):.3f}")
    print(f"Bester Trade {g_r['bester']:+.1%}, schlechtester {g_r['schlechtester']:+.1%}")
    print("-" * 80)
    if abs(t_diff) < 2:
        print("BEFUND: Kein Unterschied zu einem beliebigen Einstiegstag.")
        print("        Das Setup traegt keine Information.")
    elif a.mean() <= 0:
        print("BEFUND: Unterschied vorhanden, Rendite aber negativ.")
    else:
        print("BEFUND: R3 schlaegt den Zufallseinstieg. Weiter mit B bis D.")

    print("\n" + "=" * 80)
    print("B) DIE TREFFERQUOTE IM VERGLEICH")
    print("=" * 80)
    treffer = int(round(g_r["quote"] * g_r["n"]))
    print(f"R3:      {treffer} von {g_r['n']} Trades im Plus ({g_r['quote']:.1%})")
    print(f"Zufall:  {g_z['quote']:.1%} bei gleicher Haltedauer")
    print(f"\nGegen den Muenzwurf waere p={binomial_p(treffer, g_r['n']):.4f} -")
    print("aber der Muenzwurf ist der falsche Massstab. Der richtige ist die")
    print(f"Zufallsquote von {g_z['quote']:.1%}: So oft ist ein beliebiger Einstieg")
    print("mit derselben Haltedauer im Plus.")
    if g_r["quote"] - g_z["quote"] < 0.03:
        print("\nDie hohe Quote von R3 ist damit im Wesentlichen die Quote des")
        print("Marktes selbst, nicht die Leistung des Setups.")

    print("\n" + "=" * 80)
    print("C) TEILPERIODEN (Buch erschien 2009)")
    print("=" * 80)
    grenze = pd.Timestamp(args.split)
    for name, vor in ((f"bis {args.split}", True), (f"ab {args.split}", False)):
        teil = []
        for tk, trades, renditen, _ in zeilen:
            idx = daten[tk].index
            for (ein, dauer, rend, seite) in trades:
                if (idx[ein] < grenze) == vor:
                    teil.append(rend)
        s = stat(teil)
        if not s:
            continue
        print(f"{name:<16}{s['n']:>6} Trades   Schnitt {s['mittel']:>+8.3%}"
              f"   Quote {s['quote']:>6.1%}   t={s['t']:>6.2f}")
    print("-" * 80)
    print("Eine Anomalie, die nach der Veroeffentlichung verschwindet, war")
    print("entweder nie da oder ist abgeerntet.")

    print("\n" + "=" * 80)
    print("D) PARAMETER-GITTER (Schnittrendite je Trade)")
    print("=" * 80)
    print(f"{'Einstieg/Ausstieg':<22}{'Trades':>9}{'Schnitt':>11}{'Quote':>9}{'t':>8}")
    for ein in (5, 10, 15):
        for aus in (65, 70, 75):
            unter = argparse.Namespace(**vars(args))
            unter.ein_long, unter.aus_long = ein, aus
            r_alle = []
            for tk, df in daten.items():
                tr = handele(df, args.richtung, unter)
                r_alle.extend([x[2] for x in tr])
            s = stat(r_alle)
            if not s or s["n"] < 20:
                continue
            print(f"{f'RSI<{ein} / RSI>{aus}':<22}{s['n']:>9}"
                  f"{s['mittel']:>10.3%}{s['quote']:>9.1%}{s['t']:>8.2f}")
    print("-" * 80)
    print("Funktioniert nur eine Zelle, ist es Kurvenanpassung.\n")


# ----------------------------------------------------------------------

def selbsttest():
    print("Selbsttest\n" + "-" * 70)
    fehler = 0

    # [1] RSI gegen Handrechnung: nur Anstiege -> 100, nur Rueckgaenge -> 0
    steigend = pd.Series(np.arange(1, 30, dtype=float))
    fallend = pd.Series(np.arange(30, 1, -1, dtype=float))
    ok = (abs(rsi(steigend).iloc[-1] - 100) < 1e-9
          and abs(rsi(fallend).iloc[-1] - 0) < 1e-9)
    print(f"[1] RSI-Grenzfaelle 100 / 0: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [2] RSI liegt immer zwischen 0 und 100
    rng = np.random.default_rng(6)
    n = 2000
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    c = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n)), index=idx)
    r = rsi(c).dropna()
    ok = bool((r >= 0).all() and (r <= 100).all())
    print(f"[2] RSI bleibt in [0, 100]: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    df = pd.DataFrame({"Close": c, "Open": c.shift(1).bfill()})

    # [3] Alle vier Bedingungen muessen erfuellt sein
    long_sig, _, rr = signale(df, "long")
    sma = c.rolling(SMA_LANG).mean()
    treffer = long_sig[long_sig].index
    pruefungen = []
    for t in treffer[:50]:
        i = c.index.get_loc(t)
        if i < SMA_LANG + 5:
            continue
        pruefungen.append(
            c.iloc[i] > sma.iloc[i]
            and rr.iloc[i] < 10
            and rr.iloc[i] < rr.iloc[i-1] < rr.iloc[i-2] < rr.iloc[i-3]
            and rr.iloc[i-3] < 60)
    ok = len(pruefungen) > 0 and all(pruefungen)
    print(f"[3] Signale erfuellen alle vier Regeln ({len(treffer)} Signale): "
          f"{'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [4] Kein Blick in die Zukunft
    halb = n // 2
    man = df.copy()
    f = np.linspace(1.0, 4.0, n - halb)
    for sp in ("Close", "Open"):
        man.iloc[halb:, man.columns.get_loc(sp)] = man[sp].iloc[halb:].values * f
    a = list(signale(df, "long")[0].iloc[:halb].values)
    b = list(signale(man, "long")[0].iloc[:halb].values)
    ok = a == b and sum(a) > 0
    print(f"[4] Kein Blick in die Zukunft: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [5] Kosten senken die Rendite
    basis = argparse.Namespace(
        richtung="long", rsi_periode=2, ein_long=10, start_long=60,
        ein_short=90, start_short=40, aus_long=70, aus_short=30,
        sma=200, max_halten=20, kosten=0.0, seed=1, sims=5, split="2010-01-01")
    t0 = handele(df, "long", basis)
    teuer = argparse.Namespace(**vars(basis))
    teuer.kosten = 0.01
    t1 = handele(df, "long", teuer)
    ok = len(t0) == len(t1) and np.mean([x[2] for x in t1]) < np.mean([x[2] for x in t0])
    print(f"[5] Kosten senken die Rendite ({len(t0)} Trades): "
          f"{'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [6] Kontrollgruppe hat gleiche Dauern und gleiche Anzahl
    zt = zufallstrades(df, t0, rng, 0.0)
    ok = len(zt) == len(t0)
    print(f"[6] Kontrollgruppe gleich gross: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # [7] Ausstiegsregel greift
    dauern = [x[1] for x in t0]
    ok = len(dauern) > 0 and max(dauern) <= 20 and np.mean(dauern) < 10
    print(f"[7] Haltedauer im Rahmen (Schnitt {np.mean(dauern):.1f} Tage, "
          f"max {max(dauern)}): {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    if t0:
        s = stat([x[2] for x in t0])
        z = stat(zt)
        print(f"\n    Auf Zufallsdaten: R3 {s['mittel']:+.3%} bei Quote "
              f"{s['quote']:.1%}, Zufallseinstieg {z['mittel']:+.3%} bei "
              f"{z['quote']:.1%}")
        print("    (Hier gibt es nichts zu finden - beide Quoten sollten")
        print("     aehnlich sein, und genau das ist der Punkt der Kontrolle.)")

    # [8] Der Kern der Kritik, nachgewiesen auf driftfreien Zufallsdaten:
    # Ohne jeden Trend MUSS die mittlere Rendite bei null liegen - sonst
    # steckt ein Fehler im Code. Die Trefferquote liegt trotzdem deutlich
    # ueber 50%, weil die Ausstiegsregel wartet, bis der RSI erholt ist.
    # Genau diese Quote wird als Erfolg verkauft.
    mittel, quoten, z_mittel, z_quoten = [], [], [], []
    for s in range(30):
        r2 = np.random.default_rng(500 + s)
        m = 3000
        ix = pd.date_range("2005-01-01", periods=m, freq="B")
        cc = pd.Series(100 * np.cumprod(1 + r2.normal(0.0, 0.012, m)), index=ix)
        d2 = pd.DataFrame({"Close": cc, "Open": cc.shift(1).bfill()})
        tt = handele(d2, "long", basis)
        if len(tt) < 10:
            continue
        rr2 = [x[2] for x in tt]
        zz = zufallstrades(d2, tt, rng, 0.0)
        mittel.append(np.mean(rr2))
        quoten.append(np.mean([x > 0 for x in rr2]))
        z_mittel.append(np.mean(zz))
        z_quoten.append(np.mean([x > 0 for x in zz]))
    m_r3, q_r3 = float(np.mean(mittel)), float(np.mean(quoten))
    m_z, q_z = float(np.mean(z_mittel)), float(np.mean(z_quoten))
    streu = float(np.std(mittel, ddof=1))
    ok = abs(m_r3) < 2 * streu and q_r3 > q_z + 0.05
    print(f"[8] Driftfreie Zufallsdaten, {len(mittel)} Laeufe:")
    print(f"    Rendite {m_r3:+.4%} (Streuung {streu:.4%}) - muss nahe null sein")
    print(f"    Trefferquote {q_r3:.1%} gegen {q_z:.1%} beim Zufallseinstieg")
    print(f"    -> Quote ohne jeden Ertrag: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

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
    p.add_argument("--richtung", default="long", choices=["long", "short", "beide"])
    p.add_argument("--rsi-periode", type=int, default=RSI_PERIODE)
    p.add_argument("--ein-long", type=float, default=RSI_EINSTIEG_LONG)
    p.add_argument("--start-long", type=float, default=RSI_START_LONG)
    p.add_argument("--aus-long", type=float, default=RSI_AUSSTIEG_LONG)
    p.add_argument("--ein-short", type=float, default=RSI_EINSTIEG_SHORT)
    p.add_argument("--start-short", type=float, default=RSI_START_SHORT)
    p.add_argument("--aus-short", type=float, default=RSI_AUSSTIEG_SHORT)
    p.add_argument("--sma", type=int, default=SMA_LANG)
    p.add_argument("--max-halten", type=int, default=MAX_HALTEN)
    p.add_argument("--kosten", type=float, default=KOSTEN)
    p.add_argument("--sims", type=int, default=20)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--split", default="2010-01-01")
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
