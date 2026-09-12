#!/usr/bin/env python3
"""
fvg_backtest.py - Fair Value Gap (ICT), gegen die richtige Nullhypothese.

Die Formation
-------------
Drei Kerzen. Springt der Kurs so schnell, dass zwischen der ersten und
der dritten eine Luecke bleibt, heisst dieser Bereich Fair Value Gap.

    bullisch : Tief[i+1] > Hoch[i-1]   -> Zone = [Hoch[i-1], Tief[i+1]]
    baerisch : Hoch[i+1] < Tief[i-1]   -> Zone = [Hoch[i+1], Tief[i-1]]

Die Behauptung: Der Kurs kehrt spaeter in die Zone zurueck ("fuellt sie"),
und ein Einstieg beim Ruecklauf in Richtung der Luecke verdient Geld.

Warum die uebliche Auswertung wertlos ist
-----------------------------------------
Fast jede Zone wird irgendwann beruehrt - Kurse schwanken. Eine
Trefferquote von 80% oder 90% klingt beeindruckend und sagt nichts,
solange niemand dagegenhaelt, wie oft ein ZUFAELLIG platzierter Bereich
gleicher Breite und gleichen Abstands in derselben Zeit beruehrt worden
waere. Genau das rechnet dieses Skript aus.

Die Kontrollgruppe
------------------
Fuer jede echte FVG wird eine Zufallszone erzeugt mit
  - derselben Breite (in Prozent des Kurses),
  - demselben Abstand zum damaligen Kurs,
  - derselben Richtung,
  - aber an einem zufaelligen Zeitpunkt.
Danach laeuft exakt dieselbe Fuell- und Handelslogik darueber.
Schlaegt die echte FVG diese Kontrolle nicht, ist an der Formation nichts.

Zeitliche Sauberkeit
--------------------
Eine FVG aus den Kerzen i-1, i, i+1 ist erst mit dem SCHLUSS von i+1
bekannt. Gehandelt wird deshalb fruehestens ab Kerze i+2. Test [3] im
Selbsttest prueft, dass keine spaetere Kursbewegung in die Erkennung
zurueckwirkt.

Aufruf
------
    python3 fvg_backtest.py --selftest
    python3 fvg_backtest.py
    python3 fvg_backtest.py --tickers BTC-USD,ETH-USD --horizont 20 --halten 5
"""

import argparse
import logging
import math
import sys

import numpy as np
import pandas as pd

STANDARD_TICKER = ["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "QQQ",
                   "AAPL", "NVDA", "JPM"]
START = "2018-01-01"
MIN_GROESSE = 0.001      # Zone muss mindestens 0.1% breit sein
HORIZONT = 20            # Kerzen, in denen die Zone beruehrt werden darf
HALTEN = 5               # Kerzen Haltedauer nach dem Einstieg
KOSTEN = 0.0026          # 0.26% je Seite (Kraken Taker), rein und raus

log = logging.getLogger("fvg")


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
                    df = pd.DataFrame({"High": raw["High"][tk],
                                       "Low": raw["Low"][tk],
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
        daten[tickers[0]] = raw[["High", "Low", "Close"]].dropna()
    return daten


# ----------------------------------------------------------------------
# Erkennung
# ----------------------------------------------------------------------

def finde_fvgs(df, min_groesse=MIN_GROESSE):
    """Liste aller Fair Value Gaps.

    Jeder Eintrag: (bekannt_ab, richtung, unten, oben, referenzkurs)
    bekannt_ab ist der INDEX, ab dem gehandelt werden darf (i+2).
    """
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)
    treffer = []
    for i in range(1, n - 1):
        # bullisch: Luecke zwischen Hoch[i-1] und Tief[i+1]
        if l[i + 1] > h[i - 1]:
            unten, oben = h[i - 1], l[i + 1]
            richtung = 1
        elif h[i + 1] < l[i - 1]:
            unten, oben = h[i + 1], l[i - 1]
            richtung = -1
        else:
            continue
        ref = c[i + 1]
        if ref <= 0 or (oben - unten) / ref < min_groesse:
            continue
        if i + 2 >= n:
            continue
        treffer.append({"ab": i + 2, "richtung": richtung,
                        "unten": unten, "oben": oben, "ref": ref})
    return treffer


def zufallszonen(df, echte, rng):
    """Kontrollzonen: gleiche Breite, gleicher Abstand, gleiche Richtung,
    zufaelliger Zeitpunkt."""
    c = df["Close"].values
    n = len(df)
    kontrolle = []
    for z in echte:
        breite = (z["oben"] - z["unten"]) / z["ref"]
        # Abstand der Zonenmitte zum Referenzkurs, relativ
        abstand = ((z["oben"] + z["unten"]) / 2 - z["ref"]) / z["ref"]
        ab = int(rng.integers(2, max(3, n - 2)))
        ref = c[ab - 1]
        mitte = ref * (1 + abstand)
        halb = ref * breite / 2
        kontrolle.append({"ab": ab, "richtung": z["richtung"],
                          "unten": mitte - halb, "oben": mitte + halb,
                          "ref": ref})
    return kontrolle


# ----------------------------------------------------------------------
# Auswertung einer Zonenmenge
# ----------------------------------------------------------------------

def werte_zonen_aus(df, zonen, horizont=HORIZONT, halten=HALTEN,
                    kosten=KOSTEN):
    """Gibt (Fuellquote, Liste der Trade-Renditen) zurueck.

    Beruehrt = Kursspanne schneidet die Zone innerhalb des Horizonts.
    Einstieg am Zonenrand, Ausstieg nach `halten` Kerzen zum Schluss.
    """
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)
    gefuellt = 0
    renditen = []

    for z in zonen:
        start = z["ab"]
        ende = min(n, start + horizont)
        if start >= n:
            continue
        einstieg_idx = None
        for j in range(start, ende):
            if l[j] <= z["oben"] and h[j] >= z["unten"]:
                einstieg_idx = j
                break
        if einstieg_idx is None:
            continue
        gefuellt += 1

        # Einstieg am zuerst beruehrten Zonenrand.
        # Bullische FVG liegt unter dem Kurs -> Ruecklauf von oben -> Oberkante.
        preis = z["oben"] if z["richtung"] == 1 else z["unten"]
        aus = min(n - 1, einstieg_idx + halten)
        if aus <= einstieg_idx:
            continue
        roh = (c[aus] - preis) / preis * z["richtung"]
        renditen.append(roh - 2 * kosten)

    quote = gefuellt / len(zonen) if zonen else float("nan")
    return quote, renditen


def statistik(renditen):
    if not renditen:
        return {}
    a = np.array(renditen)
    se = a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else float("nan")
    t = a.mean() / se if se and se > 0 else float("nan")
    return {"n": len(a), "mittel": float(a.mean()), "t": float(t),
            "p": math.erfc(abs(t) / math.sqrt(2)) if not math.isnan(t) else float("nan"),
            "trefferquote": float((a > 0).mean())}


# ----------------------------------------------------------------------

def auswerten(daten, args):
    rng = np.random.default_rng(args.seed)

    print("\n" + "=" * 76)
    print("A) FUELLQUOTE - echte FVG gegen Zufallszonen")
    print("=" * 76)
    print(f"{'Wert':<10}{'FVGs':>7}{'gefuellt':>11}{'Zufall':>11}"
          f"{'Differenz':>12}")
    print("-" * 76)

    alle_echt, alle_zufall, zeilen = [], [], []
    for tk, df in daten.items():
        echte = finde_fvgs(df, args.min_groesse)
        if len(echte) < 20:
            print(f"{tk:<10}{len(echte):>7}   zu wenige Formationen")
            continue
        q_e, r_e = werte_zonen_aus(df, echte, args.horizont, args.halten, args.kosten)

        quoten_z, renditen_z = [], []
        for _ in range(args.sims):
            kz = zufallszonen(df, echte, rng)
            q, r = werte_zonen_aus(df, kz, args.horizont, args.halten, args.kosten)
            quoten_z.append(q)
            renditen_z.extend(r)
        q_z = float(np.nanmean(quoten_z))

        print(f"{tk:<10}{len(echte):>7}{q_e:>11.1%}{q_z:>11.1%}{q_e-q_z:>+12.1%}")
        zeilen.append((tk, echte, r_e, renditen_z, q_e, q_z))
        alle_echt.extend(r_e)
        alle_zufall.extend(renditen_z)

    if not zeilen:
        print("\nZu wenig Material fuer eine Auswertung.")
        return

    print("-" * 76)
    print("Eine hohe Fuellquote allein beweist nichts - entscheidend ist,")
    print("ob sie ueber der einer beliebigen Zone gleicher Groesse liegt.")

    print("\n" + "=" * 76)
    print("B) HANDELSERGEBNIS je Trade, nach Gebuehren")
    print("=" * 76)
    print(f"{'Wert':<10}{'Trades':>8}{'FVG Schnitt':>14}{'Zufall':>11}"
          f"{'Differenz':>12}{'t':>7}{'Treffer':>10}")
    print("-" * 76)
    for tk, echte, r_e, r_z, q_e, q_z in zeilen:
        s_e = statistik(r_e)
        s_z = statistik(r_z)
        if not s_e or not s_z:
            continue
        print(f"{tk:<10}{s_e['n']:>8}{s_e['mittel']:>13.3%}"
              f"{s_z['mittel']:>11.3%}{s_e['mittel']-s_z['mittel']:>+12.3%}"
              f"{s_e['t']:>7.2f}{s_e['trefferquote']:>10.1%}")

    g_e, g_z = statistik(alle_echt), statistik(alle_zufall)
    print("-" * 76)
    print(f"{'GESAMT':<10}{g_e['n']:>8}{g_e['mittel']:>13.3%}"
          f"{g_z['mittel']:>11.3%}{g_e['mittel']-g_z['mittel']:>+12.3%}"
          f"{g_e['t']:>7.2f}{g_e['trefferquote']:>10.1%}")

    # Zweistichprobentest echte gegen zufaellige Trades
    a, b = np.array(alle_echt), np.array(alle_zufall)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t_diff = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    p_diff = math.erfc(abs(t_diff) / math.sqrt(2))
    print(f"\nFVG gegen Zufallszone: t={t_diff:.2f}  p={p_diff:.3f}")
    print(f"Nullrendite-Test (FVG gegen 0): t={g_e['t']:.2f}  p={g_e['p']:.3f}")
    print("-" * 76)
    if abs(t_diff) < 2:
        print("BEFUND: Kein messbarer Unterschied zu einer beliebigen Zone.")
        print("        Die Formation traegt keine Information.")
    elif a.mean() <= 0:
        print("BEFUND: Unterschied vorhanden, aber die Rendite ist negativ.")
    else:
        print("BEFUND: Echte FVGs schlagen die Kontrolle. Weiter mit C und D.")

    print("\n" + "=" * 76)
    print("C) PARAMETER-GITTER (Schnittrendite je Trade, alle Werte)")
    print("=" * 76)
    print(f"{'Horizont/Halten':<20}{'Trades':>9}{'FVG':>11}{'Zufall':>11}{'t':>8}")
    for hz in (10, 20, 40):
        for hl in (3, 5, 10):
            r_alle, z_alle = [], []
            for tk, df in daten.items():
                e = finde_fvgs(df, args.min_groesse)
                if len(e) < 20:
                    continue
                _, re_ = werte_zonen_aus(df, e, hz, hl, args.kosten)
                r_alle.extend(re_)
                kz = zufallszonen(df, e, rng)
                _, rz_ = werte_zonen_aus(df, kz, hz, hl, args.kosten)
                z_alle.extend(rz_)
            if len(r_alle) < 30 or len(z_alle) < 30:
                continue
            aa, bb = np.array(r_alle), np.array(z_alle)
            s = math.sqrt(aa.var(ddof=1)/len(aa) + bb.var(ddof=1)/len(bb))
            tt = (aa.mean() - bb.mean()) / s if s > 0 else float("nan")
            print(f"{f'{hz}/{hl}':<20}{len(aa):>9}{aa.mean():>10.3%}"
                  f"{bb.mean():>11.3%}{tt:>8.2f}")
    print("-" * 76)
    print("Funktioniert nur eine Zelle, ist das Kurvenanpassung.\n")


# ----------------------------------------------------------------------

def selbsttest():
    print("Selbsttest\n" + "-" * 70)
    fehler = 0

    # Handgebaute Kerzen mit genau EINER bullischen Luecke
    h = [10, 15, 15, 16, 16, 16, 16]
    l = [9, 10, 13, 14, 10, 14, 14]     # Tief[2]=13 > Hoch[0]=10 -> Zone [10,13]
    c = [9.5, 10.5, 14, 15, 12, 15, 15]
    df = pd.DataFrame({"High": h, "Low": l, "Close": c})

    fvgs = finde_fvgs(df, 0.0)
    ok = len(fvgs) == 1 and fvgs[0]["richtung"] == 1
    print(f"[1] Genau eine bullische FVG erkannt: {'OK' if ok else 'FEHLER'} "
          f"({len(fvgs)} gefunden)")
    fehler += 0 if ok else 1

    if fvgs:
        z = fvgs[0]
        ok = abs(z["unten"] - 10) < 1e-9 and abs(z["oben"] - 13) < 1e-9 and z["ab"] == 3
        print(f"[2] Zone [{z['unten']}, {z['oben']}], handelbar ab Index "
              f"{z['ab']}: {'OK' if ok else 'FEHLER'}")
        fehler += 0 if ok else 1

    # Kein Blick in die Zukunft: spaetere Kerzen aendern fruehere Funde nicht
    rng = np.random.default_rng(2)
    n = 800
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    cl = 100 * np.cumprod(1 + rng.normal(0.0003, 0.025, n))
    gross = pd.DataFrame({"Close": cl,
                          "High": cl * (1 + abs(rng.normal(0, 0.02, n))),
                          "Low": cl * (1 - abs(rng.normal(0, 0.02, n)))}, index=idx)
    halb = n // 2
    man = gross.copy()
    f = np.linspace(1.0, 5.0, n - halb)
    for sp in ("Close", "High", "Low"):
        man.iloc[halb:, man.columns.get_loc(sp)] = man[sp].iloc[halb:].values * f
    a = [(z["ab"], round(z["unten"], 6)) for z in finde_fvgs(gross) if z["ab"] < halb - 2]
    b = [(z["ab"], round(z["unten"], 6)) for z in finde_fvgs(man) if z["ab"] < halb - 2]
    ok = a == b and len(a) > 5
    print(f"[3] Kein Blick in die Zukunft ({len(a)} Funde vorher): "
          f"{'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # Kontrollzonen: gleiche Anzahl und gleiche Breiten
    echte = finde_fvgs(gross)
    kz = zufallszonen(gross, echte, rng)
    br_e = sorted(round((z["oben"] - z["unten"]) / z["ref"], 6) for z in echte)
    br_z = sorted(round((z["oben"] - z["unten"]) / z["ref"], 6) for z in kz)
    ok = len(kz) == len(echte) and np.allclose(br_e, br_z, atol=1e-6)
    print(f"[4] Kontrollzonen gleich breit und gleich viele: "
          f"{'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # Gebuehren muessen die Rendite senken
    _, r0 = werte_zonen_aus(gross, echte, 20, 5, 0.0)
    _, r1 = werte_zonen_aus(gross, echte, 20, 5, 0.01)
    ok = len(r0) == len(r1) and np.mean(r1) < np.mean(r0)
    print(f"[5] Kosten senken die Rendite: {'OK' if ok else 'FEHLER'}")
    fehler += 0 if ok else 1

    # Auf Zufallsdaten darf die FVG die Kontrolle nicht schlagen
    q_e, re_ = werte_zonen_aus(gross, echte, 20, 5, 0.0)
    q_z, rz_ = werte_zonen_aus(gross, zufallszonen(gross, echte, rng), 20, 5, 0.0)
    print(f"\n    Auf Zufallsdaten: Fuellquote {q_e:.1%} gegen {q_z:.1%}, "
          f"Rendite {np.mean(re_):+.3%} gegen {np.mean(rz_):+.3%}")
    print("    (Beides sollte nahe beieinander liegen - hier gibt es nichts")
    print("     zu finden, also darf auch nichts gefunden werden.)")

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
    p.add_argument("--min-groesse", type=float, default=MIN_GROESSE)
    p.add_argument("--horizont", type=int, default=HORIZONT)
    p.add_argument("--halten", type=int, default=HALTEN)
    p.add_argument("--kosten", type=float, default=KOSTEN)
    p.add_argument("--sims", type=int, default=20)
    p.add_argument("--seed", type=int, default=23)
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
