#!/usr/bin/env python3
"""Lotto 6aus49: Sind 50 Jahre Ziehungen vom Zufall unterscheidbar?

Laedt alle Ziehungen (CSV aus dem Archiv daowa89/lottery-archive oder eine
lokale Datei), zaehlt, wie oft jede Zahl gezogen wurde, und prueft mit einem
Chi-Quadrat-Anpassungstest, ob die Abweichungen ueber normales Rauschen
hinausgehen. Zusaetzlich: Monte-Carlo-Gegenprobe mit simulierten fairen
Ziehungen und Suche nach doppelt gezogenen Sechser-Kombinationen.

Aufruf:
    python3 pruefung/lotto_zufallstest.py              # letzte 50 Jahre, Daten von GitHub
    python3 pruefung/lotto_zufallstest.py --jahre 20
    python3 pruefung/lotto_zufallstest.py --csv results.csv
    python3 pruefung/lotto_zufallstest.py --selftest   # offline
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import urllib.request
from collections import Counter
from datetime import date
from math import comb, exp, factorial

CSV_URL = ("https://raw.githubusercontent.com/daowa89/lottery-archive/"
           "main/de/lotto_6aus49/results.csv")
ZAHLEN = range(1, 50)
SPALTEN = ["n1", "n2", "n3", "n4", "n5", "n6"]


def lade_ziehungen(quelle: str | None) -> list[tuple[str, tuple[int, ...]]]:
    """Gibt [(datum, (6 Zahlen sortiert)), ...] zurueck, aelteste zuerst."""
    if quelle:
        with open(quelle, encoding="utf-8") as f:
            text = f.read()
    else:
        with urllib.request.urlopen(CSV_URL, timeout=30) as r:
            text = r.read().decode("utf-8")
    ziehungen = []
    for zeile in csv.DictReader(io.StringIO(text)):
        zahlen = tuple(sorted(int(zeile[s]) for s in SPALTEN))
        if len(set(zahlen)) != 6 or not all(1 <= z <= 49 for z in zahlen):
            raise ValueError(f"Ungueltige Ziehung am {zeile['date']}: {zahlen}")
        ziehungen.append((zeile["date"], zahlen))
    ziehungen.sort()
    return ziehungen


def zeitraum(ziehungen, jahre: int, bis: date | None = None):
    bis = bis or date.fromisoformat(ziehungen[-1][0])
    try:
        ab = bis.replace(year=bis.year - jahre)
    except ValueError:  # 29. Februar
        ab = bis.replace(year=bis.year - jahre, day=28)
    return [z for z in ziehungen if z[0] >= ab.isoformat()]


def haeufigkeiten(ziehungen) -> Counter:
    c = Counter({z: 0 for z in ZAHLEN})
    for _, zahlen in ziehungen:
        c.update(zahlen)
    return c


def chi2_sf(x: float, df: int) -> float:
    """P(Chi2 >= x) fuer gerade Freiheitsgrade, exakt (Poisson-Summe), ohne scipy."""
    assert df % 2 == 0
    h = x / 2
    return exp(-h) * sum(h ** i / factorial(i) for i in range(df // 2))


def chi_quadrat(c: Counter, n_ziehungen: int, korrigiert: bool = True) -> tuple[float, float]:
    """Chi-Quadrat-Test gegen Gleichverteilung, 48 Freiheitsgrade.

    Der Lehrbuch-Test setzt unabhaengige Einzelziehungen voraus. Beim Lotto
    werden aber 6 Kugeln OHNE Zuruecklegen gezogen: Jede Zahl kommt pro
    Ziehung hoechstens einmal vor, die Haeufigkeiten streuen dadurch um den
    Faktor (1 - 6/49) = 43/49 weniger. Der unkorrigierte Test unterschaetzt
    Auffaelligkeiten. Korrektur: Statistik mit 48/43 multiplizieren
    (bestaetigt durch die Monte-Carlo-Gegenprobe).
    """
    erwartet = n_ziehungen * 6 / 49
    stat = sum((c[z] - erwartet) ** 2 / erwartet for z in ZAHLEN)
    if korrigiert:
        stat *= 48 / 43
    return stat, chi2_sf(stat, 48)


def monte_carlo(n_ziehungen: int, beobachtet: float, laeufe: int = 2000,
                seed: int = 1) -> float:
    """Anteil simulierter fairer Lottoreihen mit mindestens so grosser Streuung."""
    rng = random.Random(seed)
    erwartet = n_ziehungen * 6 / 49
    treffer = 0
    for _ in range(laeufe):
        c = Counter()
        for _ in range(n_ziehungen):
            c.update(rng.sample(ZAHLEN, 6))
        stat = sum((c[z] - erwartet) ** 2 / erwartet for z in ZAHLEN) * 48 / 43
        treffer += stat >= beobachtet
    return treffer / laeufe


def doppelte_kombinationen(ziehungen) -> list[tuple[int, ...]]:
    c = Counter(z for _, z in ziehungen)
    return [k for k, v in c.items() if v > 1]


def bericht(ziehungen, jahre: int, mc_laeufe: int) -> None:
    auswahl = zeitraum(ziehungen, jahre)
    n = len(auswahl)
    c = haeufigkeiten(auswahl)
    erwartet = n * 6 / 49
    stat, p = chi_quadrat(c, n)
    _, p_roh = chi_quadrat(c, n, korrigiert=False)

    print(f"Zeitraum: {auswahl[0][0]} bis {auswahl[-1][0]}  ({n} Ziehungen)")
    print(f"Erwartete Haeufigkeit je Zahl bei Zufall: {erwartet:.1f}\n")

    rang = sorted(ZAHLEN, key=lambda z: (-c[z], z))
    print("Am haeufigsten:", ", ".join(f"{z} ({c[z]}x)" for z in rang[:6]))
    print("Am seltensten: ", ", ".join(f"{z} ({c[z]}x)" for z in rang[-6:]))

    print("\nChi-Quadrat-Test gegen Gleichverteilung")
    print(f"  chi2 = {stat:.1f}  bei 48 Freiheitsgraden,  p = {p:.3f}")
    print(f"  (Lehrbuch-Test ohne Korrektur fuer Ziehen ohne Zuruecklegen: p = {p_roh:.3f})")
    if mc_laeufe:
        p_mc = monte_carlo(n, stat, mc_laeufe)
        print(f"  Monte-Carlo-Gegenprobe ({mc_laeufe} faire Simulationen): p = {p_mc:.3f}")
    urteil = ("vom Zufall NICHT unterscheidbar" if p >= 0.05
              else "auffaellig - Daten und Methode pruefen, bevor man etwas glaubt")
    print(f"  Ergebnis: {urteil}")

    doppelt = doppelte_kombinationen(auswahl)
    print(f"\nDoppelt gezogene Sechser-Kombinationen: {len(doppelt)}")
    for k in doppelt:
        print("  ", k)

    print(f"\nChance auf 6 Richtige je Tipp: 1 : {comb(49, 6):,}".replace(",", "."))
    print("Mit Superzahl:                 1 : "
          f"{comb(49, 6) * 10:,}".replace(",", "."))


def selbsttest() -> None:
    """Prueft die Statistik mit Daten, deren Antwort bekannt ist."""
    rng = random.Random(42)
    fair = [(f"{i:05d}", tuple(sorted(rng.sample(ZAHLEN, 6)))) for i in range(4000)]
    _, p = chi_quadrat(haeufigkeiten(fair), len(fair))
    assert p > 0.001, f"faire Ziehungen faelschlich auffaellig (p={p})"

    # Gezinkt: Zahl 7 kommt in jeder vierten Ziehung zusaetzlich vor.
    gezinkt = []
    for i in range(4000):
        zahlen = rng.sample([z for z in ZAHLEN if z != 7], 6)
        if i % 4 == 0:
            zahlen[0] = 7
        gezinkt.append((f"{i:05d}", tuple(sorted(zahlen))))
    _, p = chi_quadrat(haeufigkeiten(gezinkt), len(gezinkt))
    assert p < 1e-6, f"gezinkte Ziehungen nicht erkannt (p={p})"

    doppelt = [("a", (1, 2, 3, 4, 5, 6)), ("b", (1, 2, 3, 4, 5, 6)), ("c", (7, 8, 9, 10, 11, 12))]
    assert doppelte_kombinationen(doppelt) == [(1, 2, 3, 4, 5, 6)]

    # Kalibrierung: Bei fairen Daten muss der korrigierte Test in ~5 % der
    # Faelle p < 0.05 liefern - nicht deutlich weniger (der Fehler des
    # Lehrbuch-Tests) und nicht deutlich mehr.
    fehlalarm = 0
    for _ in range(400):
        sim = [("x", tuple(rng.sample(ZAHLEN, 6))) for _ in range(500)]
        fehlalarm += chi_quadrat(haeufigkeiten(sim), 500)[1] < 0.05
    assert 0.02 <= fehlalarm / 400 <= 0.09, f"Test falsch kalibriert ({fehlalarm/400:.3f})"

    # chi2_sf gegen bekannte Tabellenwerte (df=48)
    assert abs(chi2_sf(65.171, 48) - 0.05) < 1e-3
    assert abs(chi2_sf(47.335, 48) - 0.50) < 1e-3

    assert comb(49, 6) == 13_983_816
    print("Selbsttest bestanden: faire Daten unauffaellig, gezinkte erkannt, "
          "Doppelte gefunden, Kombinatorik stimmt.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", help="lokale CSV statt Download")
    ap.add_argument("--jahre", type=int, default=50, help="Zeitraum in Jahren (Standard 50)")
    ap.add_argument("--mc", type=int, default=2000,
                    help="Anzahl Monte-Carlo-Laeufe, 0 = aus (Standard 2000)")
    ap.add_argument("--selftest", "--selbsttest", dest="selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selbsttest()
        return
    bericht(lade_ziehungen(a.csv), a.jahre, a.mc)


if __name__ == "__main__":
    main()
