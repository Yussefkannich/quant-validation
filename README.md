# quant-validation

Ein Werkzeugkasten, um Handelsstrategien ergebnisoffen zu prüfen — und sie fallenzulassen, wenn sie nicht tragen.

Dieses Repository enthält keine Strategie, die funktioniert. Es enthält die Methode, mit der fünf populäre Ansätze widerlegt wurden, und die Zahlen dazu. Das ist Absicht: Der Wert liegt in der Prüfkette, nicht im Ergebnis.

*An English summary follows at the end.*

---

## Ergebnisse auf einen Blick

| Ansatz | Rohergebnis | Nach Prüfung | Befund |
|---|---|---|---|
| Cross-Sectional Relative Strength (Aktien) | +9,6% p.a. gegen SPY, Sharpe 1,21 | −0,61% ohne fünf Einzelwerte | verworfen |
| Ichimoku Kinko Hyo (Aktien) | — | −18,3% p.a. gegen Buy & Hold | verworfen |
| Ichimoku Kinko Hyo (Krypto) | +4,8% p.a. bei 0,10% Gebühren | +2,5% bei realistischen 0,26% | verworfen |
| Fair Value Gap / ICT (Tagesbasis) | Füllquote 77–85% | Zufallszonen: 79–83% | verworfen |
| EMA-Crossover 12/26 (Aktien) | −11,8% p.a. gegen Buy & Hold | 36. Perzentil gegen Zufallstiming | verworfen |
| EMA-Crossover 12/26 (Krypto) | +7,0% p.a., 96,8. Perzentil | out-of-sample 87,7. Perzentil, vor 2022 kein Vorsprung | verworfen |
| **R3 / Larry Connors (RSI-2)** | **+0,50% je Trade, t=5,75 nach Cluster-Korrektur** | **out-of-sample 1 von 6 Märkten** | **verworfen** |
| Funding-Rate-Carry (Perpetuals) | Werbung: 10–30% p.a. | gemessen: 0,3–0,4% p.a. netto | derzeit unattraktiv |

Jede Zeile ist reproduzierbar. Die Befehle stehen weiter unten.

**R3 ist der interessanteste Fall.** Relative Strength, Ichimoku und Fair Value Gap brachen früh zusammen. Der EMA-Crossover auf Krypto bestand immerhin die Zufallskontrolle und verfehlte out-of-sample die Hürde nur knapp. R3 überstand drei Prüfstufen — Zufallskontrolle, Cluster-Korrektur, Ausschluss der stärksten Werte — und fiel erst an der vierten. Nach jedem üblichen Maßstab wäre es eine bestätigte Strategie gewesen.

---

## Warum das nötig ist

Ein Backtest liefert fast immer eine Zahl, die gut aussieht. Der Grund ist selten ein Vorteil der Strategie, sondern meistens einer von fünf Fehlern:

**1. Der Vergleichsmaßstab ist falsch.** Relative Strength schlug den S&P 500 um 9,6% pro Jahr. Verglichen mit einem gleichgewichteten Halten *derselben hundert Aktien* — also derselben Auswahl, die rückblickend als erfolgreich bekannt war — blieben 7,7% übrig. Nach Ausschluss von fünf Einzelwerten waren es −0,61%.

**2. Die Nullhypothese fehlt.** Ein Trendfilter ist nur zeitweise investiert und hat deshalb automatisch weniger Drawdown als Buy & Hold. Das ist kein Verdienst, das ist Abwesenheit. Die richtige Kontrolle sind zufällige Ein- und Ausstiege mit *derselben Marktzeit*. Ichimoku lag gegen diese Kontrolle im 40. Perzentil — schlechter als Würfeln.

**3. Beobachtungen werden doppelt gezählt.** Bei R3 feuern SPY, QQQ, DIA und IWM am selben Tag — das ist ein Marktereignis, nicht vier Beobachtungen. Wer das ignoriert, überschätzt seine Sicherheit.

**4. Gebühren werden zu niedrig angesetzt.** Bei Ichimoku auf Krypto halbierte sich der Vorsprung, sobald statt 0,10% die realistischen 0,26% je Seite angesetzt wurden.

**5. Alles wird auf denselben Daten gemessen.** Der wichtigste Punkt, und der, an dem R3 gescheitert ist. Wer auf einer Datenmenge neun Parameterzellen und fünf Stresstests durchprobiert, findet irgendwo etwas. Erst ein Markt, in dem nie gesucht wurde, entscheidet.

Dazu kommt der Klassiker: **der eingebaute Blick in die Zukunft.** Ein falsches Vorzeichen in einer Zeitverschiebung genügt. Jedes Skript hier prüft das im Selbsttest — und prüft zusätzlich, dass die Prüfung auch anschlägt, indem sie gegen eine absichtlich falsche Implementierung läuft.

---

## Aufbau

```
strategien/     die getesteten Ansätze, jeder mit eigenem Selbsttest
pruefung/       die Prüfketten, die auf ein Rohergebnis angewandt werden
monitor/        Messwerkzeug ohne Handelsfunktion
```

Jedes Skript läuft eigenständig, hat `--selftest` und braucht keine API-Schlüssel. Kein Skript handelt oder greift auf ein Konto zu.

### strategien/

| Datei | Inhalt |
|---|---|
| `relative_strength_backtest.py` | Monatliches Ranking, Top 10 aus ~100 Werten, Fenster 6 Monate ohne den jüngsten |
| `ichimoku_backtest.py` | Ichimoku Kinko Hyo gegen Buy & Hold *und* gegen Zufallstiming mit gleicher Marktzeit |
| `fvg_backtest.py` | Fair Value Gap gegen Zufallszonen gleicher Breite, gleichen Abstands, gleicher Richtung |
| `r3_backtest.py` | R3 nach Larry Connors gegen Zufallseinstiege gleicher Haltedauer |
| `ema_crossover_backtest.py` | EMA-Crossover gegen Buy & Hold und Zufallstiming, Parameter-Gitter mit EMA und SMA nebeneinander |

### pruefung/

| Datei | Inhalt |
|---|---|
| `rs_validate.py` | Faire Vergleichsbasis, t-Test, Parameter-Gitter, Jahresaufschlüsselung, Konzentrationsanalyse |
| `rs_stress.py` | Treiberausschluss und Leave-one-out, Teilperioden, Long-Short, Beta-Adjustierung, Zufallsportfolios |
| `r3_stress.py` | Cluster-korrigierte Signifikanz, Portfolio-Simulation mit begrenzten Plätzen, Verlustverteilung, Gebührenstaffel |
| `r3_oos.py` | Out-of-Sample auf sechs Märkten, die im Haupttest nicht vorkamen — ohne jede Parametersuche |
| `ema_oos.py` | Out-of-Sample des EMA-Crossovers auf zehn neuen Coins, Parameter und Hürde fest im Code |

### monitor/

| Datei | Inhalt |
|---|---|
| `funding_monitor.py` | Funding-Raten der Kraken-Perpetuals, annualisiert, mit Break-even-Haltedauer und Schwellwert-Alarm |

---

## Installation und Start

```bash
pip install yfinance pandas numpy
./run_selftests.sh          # alle elf Selbsttests, ohne Netzverbindung
```

---

## Die Fälle im Detail

### R3 nach Larry Connors — bestanden, bestanden, bestanden, gescheitert

Die Regeln aus *High Probability ETF Trading* (2009): Kurs über der 200-Tage-Linie, der 2-Perioden-RSI fällt drei Tage in Folge, am ersten Tag lag er unter 60, heute schließt er unter 10. Ausstieg bei RSI über 70. Kein Stop-Loss.

```bash
python3 strategien/r3_backtest.py
python3 pruefung/r3_stress.py
python3 pruefung/r3_oos.py
```

**Stufe 1 — Zufallskontrolle.** 20 US-Werte ab 2000, 1565 Trades: +0,502% je Trade gegen +0,074% bei Zufallseinstiegen gleicher Haltedauer, t=5,64. 17 von 20 Werten positiv, alle neun Parameterzellen positiv.

**Stufe 2 — Cluster-Korrektur.** Nach Gruppierung auf Handelstage bleibt t=5,75, auf Quartale noch 3,74. Die Abhängigkeit zwischen gleichzeitigen Trades erklärt den Effekt nicht.

**Stufe 3 — Gewinnerausschluss und Gebühren.** Ohne AAPL, NVDA, V und MSFT bleiben 0,355% je Trade bei t=4,64. Bis 0,20% Gebühren je Seite trägt das Ergebnis, bei 0,30% kippt es.

**Stufe 4 — Out-of-Sample.** Sechs Märkte, die im Haupttest nicht vorkamen, mit festgeschriebenen Parametern und ohne jede Suche:

| Universum | Trades | Schnitt | Zufall | t | |
|---|---|---|---|---|---|
| Deutschland | 687 | 0,152% | 0,052% | 0,28 | — |
| Frankreich/NL | 650 | 0,416% | 0,038% | 1,99 | — |
| Großbritannien | 593 | 0,139% | −0,019% | 0,93 | — |
| Japan | 588 | 0,498% | 0,103% | 2,44 | bestanden |
| Kanada/Australien | 681 | 0,176% | 0,079% | 1,50 | — |
| Krypto | 268 | 3,889% | 0,692% | 1,85 | — |

Ein von sechs Universen, nötig waren vier. Im Zeitraum ab 2022 sind vier von sechs negativ — bei Frankreich/NL schneidet der Zufallseinstieg mit +0,290% besser ab als R3 mit −0,326%.

**Die Portfolio-Rechnung war schon vorher ernüchternd:** Mit fünf gleichzeitigen Positionen kommt R3 auf 4,67% p.a. gegen 8,24% bei Buy & Hold auf SPY. Der Sharpe ist mit 0,66 gegen 0,51 besser und der Drawdown mit −18,3% gegen −55,4% deutlich kleiner, aber das Kapital liegt zu 81% brach.

**Die Trefferquote als Verkaufsargument** — 68% Gewinntrades — wird im Selbsttest direkt entkräftet. Auf **driftfreien** Zufallsdaten, wo der Ertrag zwingend null sein muss und es auch ist (−0,06% bei einer Streuung von 0,42%), liegt die Trefferquote trotzdem bei 62,4% gegen 51,0% beim Zufallseinstieg. Die Quote entsteht vollständig aus der Ausstiegsregel: Man steigt aus, sobald man im Plus ist, und hält Verlierer bis zur Notbremse.

### Relative Strength — wie ein scheinbarer Vorteil verschwindet

Roh: 23,17% p.a. gegen 13,58% beim S&P 500, Sharpe 1,21 gegen 0,94.

```bash
python3 strategien/relative_strength_backtest.py
python3 pruefung/rs_validate.py
python3 pruefung/rs_stress.py
```

- Gegen ein gleichgewichtetes Halten derselben hundert Werte blieben 7,7% statt 9,6%.
- Ohne NVDA, AMD, TSLA, AVGO und NFLX: **−0,61%** bei t=−0,15. NVDA allein steckte in 54% aller Monate im Portfolio.
- Nach Teilperioden: +16,2% bis 2021, **−1,1% danach**.
- Marktneutral als Long-Short: 3,63% bei 22,8% Volatilität, Sharpe 0,27.

Der Backtest hatte nicht eine Strategie gemessen, sondern eine Halbleiter-Position, die ein Ranking rückblickend zugeteilt bekam.

### Ichimoku — warum ein besserer Sharpe nichts beweist

```bash
python3 strategien/ichimoku_backtest.py
```

Auf zehn Aktien und Indizes: 22,2% gegen 40,5% bei Buy & Hold, t=−2,52. Der Drawdown war mit −35,7% gegen −73,5% halbiert, der Sharpe minimal besser — das übliche Argument für Trendfolge.

200 zufällige Positionsreihen mit *derselben* Marktzeit und Handelszahl schnitten im Mittel besser ab: 40. Perzentil. Die Risikoreduktion kam vom Draußenbleiben, nicht vom Erkennen.

Auf acht Kryptowährungen sah es besser aus (84. Perzentil), schrumpfte aber auf +2,5%, sobald realistische Gebühren angesetzt wurden. Die vorab festgelegte Hürde von 90 wurde nicht erreicht. Die traditionellen Werte 9/26/52 stammen aus der japanischen Sechs-Tage-Handelswoche der 1930er Jahre und waren nicht einmal der beste der sechs geprüften Parametersätze.

### EMA-Crossover — knapp daneben ist daneben

Der exponentiell gleitende Durchschnitt gewichtet jüngere Kurse stärker und reagiert schneller als der einfache. Das übliche Argument: Der SMA-Crossover scheitert an seiner Trägheit, mit dem EMA klappt es. Gehandelt wird, solange EMA(12) über EMA(26) liegt.

```bash
python3 strategien/ema_crossover_backtest.py
python3 strategien/ema_crossover_backtest.py --tickers BTC-USD,ETH-USD,SOL-USD --kosten 0.0026
python3 pruefung/ema_oos.py
```

**Aktien:** −11,84% p.a. gegen Buy & Hold, 36. Perzentil gegen Zufallstiming. Im Parameter-Gitter liegen EMA und SMA bei 12/26 gleichauf (−11,84% gegen −11,94%). Die Trägheit war nicht das Problem.

**Krypto (BTC, ETH, SOL, 0,26% Gebühren):** +7,04% p.a., 96,8. Perzentil, fünf von sechs Parameterzellen positiv. Zwei Warnzeichen gab es schon hier: Der t-Wert der Tagesdifferenz war negativ, weil der Vorsprung aus geringerer Schwankung kam und nicht aus höheren Tagesrenditen, und er stammte fast nur aus der Zeit ab 2022.

**Out-of-Sample auf zehn neuen Coins** (ADA, XRP, DOGE, LTC, BCH, LINK, BNB, TRX, AVAX, DOT), Parameter und Hürde vorab im Code festgeschrieben:

| Hürde | Ergebnis | Nötig | |
|---|---|---|---|
| Mittleres Perzentil gegen Zufallstiming | 87,7 | 90 | — |
| Coins vor Buy & Hold | 9 von 10 | 7 | erfüllt |
| Vorsprung in beiden Teilperioden | bis 2022: −1,02% | beide positiv | — |

Im Mittel +4,22% p.a., aber bei einem t-Wert auf Log-Renditen von 0,23. Dazu kommt eine Verzerrung zugunsten der Strategie: Bei acht der zehn Coins beginnen die Daten im Dezember 2017, nahe am damaligen Hoch.

Was übrig bleibt, ist Risikominderung: ein größter Rückgang von −60,4% statt −89,5% bei 38–57% Marktzeit. Das war nicht die vorab gestellte Frage und müsste als eigener Versuch auf neuen Daten geprüft werden.

### Fair Value Gap — wenn eine Trefferquote nichts misst

```bash
python3 strategien/fvg_backtest.py
```

Die Formation gilt als zuverlässig, weil ihre Zonen zu 77–85% "gefüllt" werden. Dieselbe Messung an Zufallszonen mit identischer Breite, identischem Abstand und identischer Richtung, nur zu zufälligen Zeitpunkten: **79–83%**. Bei BTC und ETH lag die echte Formation darunter.

Der Handel verlor 0,451% je Trade über 4416 Trades bei t=−3,80 — ein statistisch belastbarer Verlust. Sieben von acht Werten negativ, alle neun Parameterkombinationen negativ.

### Funding-Rate-Carry — der eine Mechanismus, der wirklich existiert

```bash
python3 monitor/funding_monitor.py --now
python3 monitor/funding_monitor.py --history --days 90
```

Anders als die geprüften Signale hat dieser Ansatz einen nachvollziehbaren Mechanismus: Eine Börse weist alle acht Stunden tatsächlich eine Zahlung zwischen Long- und Short-Seite an. Spot long plus Perpetual short kassiert sie richtungsunabhängig.

Gemessen am 12.09.2026: BTC 0,27% p.a. bei einem Break-even von 841 Tagen, ETH 0,91%, SOL 0,62%, XRP negativ. Über 90 Tage netto nach Gebühren: 0,3–0,4% p.a. bei 26–46% Perioden mit negativer Rate.

Der Mechanismus stimmt, er wird derzeit nur nicht bezahlt. Deshalb läuft `--log` weiter und meldet sich, sobald eine Rate drei Messungen in Folge über 0,01% je acht Stunden liegt.

---

## Methodische Bausteine

Wiederverwendbar für jede weitere Strategie, ungefähr in der Reihenfolge ihrer Aussagekraft:

**Out-of-Sample schlägt alles andere.** Ein Markt, in dem nie gesucht wurde, entscheidet mehr als jeder Test auf den Ursprungsdaten. Wichtig dabei: keine Parametersuche. Das OOS-Skript durchsucht dafür seinen eigenen Quelltext nach Optimierungsschleifen.

**Zufallskontrolle mit angeglichenen Eigenschaften.** Gleiche Marktzeit, gleiche Haltedauer, gleiche Zonenbreite — alles gleich außer dem geprüften Merkmal.

**Cluster-korrigierte Signifikanz.** Gleichzeitige Trades in korrelierten Werten sind ein Ereignis. Gruppieren nach Tag, Woche, Monat und auf den Gruppenmitteln testen.

**Faire Vergleichsbasis statt Index.** Der Maßstab muss dieselben Verzerrungen tragen wie die Strategie.

**Rauschgrenze bestimmen.** Vor dem ersten echten Lauf auf Zufallsdaten messen, wie groß ein Scheinvorteil allein durch Streuung ausfällt. Bei Relative Strength waren das 2,7% p.a.

**Look-Ahead-Probe mit Gegenprobe.** Kurse nach einem Stichtag verändern, prüfen dass frühere Signale unverändert bleiben — und zusätzlich gegen eine absichtlich falsche Implementierung laufen lassen, um zu belegen, dass die Probe anschlägt.

**Hürde vorher festlegen.** Bei Ichimoku 90. Perzentil, erreicht wurden 84,2. Bei R3 vier von sechs Universen, erreicht wurde eins. Beim EMA-Crossover out-of-sample 90. Perzentil, erreicht wurden 87,7. Knapp daneben ist daneben.

**Treiberausschluss.** Einzelne Werte entfernen und prüfen, ob der Effekt überlebt.

---

## Grenzen

- Datenquelle ist yfinance, also Tagesdaten aus zweiter Hand. Für Intraday-Strategien ungeeignet.
- Die Ticker-Listen bestehen aus heutigen Index-Mitgliedern. Der Survivorship Bias ist im Code dokumentiert, aber nicht behoben — dafür bräuchte es historische Index-Zusammensetzungen.
- Fair Value Gap wird üblicherweise auf Minuten- und Stundencharts gehandelt. Das Tagesergebnis widerlegt die Tagesvariante sauber, die Intraday-Variante nur indirekt.
- Die Krypto-Kursreihen von yfinance beginnen für viele Coins erst im Dezember 2017, nahe einem Höchststand. Das benachteiligt Buy & Hold.
- Die Ausstiegsregeln sind bewusst einfach gehalten, meist ohne Stop-Loss.
- Keine Steuern, keine Slippage-Modellierung über den Spread hinaus.
- Ein negatives Out-of-Sample-Ergebnis widerlegt eine Strategie in den geprüften Märkten, nicht in allen denkbaren.

---

## Lizenz und Hinweis

Keine Anlageberatung. Der Code dient der methodischen Prüfung von Handelsstrategien und enthält keine Handelsfunktion.

---

## English summary

A toolkit for testing trading strategies against proper null hypotheses — and discarding them when they don't hold up.

This repository contains no working strategy. It contains the method by which five popular approaches were falsified, plus the numbers. That is deliberate: the value is in the validation chain, not in the result.

The most instructive case is R3 (Larry Connors, 2-period RSI mean reversion). It passed three stages — beating matched random entries at +0.50% per trade, surviving cluster-corrected significance at t=5.75, and holding at 0.355% after excluding the four strongest tickers. By any conventional standard it was a confirmed strategy. It then failed out-of-sample: across six markets never used during development, only one cleared the pre-registered bar, and four of six turned negative from 2022 onward.

An EMA crossover (12/26) on crypto was a near miss: it beat matched random timing at the 96.8th percentile on BTC, ETH and SOL, then reached only the 87.7th percentile out-of-sample on ten new coins against a pre-registered bar of 90, with no edge before 2022. What remained was lower drawdown, not higher return. On equities it failed immediately.

The other cases collapsed earlier. Cross-sectional relative strength looked strong at +9.6% p.a. against the S&P 500 until compared against an equal-weighted hold of the same hundred tickers and stripped of five individual stocks. Ichimoku ranked in the 40th percentile against random entries matched for time-in-market. Fair Value Gap zones filled at 77–85%, statistically indistinguishable from random zones of identical width and distance at 79–83%.

Every script runs standalone, includes a `--selftest` that works offline, verifies absence of look-ahead bias, and additionally verifies that this check actually fires by running it against a deliberately incorrect implementation. No script trades or touches an account.
