# quant-validation

Ein Werkzeugkasten, um Handelsstrategien ergebnisoffen zu prüfen — und sie fallenzulassen, wenn sie nicht tragen.

Dieses Repository enthält keine Strategie, die funktioniert. Es enthält die Methode, mit der drei populäre Ansätze widerlegt wurden, und die Zahlen dazu. Das ist Absicht: Der Wert liegt in der Prüfkette, nicht im Ergebnis.

*An English summary follows at the end.*

---

## Ergebnisse auf einen Blick

| Ansatz | Rohergebnis | Nach Prüfung | Befund |
|---|---|---|---|
| Cross-Sectional Relative Strength (Aktien) | +9,6% p.a. gegen SPY, Sharpe 1,21 | −0,61% ohne fünf Einzelwerte | verworfen |
| Ichimoku Kinko Hyo (Aktien) | — | −18,3% p.a. gegen Buy & Hold | verworfen |
| Ichimoku Kinko Hyo (Krypto) | +4,8% p.a. bei 0,10% Gebühren | +2,5% bei realistischen 0,26% | verworfen |
| Fair Value Gap / ICT (Tagesbasis) | Füllquote 77–85% | Zufallszonen: 79–83% | verworfen |
| Funding-Rate-Carry (Perpetuals) | Werbung: 10–30% p.a. | gemessen: 0,3–0,4% p.a. netto | derzeit unattraktiv |

Jede Zeile ist reproduzierbar. Die Befehle stehen weiter unten.

---

## Warum das nötig ist

Ein Backtest liefert fast immer eine Zahl, die gut aussieht. Der Grund ist selten ein Vorteil der Strategie, sondern meistens einer von vier Fehlern:

**1. Der Vergleichsmaßstab ist falsch.** Relative Strength schlug den S&P 500 um 9,6% pro Jahr. Verglichen mit einem gleichgewichteten Halten *derselben hundert Aktien* — also derselben Auswahl, die rückblickend als erfolgreich bekannt war — blieben 7,7% übrig. Und nach Ausschluss von fünf Einzelwerten waren es −0,61%.

**2. Die Nullhypothese fehlt.** Ein Trendfilter ist nur zeitweise investiert und hat deshalb automatisch weniger Drawdown als Buy & Hold. Das ist kein Verdienst, das ist Abwesenheit. Die richtige Kontrolle sind zufällige Ein- und Ausstiege mit *derselben Marktzeit*. Ichimoku lag gegen diese Kontrolle im 40. Perzentil — schlechter als Würfeln.

**3. Gebühren werden zu niedrig angesetzt.** Bei Ichimoku auf Krypto halbierte sich der Vorsprung, sobald statt 0,10% die realistischen 0,26% je Seite angesetzt wurden.

**4. Der Blick in die Zukunft ist eingebaut.** Ein falsches Vorzeichen in einer Zeitverschiebung genügt, und der Backtest handelt mit Information, die es damals nicht gab. Jedes Skript hier prüft das im Selbsttest — und prüft zusätzlich, dass die Prüfung auch anschlägt, indem sie gegen eine absichtlich falsche Implementierung läuft.

---

## Aufbau

```
strategien/     die getesteten Ansätze, jeder mit eigenem Selbsttest
pruefung/       die Prüfkette, die auf ein Rohergebnis angewandt wird
monitor/        Messwerkzeug ohne Handelsfunktion
```

Jedes Skript läuft eigenständig, hat `--selftest` und braucht keine API-Schlüssel. Kein Skript handelt oder greift auf ein Konto zu.

### strategien/

| Datei | Inhalt |
|---|---|
| `relative_strength_backtest.py` | Monatliches Ranking, Top 10 aus ~100 Werten, Fenster 6 Monate ohne den jüngsten |
| `ichimoku_backtest.py` | Ichimoku Kinko Hyo gegen Buy & Hold *und* gegen Zufallstiming mit gleicher Marktzeit |
| `fvg_backtest.py` | Fair Value Gap gegen Zufallszonen gleicher Breite, gleichen Abstands, gleicher Richtung |

### pruefung/

| Datei | Inhalt |
|---|---|
| `rs_validate.py` | Faire Vergleichsbasis, t-Test, Parameter-Gitter, Jahresaufschlüsselung, Konzentrationsanalyse |
| `rs_stress.py` | Treiberausschluss und Leave-one-out, Teilperioden, Long-Short, Beta-Adjustierung, Zufallsportfolios |

### monitor/

| Datei | Inhalt |
|---|---|
| `funding_monitor.py` | Funding-Raten der Kraken-Perpetuals, annualisiert, mit Break-even-Haltedauer und Schwellwert-Alarm |

---

## Installation und Start

```bash
pip install yfinance pandas numpy

# Selbsttests zuerst - laufen ohne Netzverbindung
python3 strategien/relative_strength_backtest.py --selftest
python3 strategien/ichimoku_backtest.py --selftest
python3 strategien/fvg_backtest.py --selftest
python3 pruefung/rs_validate.py --selftest
python3 pruefung/rs_stress.py --selftest
python3 monitor/funding_monitor.py --selftest
```

Oder alle auf einmal:

```bash
./run_selftests.sh
```

---

## Die drei Fälle im Detail

### Relative Strength — wie ein scheinbarer Vorteil verschwindet

Roh: 23,17% p.a. gegen 13,58% beim S&P 500, Sharpe 1,21 gegen 0,94. Ein Ergebnis, das in vielen Repositories als Erfolg stehen bliebe.

```bash
python3 strategien/relative_strength_backtest.py
python3 pruefung/rs_validate.py
python3 pruefung/rs_stress.py
```

Was die Prüfung ergab:

- Gegen ein gleichgewichtetes Halten derselben hundert Werte blieben 7,7% statt 9,6%.
- Ohne NVDA, AMD, TSLA, AVGO und NFLX: **−0,61%** bei t = −0,15. NVDA allein steckte in 54% aller Monate im Portfolio.
- Nach Teilperioden: +16,2% bis 2021, **−1,1% danach**.
- Marktneutral als Long-Short: 3,63% bei 22,8% Volatilität, Sharpe 0,27 — schlechter als gleichgewichtetes Halten.

Der Backtest hatte nicht eine Strategie gemessen, sondern eine Halbleiter-Position, die ein Ranking rückblickend zugeteilt bekam.

### Ichimoku — warum ein besserer Sharpe nichts beweist

```bash
python3 strategien/ichimoku_backtest.py
python3 strategien/ichimoku_backtest.py --tickers BTC-USD,ETH-USD,SOL-USD,XRP-USD,ADA-USD,LTC-USD,DOGE-USD,LINK-USD --start 2018-01-01 --kosten 0.0026
```

Auf zehn Aktien und Indizes: 22,2% gegen 40,5% bei Buy & Hold, t = −2,52. Der Drawdown war mit −35,7% gegen −73,5% allerdings halbiert, der Sharpe minimal besser — das übliche Argument für Trendfolge.

Genau hier greift die Zufallskontrolle. 200 zufällige Positionsreihen mit *derselben* Marktzeit und Handelszahl schnitten im Mittel besser ab als Ichimoku: 40. Perzentil. Die Risikoreduktion kam vom Draußenbleiben, nicht vom Erkennen.

Auf acht Kryptowährungen sah es zunächst besser aus (84. Perzentil, +4,8% p.a.), schrumpfte aber auf +2,5%, sobald realistische Gebühren angesetzt wurden. Die vorab festgelegte Hürde von 90 wurde nicht erreicht.

Alle sechs geprüften Parametersätze verloren. Die traditionellen Werte 9/26/52 stammen aus der japanischen Sechs-Tage-Handelswoche der 1930er Jahre und waren nicht einmal der beste davon.

### Fair Value Gap — wenn eine Trefferquote nichts misst

```bash
python3 strategien/fvg_backtest.py
```

Die Formation gilt als zuverlässig, weil Zonen mit hoher Quote „gefüllt" werden. Gemessen: 77–85%.

Dieselbe Messung an Zufallszonen mit identischer Breite, identischem Abstand und identischer Richtung, nur zu zufälligen Zeitpunkten: **79–83%**. Bei BTC und ETH lag die echte Formation sogar darunter.

Der Handel selbst verlor 0,451% je Trade über 4416 Trades bei t = −3,80 — ein statistisch belastbarer Verlust, nicht bloß ein Nullergebnis. Sieben von acht Werten negativ, alle neun Parameterkombinationen negativ.

Interessant blieb ein Detail: Echte Zonen schnitten um 0,310% je Trade besser ab als Zufallszonen (t = 2,55). Der Unterschied besteht aber zwischen „verliert viel" und „verliert weniger" — vermutlich ein Effekt des Marktzustands nach schnellen Bewegungen, keine Eigenschaft der Formation.

### Funding-Rate-Carry — der eine Mechanismus, der wirklich existiert

```bash
python3 monitor/funding_monitor.py --now
python3 monitor/funding_monitor.py --history --days 90
```

Anders als die drei Chartmuster hat dieser Ansatz einen nachvollziehbaren Mechanismus: Eine Börse weist alle acht Stunden tatsächlich eine Zahlung zwischen Long- und Short-Seite an. Spot long plus Perpetual short kassiert diese Zahlung richtungsunabhängig.

Gemessen am 12.09.2026: BTC 0,27% p.a. bei einem Break-even von 841 Tagen, ETH 0,91%, SOL 0,62%, XRP negativ. Über 90 Tage netto nach Gebühren: 0,3–0,4% p.a., bei 26–46% Perioden mit negativer Rate.

Der Mechanismus stimmt, er wird derzeit nur nicht bezahlt. Deshalb läuft `--log` weiter und meldet sich, sobald eine Rate drei Messungen in Folge über 0,01% je acht Stunden liegt — der Schwelle, ab der der Break-even bei 21 Tagen liegt.

---

## Methodische Bausteine

Wiederverwendbar für jede weitere Strategie:

**Rauschgrenze bestimmen.** Vor dem ersten echten Lauf auf Zufallsdaten messen, wie groß ein Scheinvorteil allein durch Streuung ausfällt. Bei Relative Strength waren das 2,7% p.a. — jedes kleinere Ergebnis ist bedeutungslos.

**Faire Vergleichsbasis statt Index.** Der Maßstab muss dieselben Verzerrungen tragen wie die Strategie.

**Zufallskontrolle mit angeglichenen Eigenschaften.** Gleiche Marktzeit, gleiche Handelszahl, gleiche Zonenbreite — alles gleich außer dem geprüften Merkmal.

**Look-Ahead-Probe mit Gegenprobe.** Kurse nach einem Stichtag verändern und prüfen, dass die Signale davor unverändert bleiben. Zusätzlich gegen eine absichtlich falsche Implementierung laufen lassen, um zu belegen, dass die Probe überhaupt anschlägt.

**Hürde vorher festlegen.** Bei Ichimoku waren es 90. Perzentil und beide Teilperioden positiv. Erreicht wurden 84,2 — knapp daneben ist daneben.

**Treiberausschluss.** Einzelne Werte entfernen und prüfen, ob der Effekt überlebt.

---

## Grenzen

- Datenquelle ist yfinance, also Tagesdaten aus zweiter Hand. Für Intraday-Strategien ungeeignet.
- Die Ticker-Listen bestehen aus heutigen Index-Mitgliedern. Der daraus folgende Survivorship Bias ist im Code dokumentiert, aber nicht behoben — dafür bräuchte es historische Index-Zusammensetzungen.
- Fair Value Gap wird üblicherweise auf Minuten- und Stundencharts gehandelt. Das Tagesergebnis widerlegt die Tagesvariante sauber, die Intraday-Variante nur indirekt.
- Die Ausstiegsregeln sind bewusst einfach gehalten (feste Haltedauer, kein Stopp-Loss).
- Keine Steuern, keine Slippage-Modellierung über den Spread hinaus.

---

## Lizenz und Hinweis

Keine Anlageberatung. Der Code dient der methodischen Prüfung von Handelsstrategien und enthält keine Handelsfunktion.

---

## English summary

A toolkit for testing trading strategies against proper null hypotheses — and discarding them when they don't hold up.

This repository contains no working strategy. It contains the method by which three popular approaches were falsified, plus the numbers. That is deliberate: the value is in the validation chain, not in the result.

Cross-sectional relative strength looked strong at +9.6% p.a. against the S&P 500 — until compared against an equal-weighted hold of the same hundred tickers, and until five individual stocks were excluded, at which point it turned negative. Ichimoku ranked in the 40th percentile against random entries matched for time-in-market. Fair Value Gap zones were filled at 77–85%, statistically indistinguishable from random zones of identical width and distance at 79–83%, while trading them lost 0.451% per trade across 4416 trades.

Every script runs standalone, includes a `--selftest` that works offline, verifies absence of look-ahead bias, and additionally verifies that this check actually fires by running it against a deliberately incorrect implementation. No script trades or touches an account.
