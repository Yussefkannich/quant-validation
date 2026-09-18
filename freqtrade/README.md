# freqtrade/ — die Gegenrechnung

Diese drei Dateien gehören zum Abschnitt *„Kreuzprüfung gegen freqtrade"* in der
Haupt-README. Sie enthalten alles, was nötig ist, um den Vergleich nachzurechnen.

Freqtrade selbst ist hier **nicht** enthalten. Es ist ein eigenständiges Projekt
unter <https://github.com/freqtrade/freqtrade>.

| Datei | Inhalt |
|---|---|
| `EmaCrossoverCheck.py` | dieselbe Regel wie `strategien/ema_crossover_backtest.py`, als freqtrade-Strategie |
| `config.json` | Dry-Run, Binance, Tageskerzen, 0,10% je Order, keine Schlüssel |

## Nachrechnen

```bash
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
conda create -n freqtrade python=3.12 -y && conda activate freqtrade
pip install -e .

cp /pfad/zu/quant-validation/freqtrade/config.json user_data/
mkdir -p user_data/strategies
cp /pfad/zu/quant-validation/freqtrade/EmaCrossoverCheck.py user_data/strategies/

freqtrade download-data --config user_data/config.json --timeframe 1d --timerange 20150101-
freqtrade backtesting --config user_data/config.json --strategy EmaCrossoverCheck \
    --timerange 20160101- --pairs BTC/USDT
```

Und die Gegenseite, aus dem Wurzelverzeichnis dieses Repos:

```bash
python3 strategien/ema_crossover_backtest.py --tickers BTC-USD --start 2017-08-17 --kosten 0.0010
```

`2017-08-17` ist der erste Tag, an dem Binance BTC/USDT führt. Nach den 26 Kerzen
Einschwingphase beginnt der Handel am 12.09.2017 — genau dort, wo auch der
freqtrade-Lauf anfängt.

## Was beim Vergleich zu beachten ist

**Gewinnziel und Stop-Loss sind abgeschaltet** (`minimal_roi = {"0": 100.0}`,
`stoploss = -0.99`). Ohne das misst man nicht mehr den Crossover, sondern
freqtrades Ausstiegslogik.

**Zwei Sharpe-Werte.** freqtrade meldet „Sharpe (closed trades)" und „Sharpe
(daily wallet balance)". Vergleichbar mit `sharpe()` aus diesem Repo ist der
zweite — der erste rechnet über 50 unterschiedlich große Trades und ist bei
dieser Stichprobe ohne Aussage.

**Erwartbare Abweichungen**, die kein Fehler sind: Yahoo führt für `BTC-USD`
einen Mischkurs mehrerer Börsen mit Tagesgrenze nach New Yorker Zeit, Binance
eine einzelne Börse mit UTC-Grenze und USDT statt Dollar. Dazu kommt der
Ausführungszeitpunkt — dieses Repo rechnet Schluss zu Schluss, freqtrade kauft
zur Eröffnung der Folgekerze.

**Verwendete Versionen.** freqtrade 2026.8-dev-ba634ac40, ccxt 4.5.78,
Python 3.12.13. Ein Entwicklungsstand ist für einen Vergleich brauchbar, aber
bei einer unerklärlichen Abweichung der erste Verdächtige.
