"""
EmaCrossoverCheck - freqtrade-Fassung des EMA-Crossovers aus quant-validation.

Zweck: NICHT eine neue Strategie suchen, sondern das eigene Messwerkzeug
gegen ein etabliertes stellen. Dieselben Regeln, dieselben Daten, zwei
unabhaengige Backtester. Weichen die Zahlen ab, hat einer von beiden
einen Fehler - und den zu finden ist der eigentliche Gewinn.

Die Regeln (identisch zu strategien/ema_crossover_backtest.py)
--------------------------------------------------------------
    EMA_t = Kurs_t * alpha + EMA_(t-1) * (1 - alpha),  alpha = 2/(N+1)
    entspricht pandas: close.ewm(span=N, adjust=False).mean()

    investiert      wenn EMA(12) >  EMA(26)
    nicht investiert wenn EMA(12) <= EMA(26)

    Long oder flach, keine Shorts. Kein Stop-Loss, kein Gewinnziel -
    ausgestiegen wird ausschliesslich beim Kreuzen.

Bewusst abgeschaltet
--------------------
    minimal_roi = {"0": 100.0}   ein Gewinnziel von 10.000 % ist nie erreichbar
    stoploss    = -0.99          greift praktisch nie
    trailing_stop = False

    Beides muss aus dem Weg, sonst misst man nicht mehr den Crossover,
    sondern freqtrades Ausstiegslogik.

Wo die beiden Rechnungen sich systematisch unterscheiden
--------------------------------------------------------
  1. Ausfuehrung. Das eigene Skript rechnet Schluss zu Schluss
     (Signal am Schluss von t, Rendite t -> t+1). freqtrade kauft zur
     EROEFFNUNG der Kerze t+1. Auf Tagesdaten ist das ein anderer
     Ausfuehrungskurs, nicht ein anderer Tag - beide vermeiden den
     Blick in die Zukunft.
  2. Datenquelle. Yahoo BTC-USD ist ein Mischkurs, Binance BTC/USDT ist
     eine einzelne Boerse mit UTC-Tagesgrenze. Kleine Abweichungen sind
     hier normal und KEIN Programmfehler.
  3. Zinseszins. freqtrade handelt mit dem vollen Kontostand
     ("unlimited"), das eigene Skript rechnet Renditereihen. Bei einem
     Paar und voller Investition laeuft das auf dasselbe hinaus.

Aufruf
------
    freqtrade backtesting --config user_data/config.json \
        --strategy EmaCrossoverCheck --timerange 20160101-
"""

from pandas import DataFrame

from freqtrade.strategy import IStrategy


class EmaCrossoverCheck(IStrategy):

    INTERFACE_VERSION = 3

    timeframe = "1d"
    can_short = False

    # Perioden wie im MACD und wie im eigenen Backtest
    fast_period = 12
    slow_period = 26

    # Nur der Crossover soll entscheiden:
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    trailing_stop = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    process_only_new_candles = True

    # Einschwingphase des EMA: dieselben 26 Kerzen, die das eigene
    # Skript auf "nicht investiert" setzt.
    startup_candle_count: int = 26

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = (
            dataframe["close"].ewm(span=self.fast_period, adjust=False).mean()
        )
        dataframe["ema_slow"] = (
            dataframe["close"].ewm(span=self.slow_period, adjust=False).mean()
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            dataframe["ema_fast"] > dataframe["ema_slow"],
            ["enter_long", "enter_tag"],
        ] = (1, "ema_fast_ueber_slow")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            dataframe["ema_fast"] <= dataframe["ema_slow"],
            ["exit_long", "exit_tag"],
        ] = (1, "ema_fast_unter_slow")
        return dataframe
