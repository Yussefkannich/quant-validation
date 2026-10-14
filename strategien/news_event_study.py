#!/usr/bin/env python3
"""
news_event_study.py - Wie reagiert Bitcoin auf Nachrichten-Ereignisse?

Reines MESSSKRIPT, kein Handel. Es beantwortet eine Frage:
  Wenn ueber ein Thema (Zinsentscheid, Boersen-Hack, Bitcoin-ETF ...) ploetzlich
  ungewoehnlich viel berichtet wird - bewegt sich BTC danach anders als an
  zufaelligen Tagen? Und reicht das nach Gebuehren fuer long ODER short?

Datenquellen (beide kostenlos, ohne Schluessel):
  - GDELT DOC 2.0 API: weltweite Nachrichten (65 Sprachen, maschinell
    uebersetzt), pro Tag: Artikelzahl, Gesamtzahl aller Artikel, mittlere Tonalitaet.
    Historie ab 2017. Limit: hoechstens eine Anfrage alle 5 Sekunden.
  - BTC-Tageskurse: Bitstamp (ab 2011), Ersatz Binance.

Ablauf:
  python3 news_event_study.py --selftest     # ohne Netz, prueft die Methode
  python3 news_event_study.py --fetch        # laedt GDELT + BTC in den Cache
                                             # (18 Abfragen; GDELT drosselt oft stark -
                                             #  abbrechbar, der naechste Lauf setzt fort)
  python3 news_event_study.py --run          # Auswertung -> ergebnis.txt, events.csv

Methode (vorab festgelegt, siehe HUERDE weiter unten):
  1. Ereignis = Tag, an dem der Anteil eines Themas an ALLEN Nachrichten mindestens
     Z_MIN robuste Standardabweichungen ueber dem Median der 60 VORHERIGEN Tage
     liegt (nur Vergangenheit, kein Blick in die Zukunft). Danach COOLDOWN_DAYS
     Sperre pro Thema, damit eine Geschichte nicht fuenfmal zaehlt.
  2. Richtung der Meldung = Tonalitaet des Tages gegen den Mittelwert der
     60 Vortage (negativer / positiver als sonst bei diesem Thema).
  3. Einstieg zum BTC-Tagesschluss des Ereignistages (00:00 UTC; GDELT hat den Tag
     dann gerade abgeschlossen). Die Bewegung AM Ereignistag selbst zaehlt also
     nicht mit - die ist zum Zeitpunkt der Nachricht schon passiert.
  4. Vergleich: mittlere BTC-Rendite nach Ereignissen gegen Zufallstage aus
     demselben Zeitraum (Permutationstest).
  5. In-Sample bis 31.12.2022 legt Richtung (long/short) fest,
     Out-of-Sample ab 01.01.2023 muss bestaetigen - ohne Parameteraenderung.
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Einstellungen
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
GDELT_DIR = os.path.join(CACHE_DIR, "gdelt")
BTC_FILE = os.path.join(CACHE_DIR, "btc_daily.csv")
RESULT_FILE = os.path.join(HERE, "ergebnis.txt")
EVENTS_FILE = os.path.join(HERE, "events.csv")

DATA_START = date(2017, 1, 1)        # GDELT DOC-Timeline reicht bis 2017 zurueck
ANALYSIS_START = date(2017, 3, 15)   # erst wenn 60 Tage Vorlauf da sind
OOS_START = date(2023, 1, 1)         # ab hier Out-of-Sample
CHUNK_DAYS = 4000                    # ganzer Zeitraum in EINER Abfrage (getestet: 2017-2026
                                     # kommt in Tagesaufloesung) -> 18 statt 720 Abfragen
GDELT_PAUSE = 8.0                    # Sekunden zwischen Anfragen (Limit: 5, mit Puffer)
GDELT_MAX_TRIES = 6                  # Versuche je Abfrage
GDELT_GIVE_UP_STREAK = 3             # so viele Abfragen in Folge erfolglos -> Abbruch

# Themen als GDELT-Suchanfragen. Englische Begriffe genuegen: GDELT durchsucht
# die maschinellen Uebersetzungen aller 65 Sprachen, deutsche Zeitungen inklusive.
# (a OR b) = eines davon; Leerzeichen zwischen Bloecken = UND; "..." = exakte Phrase.
CATEGORIES = {
    "fed_zinsen":        '"federal reserve" ("rate hike" OR "rate cut" OR "raises rates" OR "cuts rates" OR "interest rate decision")',
    "inflation":         '("consumer price index" OR "inflation data" OR "inflation report" OR "inflation figures")',
    "krypto_regulierung": '(bitcoin OR cryptocurrency) ("securities and exchange commission" OR regulators OR crackdown OR "crypto ban")',
    "bitcoin_etf":       '("bitcoin etf" OR "spot bitcoin" OR "bitcoin exchange-traded")',
    "hack_pleite":       '(cryptocurrency OR bitcoin OR "crypto exchange") (hacked OR hackers OR bankruptcy OR insolvency OR "withdrawals halted")',
    "firmen_kaufen_btc": '("buys bitcoin" OR "bought bitcoin" OR "bitcoin purchase" OR "bitcoin treasury" OR "adds bitcoin")',
    "staat_adoption":    'bitcoin ("legal tender" OR "strategic reserve" OR "national reserve")',
    "bankenkrise":       '("bank run" OR "bank collapse" OR "bank failure" OR "banking crisis")',
    "krieg_geopolitik":  '(invasion OR "declares war" OR "military strike" OR airstrikes OR "missile attack")',
}

# Ereignis-Erkennung
BASELINE_DAYS = 60
BASELINE_MIN = 30
Z_MIN = 3.0
MIN_ARTICLES = 10
COOLDOWN_DAYS = 7

# Messung
HORIZONS = [1, 3, 7]
PRIMARY_HORIZON = 3       # Huerde wird NUR auf diesem Horizont geprueft
ENTRY_LAG = 0             # 0 = Schluss des Ereignistages; 1 = einen Tag spaeter (Info)
COST_ROUNDTRIP = 0.0052   # 2 x 0,26 % Taker (Kraken Spot), konservativ
N_PERM = 10000

# --------------------------------------------------------------------------
# HUERDE - vor dem ersten echten Lauf festgelegt, danach nicht mehr aendern.
# --------------------------------------------------------------------------
HURDLE = {
    "min_n_is": 15,          # mind. 15 Ereignisse In-Sample je Zelle
    "holm_alpha_is": 0.10,   # Holm-korrigiert ueber alle getesteten Zellen
    "min_n_oos": 10,         # mind. 10 Ereignisse Out-of-Sample
    "p_oos": 0.10,           # einseitig, in der In-Sample-Richtung
    "net_oos_positive": True # OOS netto nach Gebuehren > 0 in der gewaehlten Richtung
}


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# GDELT laden
# --------------------------------------------------------------------------
def _gdelt_url(query, mode, start, end):
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d") + "000000",
        "enddatetime": end.strftime("%Y%m%d") + "235959",
    }
    return "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)


def _http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "news-event-study/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_gdelt_timeline(text):
    """GDELT-JSON -> Liste von dicts {date, value, norm}. Leere Liste bei 'keine Treffer'.
    Wirft ValueError bei Fehlermeldung statt JSON (z.B. Rate-Limit)."""
    text = text.strip()
    if not text:
        return []
    if not text.startswith("{"):
        raise ValueError(text[:200])
    obj = json.loads(text)
    out = []
    for series in obj.get("timeline", []) or []:
        for row in series.get("data", []) or []:
            d = datetime.strptime(row["date"][:8], "%Y%m%d").date()
            out.append({"date": d.isoformat(), "value": row.get("value"), "norm": row.get("norm")})
        break  # nur die erste Serie (Artikelzahl bzw. Tonalitaet)
    return out


def fetch_gdelt(end_date=None):
    os.makedirs(GDELT_DIR, exist_ok=True)
    end_date = end_date or (datetime.now(timezone.utc).date() - timedelta(days=1))
    chunks = []
    s = DATA_START
    while s <= end_date:
        e = min(s + timedelta(days=CHUNK_DAYS - 1), end_date)
        chunks.append((s, e))
        s = e + timedelta(days=1)
    todo = []
    for cat in CATEGORIES:
        for mode in ("timelinevolraw", "timelinetone"):
            for s, e in chunks:
                fn = os.path.join(GDELT_DIR, f"{cat}_{mode}_{s:%Y%m%d}_{e:%Y%m%d}.json")
                # fertige Fenster nie neu holen; das laufende nur, wenn aelter als 12 Stunden
                if os.path.exists(fn) and (e < end_date - timedelta(days=2)
                                           or time.time() - os.path.getmtime(fn) < 12 * 3600):
                    continue
                todo.append((cat, mode, s, e, fn))
    log(f"GDELT: {len(todo)} Abfragen offen (ca. {len(todo) * GDELT_PAUSE / 60:.0f} Minuten). "
        f"Abbruch mit Strg+C ist sicher, der naechste --fetch setzt fort.")
    ok_count, failed, streak = 0, 0, 0
    for i, (cat, mode, s, e, fn) in enumerate(todo, 1):
        url = _gdelt_url(CATEGORIES[cat], mode, s, e)
        success = False
        for attempt in range(GDELT_MAX_TRIES):
            try:
                rows = parse_gdelt_timeline(_http_get(url))
                with open(fn, "w") as f:
                    json.dump(rows, f)
                success = True
                break
            except ValueError as ex:
                msg = str(ex)
                if "limit requests" in msg.lower() or "one every" in msg.lower():
                    wait = _backoff(attempt)
                    log(f"  Rate-Limit (Text), Versuch {attempt + 1}/{GDELT_MAX_TRIES}, warte {wait}s ...")
                    time.sleep(wait)
                    continue
                log(f"  FEHLER {cat}/{mode} {s}..{e}: {msg[:150]}")
                break
            except urllib.error.HTTPError as ex:
                wait = _backoff(attempt) if ex.code == 429 else 20 * (attempt + 1)
                log(f"  HTTP {ex.code}, Versuch {attempt + 1}/{GDELT_MAX_TRIES}, warte {wait}s ...")
                time.sleep(wait)
            except Exception as ex:  # Netz
                wait = 20 * (attempt + 1)
                log(f"  Netzfehler ({ex}), Versuch {attempt + 1}/{GDELT_MAX_TRIES}, warte {wait}s ...")
                time.sleep(wait)
        if success:
            ok_count += 1
            streak = 0
        else:
            failed += 1
            streak += 1
            log(f"  UEBERSPRUNGEN {cat}/{mode} {s}..{e} - wird beim naechsten --fetch erneut versucht")
            if streak >= GDELT_GIVE_UP_STREAK:
                log(f"\nGDELT sperrt gerade dauerhaft ({streak} Abfragen in Folge ohne Erfolg). "
                    f"Abbruch, damit die Sperre nicht verlaengert wird.\n"
                    f"Bitte in 1-2 Stunden einfach erneut starten - fertige Teile bleiben erhalten.")
                break
        if i % 20 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} abgearbeitet (geladen {ok_count}, uebersprungen {failed})")
        time.sleep(GDELT_PAUSE)
    rest = len(todo) - ok_count
    log("GDELT vollstaendig." if rest == 0 else
        f"GDELT: {ok_count} geladen, {rest} noch offen -> --fetch spaeter erneut starten.")


def _backoff(attempt):
    """429-Wartezeit: 60, 120, 240, 300, 300 ... Sekunden."""
    return min(300, 60 * 2 ** attempt)


def load_gdelt():
    """-> DataFrame index=date, Spalten <cat>_count, <cat>_norm, <cat>_tone"""
    frames = {}
    if not os.path.isdir(GDELT_DIR):
        return pd.DataFrame()
    for fn in sorted(os.listdir(GDELT_DIR)):
        if not fn.endswith(".json"):
            continue
        parts = fn[:-5].split("_")
        mode = parts[-3]
        cat = "_".join(parts[:-3])
        with open(os.path.join(GDELT_DIR, fn)) as f:
            rows = json.load(f)
        for r in rows:
            d = r["date"]
            if mode == "timelinevolraw":
                frames.setdefault(f"{cat}_count", {})[d] = r["value"]
                frames.setdefault(f"{cat}_norm", {})[d] = r["norm"]
            else:
                frames.setdefault(f"{cat}_tone", {})[d] = r["value"]
    df = pd.DataFrame(frames)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# --------------------------------------------------------------------------
# BTC laden
# --------------------------------------------------------------------------
def fetch_btc():
    os.makedirs(CACHE_DIR, exist_ok=True)
    start_ts = int(datetime(2016, 10, 1, tzinfo=timezone.utc).timestamp())
    closes = {}
    try:
        ts = start_ts
        while True:
            url = f"https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400&limit=1000&start={ts}"
            data = json.loads(_http_get(url))["data"]["ohlc"]
            if not data:
                break
            for c in data:
                d = datetime.fromtimestamp(int(c["timestamp"]), tz=timezone.utc).date()
                closes[d.isoformat()] = float(c["close"])
            last = int(data[-1]["timestamp"])
            if last <= ts or len(data) < 2:
                break
            ts = last + 86400
            time.sleep(1)
        source = "bitstamp"
    except Exception as ex:
        log(f"Bitstamp nicht erreichbar ({ex}), versuche Binance ...")
        closes = {}
        ms = start_ts * 1000
        while True:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000&startTime={ms}"
            data = json.loads(_http_get(url))
            if not data:
                break
            for k in data:
                d = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).date()
                closes[d.isoformat()] = float(k[4])
            ms = data[-1][0] + 86400000
            if len(data) < 1000:
                break
            time.sleep(1)
        source = "binance"
    s = pd.Series(closes).sort_index()
    # laufender Tag ist noch nicht geschlossen
    today = datetime.now(timezone.utc).date().isoformat()
    s = s[s.index < today]
    s.to_frame("close").to_csv(BTC_FILE, index_label="date")
    log(f"BTC: {len(s)} Tageskurse von {source} ({s.index[0]} bis {s.index[-1]})")


def load_btc():
    df = pd.read_csv(BTC_FILE, parse_dates=["date"]).set_index("date")
    return df["close"].sort_index()


# --------------------------------------------------------------------------
# Ereignisse erkennen (nur Vergangenheit!)
# --------------------------------------------------------------------------
def _rolling_mad(x):
    m = np.median(x)
    return np.median(np.abs(x - m))


def detect_events(gd, cat):
    """-> DataFrame der Ereignistage fuer ein Thema mit z, count, tone_delta."""
    if f"{cat}_count" not in gd:
        return pd.DataFrame()
    cnt = gd[f"{cat}_count"].astype(float)
    norm = gd.get(f"{cat}_norm")
    share = cnt / norm.replace(0, np.nan) if norm is not None else cnt
    share = share.fillna(0.0)
    past = share.shift(1)  # Basislinie sieht den heutigen Tag NICHT
    med = past.rolling(BASELINE_DAYS, min_periods=BASELINE_MIN).median()
    mad = past.rolling(BASELINE_DAYS, min_periods=BASELINE_MIN).apply(_rolling_mad, raw=True)
    denom = np.maximum(1.4826 * mad, 0.1 * med).replace(0, np.nan)
    z = (share - med) / denom

    tone = gd.get(f"{cat}_tone")
    if tone is not None:
        tone = tone.astype(float)
        tone_base = tone.shift(1).rolling(BASELINE_DAYS, min_periods=BASELINE_MIN).mean()
        tone_delta = tone - tone_base
    else:
        tone_delta = pd.Series(0.0, index=gd.index)

    cand = (z >= Z_MIN) & (cnt >= MIN_ARTICLES)
    events, last = [], None
    for d in cand[cand].index:
        if last is not None and (d - last).days < COOLDOWN_DAYS:
            continue
        events.append({"date": d, "category": cat, "count": int(cnt[d]), "z": float(z[d]),
                       "tone_delta": float(tone_delta.get(d, np.nan))})
        last = d
    return pd.DataFrame(events)


# --------------------------------------------------------------------------
# Renditen und Tests
# --------------------------------------------------------------------------
def forward_returns(close, h, lag=0):
    """r[d] = log(close[d+lag+h] / close[d+lag]). Einstieg zum Schluss von Tag d+lag."""
    c = np.log(close)
    return (c.shift(-(lag + h)) - c.shift(-lag)).dropna()


def perm_pvalues(event_r, pool_r, n_perm, rng):
    """Mittelwert der Ereignis-Renditen gegen Zufallstage gleicher Anzahl.
    -> (excess, p_zweiseitig, p_einseitig_oben, p_einseitig_unten)"""
    n = len(event_r)
    pool = np.asarray(pool_r)
    mu = pool.mean()
    obs = np.mean(event_r) - mu
    idx = rng.integers(0, len(pool), size=(n_perm, n))
    sims = pool[idx].mean(axis=1) - mu
    p_two = (np.sum(np.abs(sims) >= abs(obs)) + 1) / (n_perm + 1)
    p_up = (np.sum(sims >= obs) + 1) / (n_perm + 1)
    p_dn = (np.sum(sims <= obs) + 1) / (n_perm + 1)
    return obs, p_two, p_up, p_dn


def holm(pvals):
    """Holm-Korrektur, gibt angepasste p-Werte in Originalreihenfolge zurueck."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def analyze(events, close, horizon=PRIMARY_HORIZON, lag=ENTRY_LAG, n_perm=N_PERM,
            oos_start=OOS_START, seed=42, analysis_start=ANALYSIS_START):
    """Zellen = Thema x Tonrichtung. -> DataFrame mit Huerden-Urteil."""
    rng = np.random.default_rng(seed)
    fr = forward_returns(close, horizon, lag)
    fr = fr[fr.index >= pd.Timestamp(analysis_start)]
    oos_ts = pd.Timestamp(oos_start)
    pool_is, pool_oos = fr[fr.index < oos_ts], fr[fr.index >= oos_ts]

    ev = events.copy()
    ev["tone"] = np.where(ev["tone_delta"] < 0, "negativ", "positiv")
    ev = ev[ev["date"].isin(fr.index)]

    rows = []
    for (cat, tone), g in ev.groupby(["category", "tone"]):
        r = fr.loc[g["date"]]
        r_is, r_oos = r[r.index < oos_ts], r[r.index >= oos_ts]
        row = {"thema": cat, "ton": tone, "n_is": len(r_is), "n_oos": len(r_oos)}
        if len(r_is) >= 2:
            ex, p2, _, _ = perm_pvalues(r_is.values, pool_is.values, n_perm, rng)
            row.update(excess_is=ex, p_is=p2)
        else:
            row.update(excess_is=np.nan, p_is=np.nan)
        side = 1 if row["excess_is"] > 0 else -1 if row["excess_is"] < 0 else 0
        row["seite"] = {1: "long", -1: "short", 0: "-"}[side]
        if len(r_oos) >= 2 and len(pool_oos) > 0:
            ex_o, _, p_up, p_dn = perm_pvalues(r_oos.values, pool_oos.values, n_perm, rng)
            row["excess_oos"] = ex_o
            row["p_oos"] = p_up if side == 1 else p_dn if side == -1 else np.nan
            row["netto_oos"] = float(np.mean(side * (np.exp(r_oos.values) - 1)) - COST_ROUNDTRIP)
        else:
            row.update(excess_oos=np.nan, p_oos=np.nan, netto_oos=np.nan)
        rows.append(row)

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    tested = res["n_is"] >= HURDLE["min_n_is"]
    res["p_holm"] = np.nan
    res.loc[tested, "p_holm"] = holm(res.loc[tested, "p_is"].values)

    def verdict(r):
        if r["n_is"] < HURDLE["min_n_is"]:
            return "zu wenige Ereignisse"
        if not (r["p_holm"] <= HURDLE["holm_alpha_is"]):
            return "In-Sample durchgefallen"
        if r["n_oos"] < HURDLE["min_n_oos"]:
            return "OOS zu klein"
        if not (r["p_oos"] <= HURDLE["p_oos"]):
            return "OOS bestaetigt nicht"
        if HURDLE["net_oos_positive"] and not (r["netto_oos"] > 0):
            return "nach Gebuehren negativ"
        return "BESTANDEN"

    res["urteil"] = res.apply(verdict, axis=1)
    return res.sort_values(["urteil", "p_is"]).reset_index(drop=True)


def raw_reaction_table(events, close, lag=ENTRY_LAG):
    """Info: mittlere Ueberrendite je Thema/Ton fuer alle Horizonte, IS und OOS."""
    out = []
    ev = events.copy()
    ev["tone"] = np.where(ev["tone_delta"] < 0, "negativ", "positiv")
    oos_ts = pd.Timestamp(OOS_START)
    for h in HORIZONS:
        fr = forward_returns(close, h, lag)
        fr = fr[fr.index >= pd.Timestamp(ANALYSIS_START)]
        mu_is, mu_oos = fr[fr.index < oos_ts].mean(), fr[fr.index >= oos_ts].mean()
        for (cat, tone), g in ev.groupby(["category", "tone"]):
            r = fr.reindex(g["date"]).dropna()
            out.append({"h": h, "thema": cat, "ton": tone,
                        "is": r[r.index < oos_ts].mean() - mu_is,
                        "oos": r[r.index >= oos_ts].mean() - mu_oos})
    return pd.DataFrame(out)


def fmt_pct(x):
    return "   -   " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+6.2f}%"


def fmt_p(x):
    return "  -  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:5.3f}"


def run():
    gd = load_gdelt()
    if gd.empty or not os.path.exists(BTC_FILE):
        log("Keine Daten im Cache. Zuerst: python3 news_event_study.py --fetch")
        sys.exit(1)
    close = load_btc()
    close.index = pd.to_datetime(close.index)

    all_ev = []
    for cat in CATEGORIES:
        e = detect_events(gd, cat)
        if e.empty:
            log(f"  {cat}: keine Daten oder keine Ereignisse")
            continue
        all_ev.append(e)
    events = pd.concat(all_ev, ignore_index=True)
    events = events[events["date"] >= pd.Timestamp(ANALYSIS_START)]

    for h in HORIZONS:
        fr = forward_returns(close, h, ENTRY_LAG)
        events[f"r_{h}d"] = events["date"].map(fr)
    events.sort_values("date").to_csv(EVENTS_FILE, index=False, float_format="%.5f")

    lines = []
    P = lines.append
    P(f"News-Event-Studie BTC  |  Lauf {datetime.now():%Y-%m-%d %H:%M}")
    P(f"GDELT {gd.index.min():%Y-%m-%d} bis {gd.index.max():%Y-%m-%d}, "
      f"BTC bis {close.index.max():%Y-%m-%d}")
    P(f"Ereignis: z >= {Z_MIN}, mind. {MIN_ARTICLES} Artikel, Sperre {COOLDOWN_DAYS} Tage. "
      f"Einstieg Tagesschluss + {ENTRY_LAG} Tag(e). Kosten {COST_ROUNDTRIP:.2%} je Rundreise.")
    P(f"In-Sample bis {OOS_START - timedelta(days=1)}, Out-of-Sample ab {OOS_START}.")
    P(f"Ereignisse gesamt: {len(events)}  " +
      "  ".join(f"{c}={n}" for c, n in events['category'].value_counts().items()))
    P("")

    res = analyze(events, close)
    P(f"=== HUERDE: Horizont {PRIMARY_HORIZON} Tage (vorab festgelegt) ===")
    P(f"{'Thema':<19}{'Ton':<9}{'n_IS':>5}{'Ueber_IS':>10}{'p_IS':>7}{'p_Holm':>8}"
      f"{'Seite':>7}{'n_OOS':>6}{'Ueber_OOS':>10}{'p_OOS':>7}{'netto_OOS':>10}  Urteil")
    for _, r in res.iterrows():
        P(f"{r['thema']:<19}{r['ton']:<9}{r['n_is']:>5}{fmt_pct(r['excess_is']):>10}"
          f"{fmt_p(r['p_is']):>7}{fmt_p(r['p_holm']):>8}{r['seite']:>7}{r['n_oos']:>6}"
          f"{fmt_pct(r['excess_oos']):>10}{fmt_p(r['p_oos']):>7}{fmt_pct(r['netto_oos']):>10}  {r['urteil']}")
    passed = res[res["urteil"] == "BESTANDEN"]
    P("")
    if passed.empty:
        P("ERGEBNIS: Keine Zelle besteht die Huerde. Kein Paper-Bot auf dieser Grundlage.")
    else:
        P("ERGEBNIS: Bestanden: " + ", ".join(f"{r.thema}/{r.ton} -> {r.seite}"
                                              for r in passed.itertuples()))
        P("Naechster Schritt: Live-Paper-Test NUR dieser Zellen, Parameter eingefroren.")

    P("")
    P("=== Info (keine Huerde): Robustheit mit Einstieg einen Tag spaeter ===")
    res_lag = analyze(events, close, lag=1, seed=7)
    for _, r in res_lag.iterrows():
        if r["n_is"] >= HURDLE["min_n_is"]:
            P(f"  {r['thema']:<19}{r['ton']:<9} IS {fmt_pct(r['excess_is'])} (p {fmt_p(r['p_is'])})"
              f"  OOS {fmt_pct(r['excess_oos'])}  -> {r['urteil']}")

    P("")
    P("=== Info: Ueberrendite gegen Zufallstage je Horizont (IS | OOS) ===")
    raw = raw_reaction_table(events, close)
    piv = raw.pivot_table(index=["thema", "ton"], columns="h", values=["is", "oos"])
    for (cat, tone), r in piv.iterrows():
        cells = "  ".join(f"{h}T: {fmt_pct(r[('is', h)])} | {fmt_pct(r[('oos', h)])}" for h in HORIZONS)
        P(f"  {cat:<19}{tone:<9}{cells}")

    text = "\n".join(lines)
    print(text)
    with open(RESULT_FILE, "w") as f:
        f.write(text + "\n")
    log(f"\nGespeichert: {RESULT_FILE}, {EVENTS_FILE}")


# --------------------------------------------------------------------------
# Selbsttest (ohne Netz)
# --------------------------------------------------------------------------
def _synthetic(seed, effect_cat=None, effect=0.0, same_day_move=0.0, years=(2017, 2025)):
    """Kuenstliche GDELT-Daten + BTC-Kurse. Optional: eingepflanzte Reaktion NACH
    Ereignissen (effect pro Tag, 3 Tage) oder Bewegung AM Ereignistag (same_day_move)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31", freq="D")
    n = len(idx)
    gd = {}
    spikes = {}
    for cat in ("A", "B"):
        base = rng.poisson(40, n).astype(float)
        sp = np.zeros(n, bool)
        sp[rng.choice(np.arange(70, n - 10), size=n // 35, replace=False)] = True
        base[sp] *= rng.uniform(4, 8, sp.sum())
        gd[f"{cat}_count"] = base
        gd[f"{cat}_norm"] = rng.normal(100000, 3000, n)
        gd[f"{cat}_tone"] = rng.normal(-1, 1, n)
        spikes[cat] = sp
    gd = pd.DataFrame(gd, index=idx)
    ret = rng.normal(0.0005, 0.03, n)
    if effect_cat:
        for i in np.where(spikes[effect_cat])[0]:
            ret[i + 1:i + 4] += effect          # Tage NACH dem Ereignis
            ret[i] += same_day_move             # Ereignistag selbst
    close = pd.Series(20000 * np.exp(np.cumsum(ret)), index=idx)
    return gd, close


def selftest():
    global CATEGORIES
    saved = CATEGORIES
    CATEGORIES = {"A": "", "B": ""}
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'FEHLER'}] {name} {info}")

    def ev_of(gd):
        return pd.concat([detect_events(gd, c) for c in ("A", "B")], ignore_index=True)

    print("Selbsttest news_event_study.py")

    # [1] Parser mit echter GDELT-Antwort (Ausschnitt aus einer Testabfrage)
    sample = ('{"query_details": {"title": "x", "date_resolution": "day"}, "timeline": [ '
              '{ "series": "Article Count", "data": [ { "date": "20180101T000000Z", "value": 12, "norm": 51234 },'
              '{ "date": "20180102T000000Z", "value": 30, "norm": 60000 } ] } ] }')
    rows = parse_gdelt_timeline(sample)
    check("[1] GDELT-Parser", rows == [{"date": "2018-01-01", "value": 12, "norm": 51234},
                                        {"date": "2018-01-02", "value": 30, "norm": 60000}])
    try:
        parse_gdelt_timeline("Please limit requests to one every 5 seconds")
        check("[1b] Rate-Limit-Text wird als Fehler erkannt", False)
    except ValueError:
        check("[1b] Rate-Limit-Text wird als Fehler erkannt", True)

    # [2] Kein Blick in die Zukunft: Zukunft veraendern darf heutige Ereignisse nicht aendern
    gd, close = _synthetic(1)
    cut = gd.index[1500]
    gd2 = gd.copy()
    gd2.loc[gd2.index > cut, "A_count"] *= 10
    e1 = detect_events(gd, "A"); e2 = detect_events(gd2, "A")
    same = e1[e1["date"] <= cut].reset_index(drop=True).equals(e2[e2["date"] <= cut].reset_index(drop=True))
    check("[2] Ereignisse bis Tag t haengen nicht von Daten nach t ab", same)

    # [3] Sperrfrist: zwei Spitzen im Abstand von 3 Tagen ergeben ein Ereignis
    gd3 = gd.copy()
    gd3["A_count"] = 40.0
    gd3.iloc[400, gd3.columns.get_loc("A_count")] = 400
    gd3.iloc[403, gd3.columns.get_loc("A_count")] = 400
    gd3.iloc[420, gd3.columns.get_loc("A_count")] = 400
    e3 = detect_events(gd3, "A")
    check("[3] Sperrfrist 7 Tage", len(e3) == 2, f"(gefunden {len(e3)}, erwartet 2)")

    # [4] Eingepflanzte Reaktion nach Ereignissen wird gefunden, richtige Seite, nur Thema A
    gd, close = _synthetic(2, effect_cat="A", effect=0.012)
    res = analyze(ev_of(gd), close, n_perm=4000, analysis_start=date(2017, 3, 15))
    a_pass = res[(res["thema"] == "A") & (res["urteil"] == "BESTANDEN")]
    b_pass = res[(res["thema"] == "B") & (res["urteil"] == "BESTANDEN")]
    check("[4] Eingepflanzter Effekt (+1,2 %/Tag ueber 3 Tage) besteht",
          len(a_pass) >= 1 and (a_pass["seite"] == "long").all(),
          f"(A bestanden: {len(a_pass)} Zellen)")
    check("[4b] Thema ohne Effekt besteht nicht", len(b_pass) == 0)

    # [5] Zu kleiner Effekt (+0,05 %/Tag) darf nicht durchkommen (Rauschen bzw. Gebuehren)
    gd, close = _synthetic(3, effect_cat="A", effect=0.0005)
    res5 = analyze(ev_of(gd), close, n_perm=4000)
    check("[5] Winziger Effekt kommt nicht durch",
          (res5["urteil"] != "BESTANDEN").all())

    # [6] Bewegung AM Ereignistag (Nachricht kommt zu spaet) darf NICHT als Signal gelten.
    #     Gegenprobe: eine falsche Implementierung mit Einstieg am Vortag findet sie.
    gd, close = _synthetic(4, effect_cat="A", same_day_move=0.06)
    ev = ev_of(gd)
    res_ok = analyze(ev, close, n_perm=4000)
    res_wrong = analyze(ev, close, lag=-1, horizon=PRIMARY_HORIZON, n_perm=4000)
    check("[6] Bewegung am Ereignistag wird korrekt NICHT gezaehlt",
          (res_ok["urteil"] != "BESTANDEN").all())
    check("[6b] Gegenprobe: falscher Einstieg (Vortag) faellt darauf herein",
          (res_wrong[res_wrong["thema"] == "A"]["urteil"] == "BESTANDEN").any())

    # [7] Nullwelt: 10 Zufallswelten ohne Effekt - Fehlalarme zaehlen
    false_pass = 0
    for s in range(10, 20):
        gd, close = _synthetic(s)
        r = analyze(ev_of(gd), close, n_perm=2000, seed=s)
        false_pass += int((r["urteil"] == "BESTANDEN").sum())
    check("[7] Nullwelt: hoechstens 1 Fehlalarm in 40 Zellen", false_pass <= 1,
          f"(Fehlalarme: {false_pass})")

    # [8] Short-Seite: eingepflanzter Rueckgang wird als short erkannt
    gd, close = _synthetic(5, effect_cat="B", effect=-0.012)
    res8 = analyze(ev_of(gd), close, n_perm=4000)
    b8 = res8[(res8["thema"] == "B") & (res8["urteil"] == "BESTANDEN")]
    check("[8] Fallende Reaktion wird als short erkannt",
          len(b8) >= 1 and (b8["seite"] == "short").all(), f"(B bestanden: {len(b8)})")

    CATEGORIES = saved
    print("Selbsttest", "BESTANDEN" if ok else "FEHLGESCHLAGEN")
    return ok


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="BTC-Reaktion auf Nachrichten-Ereignisse messen (kein Handel)")
    ap.add_argument("--fetch", action="store_true", help="GDELT + BTC laden (setzt fort)")
    ap.add_argument("--run", action="store_true", help="Auswertung")
    ap.add_argument("--selftest", action="store_true", help="Methodenpruefung ohne Netz")
    ap.add_argument("--themen", action="store_true", help="Themen und Suchanfragen anzeigen")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.themen:
        for k, v in CATEGORIES.items():
            print(f"{k:<20}{v}")
        return
    if a.fetch:
        fetch_btc()
        fetch_gdelt()
    if a.run:
        run()
    if not (a.fetch or a.run):
        ap.print_help()


if __name__ == "__main__":
    main()
