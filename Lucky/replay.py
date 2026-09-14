"""Offline reconstruction of a missed live session (or a range of them).

`Lucky.runner` is real-time only: it sleeps to 09:31 ET and trades the
session as bars arrive. If it did not run on a given day, that day never
enters the forward record. This module reconstructs it afterwards from
now-settled bars, using the SAME functions the live path calls -
`runner.candidates` / `runner.verify_gaps` / `runner.baselines_for` and
`S.backtest` for the long side, `eod_short.top_gainer` /
`eod_short.simulate_short` for the EOD short - and appends the results to
the same files a live run writes: `data/trade_log.csv`,
`data/eod_short_log.csv`, and a per-day `data/candidates_YYYYMMDD.csv`.

This was done by hand for the 2026-08-10 and 2026-08-12 missed sessions
with scripts that were not kept. This module is that procedure, kept.

    python -m Lucky.replay 2026-08-17 2026-08-27   # inclusive range, weekdays only
    python -m Lucky.replay 2026-08-25              # a single day
    python -m Lucky.replay 2026-08-17 2026-08-27 --dry-run   # print, write nothing
    python -m Lucky.replay --check                 # self-check (no network)

LIMITATIONS
  - Settled bars carry none of the poll-by-poll revision history the live
    runner sees, so a reconstructed fill can differ from what a live fill
    would have been. These rows read as `live` on the
    dashboard - same as the 08-10/08-12 reconstructions already in the log.
  - yfinance keeps extended-hours 1m data only ~8 trading days, so the EOD
    short cannot be reconstructed past that window - it is skipped, with a
    note, rather than logging a fabricated row.
  - No holiday calendar (same gap as `runner._next_open`): weekends are
    skipped automatically, a market holiday in the range must be skipped by
    hand (pass explicit single dates around it).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
import warnings
from functools import lru_cache

import pandas as pd

warnings.filterwarnings("ignore")

import gap_vwap_strategy as S

from . import eod_short
from .config import DATA_DIR
from .kill_switch import evaluate as kill_evaluate
from .runner import _append_trade_log, verify_gaps
from .yf_utils import _import_yf, download_batch, to_exchange_time

log = logging.getLogger(__name__)
P = S.DEFAULT

# mirrors runner.candidates(): a prior close outside this band cannot open in
# $1-3 on an 8-20% gap, so the universe is cut here before any request.
_PRIOR_CLOSE_BAND = (0.75, 3.10)


# --------------------------------------------------------------- data fetch --


def _daily_history(tickers: list[str], start: dt.date, end: dt.date) -> dict[str, pd.DataFrame]:
    """One batched daily-OHLCV pull for the whole run, chunked like
    runner.candidates(). `end` is exclusive-ish (yfinance convention); pass
    the day after the last session you need. Raw prices (auto_adjust=False,
    same as runner) - splits are handled separately, per-ticker, in _splits()."""
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 200):
        chunk = tickers[i : i + 200]
        out.update(download_batch(chunk, start=str(start), end=str(end), interval="1d"))
        time.sleep(1.0)
    return out


def _session_rows(df: pd.DataFrame, day: dt.date) -> tuple[pd.Series, pd.Series] | None:
    """(today_row, prior_row) for `day`, or None if this ticker had no session
    that day or no prior session to gap from. Each row's `.name` is its date."""
    if df is None or df.empty:
        return None
    d = df.dropna(subset=["Open", "Close"])
    idx = [t for t in d.index if t.date() <= day]
    if len(idx) < 2 or idx[-1].date() != day:
        return None
    prior = d.loc[idx[-2]]
    if prior.Close <= 0:
        return None
    return d.loc[idx[-1]], prior


@lru_cache(maxsize=8192)
def _splits(ticker: str) -> tuple[dt.date, ...]:
    """Ex-dates of every stock split yfinance knows for this ticker. Fetched
    per-ticker, not from the batch daily pull: `yf.download(actions=True)`
    silently drops the split's own NaN-OHLC row in `_clean`, taking the split
    flag with it, so the batch frame shows `Stock Splits == 0` even across a
    real split (GRML 2026-08-25, a 1:50 reverse split)."""
    try:
        s = _import_yf().Ticker(ticker).splits
    except Exception:
        return ()
    return tuple(pd.Timestamp(d).date() for d in s.index)


def _split_between(ticker: str, after: dt.date, upto: dt.date) -> bool:
    """Did `ticker` split in (after, upto]? A raw prior close from before a
    reverse split manufactures a fake +thousands-% move; the name is not a
    clean candidate or gainer for that day either way."""
    return any(after < d <= upto for d in _splits(ticker))


def candidates_on(day: dt.date, daily: dict[str, pd.DataFrame], max_names: int = 40) -> pd.DataFrame:
    """Gap-ups for `day`, from a pre-fetched daily-bars dict.

    Date-parameterised copy of runner.candidates() - same filters, same
    ordering; only the data source differs (a dict slice instead of a
    period="5d" pull, which always means "latest").
    """
    rows = []
    for t, df in daily.items():
        pair = _session_rows(df, day)
        if pair is None:
            continue
        today, prior = pair
        d = df.dropna(subset=["Open", "Close"])
        prior_vol = d.loc[[x for x in d.index if x.date() < day], "Volume"]
        rows.append({
            "ticker": t, "open_price": float(today.Open), "prior_close": float(prior.Close),
            "prior_date": prior.name.date(),
            "avg_volume": float(prior_vol.tail(4).mean()) if len(prior_vol) else 0.0,
            "gap_pct": (today.Open - prior.Close) / prior.Close * 100,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    q = df[df.open_price.between(P.price_min, P.price_max)
           & df.gap_pct.between(P.gap_min, P.gap_max, inclusive="left")
           & (df.avg_volume >= 25_000)]
    # a plain list (rather than a Series) is what q[...] wants for boolean
    # row-selection, but pandas can't tell an EMPTY list of row-bools from an
    # empty list of column names and picks the latter - on a day with zero
    # candidates surviving the filters above, that silently drops every
    # column (prior_date included) instead of every row, and the .drop()
    # below then KeyErrors on a column that was there a line ago. Wrapping in
    # a Series (with q's index, and an explicit bool dtype so a zero-length
    # one isn't inferred as float64/object) keeps it row-selection at any length.
    not_split = pd.Series([not _split_between(r.ticker, r.prior_date, day) for r in q.itertuples()],
                          index=q.index, dtype=bool)
    q = q[not_split]
    return (q.drop(columns=["prior_date"]).sort_values("gap_pct", ascending=False)
            .head(max_names).reset_index(drop=True))


def baselines_on(tickers: list[str], day: dt.date) -> tuple[dict, dict]:
    """(baseline-by-ticker for `day`, 1m-bars-by-ticker incl. `day`'s session).

    Date-parameterised copy of runner.baselines_for(): 20-day 1m window
    ending the day after `day`, walked in 7-day steps (yfinance serves at
    most 8 days of 1m per request), regular hours only.
    """
    end = day + dt.timedelta(days=1)
    start = end - dt.timedelta(days=20)
    bars: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 40):
        chunk = tickers[i : i + 40]
        cur, parts = start, {}
        while cur < end:
            stop = min(cur + dt.timedelta(days=7), end)
            for t, d in download_batch(chunk, start=str(cur), end=str(stop), interval="1m").items():
                if not d.empty:
                    parts.setdefault(t, []).append(d)
            cur = stop
            time.sleep(1.0)
        for t, frames in parts.items():
            d = pd.concat(frames)
            d = d[~d.index.duplicated()]
            bars[t] = to_exchange_time(d).between_time("09:30", "16:00", inclusive="left")
    base = S.candle_baselines(bars, P)
    return {t: v for (t, d), v in base.items() if d == day}, bars


# ----------------------------------------------------------------- EOD short --


def top_gainer_on(day: dt.date, daily: dict[str, pd.DataFrame]) -> dict | None:
    """Day's single most-extended name vs prior close, measured at the close,
    close $1-10. Date-parameterised copy of eod_short.top_gainer()."""
    rows = []
    for t, df in daily.items():
        pair = _session_rows(df, day)
        if pair is None:
            continue
        today, prior = pair
        rows.append({"ticker": t, "close": float(today.Close), "prior_date": prior.name.date(),
                     "pct_gain": (today.Close - prior.Close) / prior.Close * 100})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df = df[df.close.between(1.0, 10.0)].sort_values("pct_gain", ascending=False)
    for r in df.itertuples():
        if _split_between(r.ticker, r.prior_date, day):
            log.info("%s: EOD short - skipping %s (%.0f%%), split in window", day, r.ticker, r.pct_gain)
            continue
        return {"ticker": r.ticker, "close": float(r.close), "pct_gain": float(r.pct_gain)}
    return None


# ---------------------------------------------------------------- per-day run --


def _already_done(day: dt.date) -> str | None:
    """Reason this day looks already-processed, or None. The runner's sec-27
    double-write safeguard, applied to this path: a rerun must not stack a
    second set of rows on top of the first."""
    stamp = day.strftime("%Y%m%d")
    if (DATA_DIR / f"candidates_{stamp}.csv").exists():
        return f"data/candidates_{stamp}.csv already exists"
    for name, col_date in [("trade_log.csv", "date"), ("eod_short_log.csv", "date")]:
        path = DATA_DIR / name
        if not path.exists():
            continue
        try:
            d = pd.read_csv(path, usecols=[col_date])
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if (d[col_date].astype(str) == str(day)).any():
            return f"data/{name} already has rows dated {day}"
    return None


def _write_candidates(day: dt.date, cands: pd.DataFrame, base: dict,
                      bars: dict, filled: set[str]) -> None:
    """Reproduce runner._finish()'s candidates_YYYYMMDD.csv: every watched
    ticker, whether it would have signalled (uncapped, like the live
    all_signals loop), and whether it got a fill slot."""
    rows = []
    for r in cands.itertuples():
        sig = None
        df = bars.get(r.ticker)
        bl = base.get(r.ticker)
        if df is not None and bl:
            session = df[[t.date() == day for t in df.index]]
            if len(session) >= 2:
                sig = S.find_entry(session, bl, P)
        rows.append({
            "date": day, "ticker": r.ticker, "gap_pct": r.gap_pct,
            "open_price": r.open_price, "avg_volume": r.avg_volume,
            "gap_rank": r.Index + 1, "signalled": sig is not None,
            "signal_time": sig["signal_time"] if sig else None,
            "signal_multiple": sig["signal_multiple"] if sig else None,
            "entry_time": sig["entry_time"] if sig else None,
            "filled": r.ticker in filled,
        })
    pd.DataFrame(rows).to_csv(DATA_DIR / f"candidates_{day.strftime('%Y%m%d')}.csv", index=False)


def replay_day(day: dt.date, daily: dict[str, pd.DataFrame],
               nasdaq: set[str], write: bool = True,
               skip_eod_short: bool = False, force: bool = False) -> dict:
    """Reconstruct one session. Returns a summary dict."""
    out: dict = {"day": str(day), "candidates": 0, "trades": [],
                 "long_avg": None, "eod_short": None, "skipped": None}

    if not force and (reason := _already_done(day)):
        out["skipped"] = reason
        log.warning("%s: SKIP - %s (pass --force to override)", day, reason)
        return out

    cands = candidates_on(day, daily)
    out["candidates"] = len(cands)
    if cands.empty:
        log.info("%s: no qualifying gap-ups", day)
    else:
        log.info("%s: %d candidates: %s", day, len(cands),
                 ", ".join(f"{r.ticker}({r.gap_pct:.0f}%)" for r in cands.itertuples()))

    trades = pd.DataFrame()
    base: dict = {}
    bars: dict = {}
    if not cands.empty:
        base, bars = baselines_on(list(cands.ticker), day)
        keep = set(verify_gaps(cands, bars, day).ticker)
        dropped = sorted(set(cands.ticker) - keep)
        if dropped:
            log.info("%s: real-open check dropped %s", day, ", ".join(dropped))
        ev = pd.DataFrame([{"ticker": r.ticker, "date": pd.Timestamp(day),
                            "gap_pct": r.gap_pct, "open_price": r.open_price}
                           for r in cands.itertuples() if r.ticker in keep])
        if not ev.empty:
            bl = {(t, day): v for t, v in base.items()}
            trades = S.backtest(ev, bars, bl, P)

    if len(trades):
        out["trades"] = trades[["ticker", "signal_time", "entry_time", "fill",
                                "outcome", "exit_time", "exit_price", "return_pct"]].to_dict("records")
        out["long_avg"] = float(trades.return_pct.mean())
        for r in trades.itertuples():
            log.info("%s: LONG %-6s %s @ %.4f -> %s @ %.4f  %+.2f%%",
                     day, r.ticker, r.entry_time, r.fill, r.outcome, r.exit_price, r.return_pct)

    # EOD short
    if not skip_eod_short:
        best = top_gainer_on(day, {t: daily[t] for t in nasdaq if t in daily})
        if best is None:
            log.info("%s: EOD short - no gainer found", day)
        else:
            post = eod_short.fetch_post_market(best["ticker"], date=day)
            res = eod_short.simulate_short(post, best["close"], eod_short.DEFAULT)
            if res["outcome"] == "NO_DATA":
                out["eod_short"] = {"ticker": best["ticker"], "outcome": "NO_DATA",
                                   "note": "post-market 1m data unavailable (aged out of yfinance window)"}
                log.warning("%s: EOD short %s - post-market data unavailable, not logged",
                            day, best["ticker"])
            else:
                out["eod_short"] = {"ticker": best["ticker"], "pct_gain": best["pct_gain"],
                                   "entry": best["close"], **res}
                log.info("%s: EOD SHORT %-6s @ %.4f (+%.0f%%) -> %s @ %.4f  %+.2f%%",
                         day, best["ticker"], best["close"], best["pct_gain"],
                         res["outcome"], res["exit_price"], res["return_pct"])

    if write:
        if len(trades):
            _append_trade_log(trades, day)
        if not cands.empty:
            _write_candidates(day, cands, base, bars, set(trades.ticker) if len(trades) else set())
        es = out["eod_short"]
        if es and es.get("outcome") not in (None, "NO_DATA"):
            eod_short._log({"ticker": es["ticker"], "date": day, "pct_gain": es["pct_gain"],
                            "entry_price": es["entry"], "source": "live", "is_real": False,
                            "position_size": None, "outcome": es["outcome"],
                            "exit_time": es["exit_time"], "exit_price": es["exit_price"],
                            "return_pct": es["return_pct"]})

    return out


# ------------------------------------------------------------------- driver --


def _weekdays(start: dt.date, end: dt.date) -> list[dt.date]:
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += dt.timedelta(days=1)
    return days


def _universe() -> tuple[list[str], set[str]]:
    """(prior-close-prefiltered tickers for the long scan, NASDAQ set for the
    EOD short) - both from data/universe.csv, matching runner."""
    u = pd.read_csv(DATA_DIR / "universe.csv")
    sym = "ticker" if "ticker" in u else "symbol"
    lo, hi = _PRIOR_CLOSE_BAND
    pre = u[pd.to_numeric(u["last_close"], errors="coerce").between(lo, hi)]
    nasdaq = set(u.loc[u["exchange"].eq("NASDAQ"), sym]) if "exchange" in u else set(u[sym])
    return sorted(pre[sym].unique()), nasdaq


def replay_range(start: dt.date, end: dt.date, write: bool = True,
                 skip_eod_short: bool = False, force: bool = False) -> list[dict]:
    days = _weekdays(start, end)
    if not days:
        log.error("no weekdays between %s and %s", start, end)
        return []
    long_tk, nasdaq = _universe()
    scan_tk = sorted(set(long_tk) | nasdaq)
    log.info("replaying %d session(s) %s .. %s; daily scan over %d tickers",
             len(days), days[0], days[-1], len(scan_tk))
    daily = _daily_history(scan_tk, start - dt.timedelta(days=12), end + dt.timedelta(days=1))
    log.info("daily history: %d/%d tickers returned", len(daily), len(scan_tk))
    return [replay_day(d, daily, nasdaq, write, skip_eod_short, force) for d in days]


def _print_summary(results: list[dict]) -> None:
    long_ret, short_ret = [], []
    print(f"\n{'=' * 72}\n  REPLAY SUMMARY\n{'=' * 72}")
    for r in results:
        if r["skipped"]:
            print(f"  {r['day']}  SKIPPED - {r['skipped']}")
            continue
        line = f"  {r['day']}  {r['candidates']} cand"
        if r["trades"]:
            line += f", {len(r['trades'])} long ({r['long_avg']:+.2f}%/trade)"
            long_ret += [t["return_pct"] for t in r["trades"]]
        else:
            line += ", 0 long"
        es = r["eod_short"]
        if es and es.get("outcome") == "NO_DATA":
            line += f"  |  short {es['ticker']}: NO DATA"
        elif es:
            line += f"  |  short {es['ticker']} {es['outcome']} {es['return_pct']:+.2f}%"
            short_ret.append(es["return_pct"])
        else:
            line += "  |  no short"
        print(line)
    if long_ret:
        print(f"\n  long : {len(long_ret)} trades, {sum(long_ret) / len(long_ret):+.2f}%/trade")
    if short_ret:
        print(f"  short: {len(short_ret)} trades, {sum(short_ret) / len(short_ret):+.2f}%/trade")

    # kill-switch verdict on the same sequence the dashboard feeds it: live
    # long fills only, week_trades.csv curation applied (see dashboard.build_state).
    try:
        from .dashboard import load_trades
        t = load_trades()
        real_long = t[(t.source == "live") & (t.side == "long")].sort_values(["date", "entry_time"])
        v = kill_evaluate(list(real_long.return_pct))
        extra = "" if v["avg"] is None else f", avg {v['avg']:+.2f}%/trade (lines {v['kill_line']:+.2f} / {v['confirm_line']:+.2f})"
        print(f"\n  kill switch (live long fills): {v['verdict']} on {v['n']} trades{extra}")
    except Exception as exc:  # a summary line must never be the thing that fails a run
        print(f"\n  kill switch: could not evaluate ({exc})")


def _self_check() -> None:
    global _splits
    ts = lambda d: pd.Timestamp(f"2026-06-{d:02d} 00:00", tz="America/New_York")

    def daily(spec: dict[str, list[tuple]]) -> dict[str, pd.DataFrame]:
        # spec: ticker -> [(open, close, volume), ...] for consecutive days 1..N
        out = {}
        for t, days in spec.items():
            idx = [ts(i + 1) for i in range(len(days))]
            out[t] = pd.DataFrame(
                {"Open": [r[0] for r in days], "High": [max(r[0], r[1]) for r in days],
                 "Low": [min(r[0], r[1]) for r in days], "Close": [r[1] for r in days],
                 "Volume": [r[2] for r in days]}, index=idx)
        return out

    day = ts(4).date()  # the 4th of the synthetic month is "today"
    d = daily({
        # +15% gap, opens $2.30, deep prior volume -> the one real candidate
        "GAP":   [(2.0, 2.0, 2e5), (2.0, 2.0, 2e5), (2.0, 2.0, 2e5), (2.30, 2.40, 3e5)],
        # +20% gap but opens below $1 -> filtered on price
        "SMALL": [(0.5, 0.5, 2e5), (0.5, 0.5, 2e5), (0.5, 0.5, 2e5), (0.60, 0.61, 3e5)],
        # +15% gap but averages 1k shares/day -> filtered on the 25k liquidity floor
        "THIN":  [(2.0, 2.0, 1e3), (2.0, 2.0, 1e3), (2.0, 2.0, 1e3), (2.30, 2.31, 1e3)],
        # +25% gap -> outside the 8-20 band (high end is exclusive)
        "BIG":   [(2.0, 2.0, 2e5), (2.0, 2.0, 2e5), (2.0, 2.0, 2e5), (2.50, 2.55, 3e5)],
    })
    real_splits = _splits
    _splits = lambda tkr: (ts(2).date(),) if tkr == "SPLITS" else ()
    try:
        assert list(candidates_on(day, d).ticker) == ["GAP"], "expected only GAP"

        # top_gainer_on: biggest close-vs-prior-close move that closes in $1-10.
        # SPLITS: raw $0.20 -> $10 across a 1:50 reverse split (ex-date the 2nd)
        # manufactures a fake +4900% and lands in band - must be dropped, not
        # picked. The GRML 2026-08-25 case.
        g = daily({
            "RUN":   [(1.0, 1.0, 1e5), (1.0, 3.0, 1e5)],       # +200%, closes $3 -> pick
            "PENNY": [(0.01, 0.01, 1e5), (0.01, 0.06, 1e5)],   # +500% but $0.06 close -> out
            "HIGH":  [(50.0, 50.0, 1e5), (50.0, 130.0, 1e5)],  # +160% but $130 close -> out
            "SPLITS": [(0.20, 0.20, 1e5), (10.0, 10.0, 1e5)],  # +4900% raw, split -> drop
        })
        best = top_gainer_on(ts(2).date(), g)
        assert best and best["ticker"] == "RUN", f"expected RUN (SPLITS dropped), got {best}"

        # a day with no session for anyone -> no candidates, no gainer, no crash
        assert candidates_on(ts(20).date(), d).empty
        assert top_gainer_on(ts(20).date(), g) is None
    finally:
        _splits = real_splits

    # the already-done guard: no false positive for a fresh future date
    assert _already_done(dt.date(2099, 1, 1)) is None

    print("replay self-check passed: 6/6")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dates", nargs="*", help="one date (single day) or two dates (inclusive range), YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true", help="print the reconstruction, write nothing")
    ap.add_argument("--force", action="store_true", help="reconstruct even if the day already has rows/files")
    ap.add_argument("--skip-eod-short", action="store_true", help="long side only")
    ap.add_argument("--check", action="store_true", help="run the self-check and exit")
    args = ap.parse_args()

    if args.check:
        _self_check()
        return
    if not args.dates or len(args.dates) > 2:
        ap.error("pass one date (a single day) or two dates (an inclusive range)")
    start = dt.date.fromisoformat(args.dates[0])
    end = dt.date.fromisoformat(args.dates[-1])
    if end < start:
        ap.error(f"end {end} is before start {start}")

    results = replay_range(start, end, write=not args.dry_run,
                           skip_eod_short=args.skip_eod_short, force=args.force)
    _print_summary(results)
    if args.dry_run:
        print("\n  (--dry-run: nothing was written)")


if __name__ == "__main__":
    main()
