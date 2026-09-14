"""EOD short: short the day's single biggest %-gainer ($1-10) right after the close.

Thesis (from a live trade, 2026-08-06): XHLD ran +250% intraday on a gap
that started at only +1.6% - this ticker was never even a candidate for the
long strategy. Shorted by hand at the 16:00 close, covered near +30%.
Separate mechanism from
gap_vwap_strategy.py's long thesis - different universe (today's biggest
gainer, not the open-gap scanners), different side, different time of day.

One trade a day: the single biggest gainer, no runner-up fallback.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .runner import ET, _today, read_intents
from .yf_utils import download_batch, _import_yf

log = logging.getLogger(__name__)

# ADDED 2026-08-07: the dashboard's "Open positions" table only ever reads
# live_state.json, which only the long-side intraday runner writes to -
# this module (including a REAL position tracked via track_manual_short())
# had zero live visibility, only a post-resolution row in eod_short_log.csv.
# Separate file, not merged into live_state.json, because this module runs
# as an independent process from the intraday runner and the two must never
# race writing the same file.
EOD_SHORT_STATE = DATA_DIR / "eod_short_state.json"


class Params:
    def __init__(self, take_profit=40.0, stop_loss=15.0, cover_by="18:00", cost_pct=1.0):
        self.take_profit, self.stop_loss = take_profit, stop_loss
        self.cover_by, self.cost_pct = cover_by, cost_pct


# stop_loss CHANGED 2026-09-13 (was 20.0): a post-hoc SL/TP sweep over the
# only 4 trades with archived post-market bars (INDP/SUNE/TNON/MKDW,
# 2026-09-08..11 - everything older had already aged out of Yahoo's ~8-day
# window) found every SL>=15 tied at the same result (+6.84%/trade, 4/4) and
# every SL<15 caught two of those four (SUNE, TNON) on an intraday spike that
# later reverted, turning a TIME win into a stop-out. TP was untouched by any
# level tried 20-80 - none of the 4 ever fell that far - so it stays at its
# original value on no evidence either way.
#
# UNLIKE kill_switch.py's bounds, this was NOT decided before the data
# existed - it's tuned on the same 4-trade sample it's graded against, which
# is exactly the failure mode this project's own kill-switch rationale exists
# to prevent. Kept anyway per an explicit user decision after that caveat was
# raised (data/eod_short_sl_tp_sweep.csv has the full grid). Revisit once
# enough new trades exist to check this out-of-sample; don't re-tune it again
# off a losing stretch without that.
DEFAULT = Params()


def top_gainer(tickers: list[str]) -> dict | None:
    """Today's single most-extended name vs YESTERDAY'S close, measured AT
    THE CLOSE (the actual short entry price), $1-10 at the close.

    CHANGED 2026-08-10 (was close-vs-today's-open): that metric requires the
    run to still be happening at the open, so it missed XHLD-shaped moves
    that front-load into an overnight/pre-market gap and fade before the
    open - SCKT that day gapped 0.39 -> 2.61 open, ran to a 2.76 high, then
    gave it all back to close DOWN 18% from its open, so the old metric
    scored it as a loser and passed it over entirely, even though it closed
    at 2.13 - still +446% vs yesterday's 0.39 close, extended enough to be a
    short. Keying off yesterday's close instead of today's open catches that
    shape.

    CHANGED AGAIN 2026-08-12 (was close-vs-prior-close via today's HIGH, not
    Close): the 08-10 fix ranked by how far the name ran at its INTRADAY PEAK,
    not by how extended it still was at the close. Live example the same day:
    OFAL spiked to +419% off its prior close, then gave almost all of it back
    to close at only +90.7% - the pick that day over BOXL, which ran a
    smaller peak (+236%) but held it much better and closed at +168.6%. By
    the time you're actually selling the entry (16:00), OFAL had already
    round-tripped most of its move and BOXL was the one still extended.
    Ranking off the CLOSE - same field the entry price and the $1-10 filter
    already use - picks the name that's actually still stretched when the
    trade opens, not one whose real move happened hours earlier. SCKT's
    close-vs-prior-close (+446%) is comfortably its own high-water mark
    among that day's names too, so this doesn't reopen the 08-10 bug.

    $1-10 filter on the close ADDED 2026-08-10: the old code had none despite
    this module's docstring claiming one, and a %-based ranking is exposed to
    a sub-penny name's swings dominating it (live example: MGN, close
    $0.127, out-ranked everything on 2026-08-10 under the pre-fix metric).

    Run once after the close - mirrors runner.candidates(), but keyed off
    the day's close-to-close move instead of the open gap."""
    rows = []
    for i in range(0, len(tickers), 200):
        frames = download_batch(tickers[i : i + 200], period="2d", interval="1d")
        for t, d in frames.items():
            d = d.dropna(subset=["Open", "Close"])
            if len(d) < 2:
                continue
            today, prior = d.iloc[-1], d.iloc[-2]
            if prior.Close <= 0:
                continue
            rows.append({
                "ticker": t,
                "close": float(today.Close),
                "pct_gain": (today.Close - prior.Close) / prior.Close * 100,
            })
        time.sleep(1.0)
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df = df[df.close.between(1.0, 10.0)]
    if df.empty:
        return None
    best = df.sort_values("pct_gain", ascending=False).iloc[0]
    return {"ticker": best.ticker, "close": float(best.close), "pct_gain": float(best.pct_gain)}


def fetch_post_market(ticker: str, date: dt.date | None = None) -> pd.DataFrame:
    """This ticker's post-16:00 bars for `date` (defaults to the most recent
    date in the fetched window - i.e. "today" for the live caller).

    `date=None` was the only mode this had until 2026-08-07: it always
    resolved to `df.index.max().date()`, which is correct for the live path
    (there is no other "today") but silently breaks anything that pins a
    PAST date - including this module's own self-check, which reproduces
    the 2026-08-06 XHLD trade and started returning NO_DATA the moment it
    was run on a later day, because "today" had moved on to a session with
    no post-close bars yet. `period` widens automatically for an explicit
    past date so it's still in the fetched window."""
    yf = _import_yf()
    period = "2d" if date is None else f"{max(2, (dt.date.today() - date).days + 2)}d"
    df = yf.Ticker(ticker).history(period=period, interval="1m", prepost=True)
    if df.empty:
        return df
    df = df.tz_convert("America/New_York") if df.index.tz else df.tz_localize("America/New_York")
    target = date if date is not None else df.index.max().date()
    return df[df.index.date == target].between_time("16:00", "20:00")


def scan_short(post_market: pd.DataFrame, entry_price: float, p: Params = DEFAULT) -> dict | None:
    """First bar (if any) that hits the stop or the target. Stop checked
    first within a bar - same worst-case assumption as gap_vwap_strategy's
    simulate_exit. None means unresolved in the bars given so far (still
    open, or ran out of cover_by without data - caller decides what that means)."""
    target = entry_price * (1 - p.take_profit / 100)
    stop = entry_price * (1 + p.stop_loss / 100)
    for t, row in post_market.iterrows():
        if row.High >= stop:
            return _result("SL", stop, t, entry_price, p)
        if row.Low <= target:
            return _result("TP", target, t, entry_price, p)
    return None


def simulate_short(post_market: pd.DataFrame, entry_price: float, p: Params = DEFAULT) -> dict:
    """Backtest entry point: walk the post-market bars up to cover_by. Cover
    at target/stop, or flat at cover_by / end of data if neither fires."""
    bars = post_market[[t.strftime("%H:%M") <= p.cover_by for t in post_market.index]]
    hit = scan_short(bars, entry_price, p)
    if hit:
        return hit
    if bars.empty:
        return _result("NO_DATA", entry_price, None, entry_price, p)
    return _result("TIME", bars.Close.iloc[-1], bars.index[-1], entry_price, p)


def _result(outcome, exit_price, exit_time, entry_price, p: Params) -> dict:
    ret = (entry_price - exit_price) / entry_price * 100 - p.cost_pct   # short: profit on the way down
    return {"outcome": outcome, "exit_price": float(exit_price),
            "exit_time": exit_time.strftime("%H:%M") if exit_time is not None else None,
            "return_pct": ret}


def _log(row: dict) -> None:
    path = DATA_DIR / "eod_short_log.csv"
    # "source" keeps its ORIGINAL meaning (live poll vs backtest/replay
    # reconstruction - dashboard.py's "live fills only" toggle and the kill
    # switch both filter on source=="live"). Every row this module writes is
    # "live" by that definition, whether the bot picked it or a real
    # position was tracked by hand - there is no backtest/replay path here.
    # "is_real" / "position_size" ADDED 2026-08-07 for the actual real-money
    # distinction (was wrongly conflated into "source" at first, which
    # silently dropped every eod_short row - real or paper - out of the
    # "live fills only" view since none of them said the literal string
    # "live" anymore).
    cols = ["ticker", "date", "pct_gain", "entry_price", "outcome",
            "exit_time", "exit_price", "return_pct", "source",
            "is_real", "position_size"]
    entry = pd.DataFrame([{k: row.get(k) for k in cols}])
    entry.to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_state(ticker: str, entry: float, p: Params, is_real: bool,
                 started_at: str, last: float | None, resolved: bool,
                 result: dict | None = None, position_size: float | None = None) -> None:
    target = entry * (1 - p.take_profit / 100)
    stop = entry * (1 + p.stop_loss / 100)
    unrealised_pct = None if last is None else (entry - last) / entry * 100
    payload = {
        "ticker": ticker, "side": "short", "is_real": is_real,
        "entry_price": entry, "target": target, "stop": stop,
        "cover_by": p.cover_by, "started_at": started_at,
        "updated": pd.Timestamp.now(tz=ET).strftime("%H:%M:%S"),
        "last": last,
        # gross of cost: a mark, not a realised trade - short profits as price falls
        "unrealised_pct": unrealised_pct,
        "position_size": position_size,
        "unrealised_usd": (None if unrealised_pct is None or position_size is None
                           else unrealised_pct / 100 * position_size),
        "resolved": resolved,
        "result": result,
    }
    try:
        EOD_SHORT_STATE.write_text(json.dumps(payload, default=str))
    except OSError:
        log.exception("eod-short state write failed - tracking continues")


INTENT_PREFIX = "SHORT:"     # namespaced so a short's "close" can't collide with
                              # a same-day long position in the same ticker


def _poll_until_resolved(ticker: str, entry: float, p: Params, poll_interval: int,
                         is_real: bool, position_size: float | None = None) -> dict:
    """Shared by run_eod_short() and track_manual_short(): poll the
    post-market tape until target/stop fires, cover_by arrives, or the
    dashboard's Close button is pressed (mirrors the long side's close/ride
    intent - same intents.json, namespaced by INTENT_PREFIX so it can't
    collide with a long position in the same ticker). Writes
    eod_short_state.json every poll so the dashboard can show it live."""
    started_at = pd.Timestamp.now(tz=ET).strftime("%H:%M:%S")
    result = None
    while pd.Timestamp.now(tz=ET).strftime("%H:%M") < p.cover_by:
        post = fetch_post_market(ticker)
        last = float(post.Close.iloc[-1]) if not post.empty else None
        _write_state(ticker, entry, p, is_real, started_at, last, resolved=False,
                    position_size=position_size)
        if last is not None and read_intents().get(INTENT_PREFIX + ticker, {}).get("action") == "close":
            now = pd.Timestamp.now(tz=ET)
            result = _result("MANUAL", last, now, entry, p)
            break
        if not post.empty:
            result = scan_short(post, entry, p)
            if result:
                break
        time.sleep(poll_interval)
    if result is None:
        post = fetch_post_market(ticker)
        result = (_result("TIME", post.Close.iloc[-1], post.index[-1], entry, p)
                  if not post.empty else _result("NO_DATA", entry, None, entry, p))
    _write_state(ticker, entry, p, is_real, started_at, result.get("exit_price"),
                resolved=True, result=result, position_size=position_size)
    return result


def run_eod_short(universe_tickers: list[str], p: Params = DEFAULT,
                  poll_interval: int = 60) -> dict | None:
    """DRY RUN - after the close, short today's single biggest gainer, poll
    the post-market tape until target/stop/cover_by. Logs to
    data/eod_short_log.csv (source='live', is_real=False). No order is ever
    submitted."""
    best = top_gainer(universe_tickers)
    if best is None:
        log.info("eod-short: no gainer found today")
        return None
    ticker, entry = best["ticker"], best["close"]
    log.info("EOD SHORT %s @ %.4f (+%.0f%% today, TP%.0f/SL%.0f)  [DRY RUN, no order sent]",
             ticker, entry, best["pct_gain"], p.take_profit, p.stop_loss)

    result = _poll_until_resolved(ticker, entry, p, poll_interval, is_real=False)
    row = {"ticker": ticker, "date": _today(), "pct_gain": best["pct_gain"],
           "entry_price": entry, "source": "live", "is_real": False,
           "position_size": None, **result}
    log.info("EOD SHORT %s closed %s @ %.4f -> %+.2f%%",
             ticker, result["outcome"], result["exit_price"], result["return_pct"])
    _log(row)
    return row


def track_manual_short(ticker: str, entry_price: float, p: Params = DEFAULT,
                       poll_interval: int = 60, position_size: float | None = None) -> dict:
    """Track a REAL, manually-placed short - NOT a dry run.

    This module has never submitted a real order (see the module docstring)
    - everything else it has ever logged is paper. Use this when a short was
    actually opened by hand outside the system (e.g. because the automated
    scan failed or picked something untradeable that day). Polls exactly
    like run_eod_short(), but skips the biggest-gainer scan - the ticker and
    entry are already fixed by the real fill - and logs with is_real=True so
    it can never be silently blended with this module's own paper picks in
    any aggregate stat (source stays 'live' - see _log()'s comment - this
    IS a live poll, just also a real one).

    `position_size`: actual dollars risked, if known. Feeds the real $ P&L
    shown live and folded into the equity curve once this resolves - without
    it, no dollar figure is shown (assuming the paper POSITION_SIZE for a
    real trade of unknown size would be a guess dressed up as a fact).
    """
    log.info("TRACKING REAL SHORT %s @ %.4f (TP%.0f/SL%.0f, cover by %s, size %s)  "
             "[REAL POSITION - opened manually, NOT by this system]",
             ticker, entry_price, p.take_profit, p.stop_loss, p.cover_by,
             f"${position_size:,.0f}" if position_size else "unknown")
    result = _poll_until_resolved(ticker, entry_price, p, poll_interval,
                                  is_real=True, position_size=position_size)
    row = {"ticker": ticker, "date": _today(), "pct_gain": None,
           "entry_price": entry_price, "source": "live", "is_real": True,
           "position_size": position_size, **result}
    log.info("REAL SHORT %s closed %s @ %.4f -> %+.2f%%",
             ticker, result["outcome"], result["exit_price"], result["return_pct"])
    _log(row)
    return row


def _self_check() -> None:
    global fetch_post_market, read_intents, EOD_SHORT_STATE
    # reproduce the actual XHLD trade that prompted this module - pinned
    # params AND a pinned date, independent of whatever DEFAULT currently is
    # or what day this happens to run on (see fetch_post_market's docstring -
    # this broke silently for a day once the calendar moved past 2026-08-06)
    post = fetch_post_market("XHLD", date=dt.date(2026, 8, 6))
    entry = 2.81   # 2026-08-06 16:00 close
    ex = simulate_short(post, entry, Params(take_profit=30.0, stop_loss=10.0))
    assert ex["outcome"] == "TP", f"expected the 30% target to hit, got {ex}"
    assert ex["return_pct"] > 25, f"expected roughly +30% (less cost), got {ex}"
    # a hypothetical stop-out: entry near the day's low so a tight stop is easy to hit
    ex2 = simulate_short(post, entry * 0.7, Params(stop_loss=1.0))
    assert ex2["outcome"] == "SL", f"expected a tight stop to fire first, got {ex2}"

    # dashboard Close button: _poll_until_resolved must resolve MANUAL, at
    # the last-seen price, the FIRST poll a SHORT:<ticker> close intent is
    # present - not wait for cover_by. Monkeypatch the module's own
    # fetch_post_market/read_intents AND EOD_SHORT_STATE (the latter learned
    # the hard way: _write_state() writes there on every poll, so leaving it
    # pointed at the real path clobbered a genuine same-day BOXL result with
    # this test's "ZZZZ" the first time this check ran).
    import tempfile
    real_fpm, real_ri, real_state_path = fetch_post_market, read_intents, EOD_SHORT_STATE
    try:
        stub_bars = pd.DataFrame(
            {"Open": [5.0], "High": [5.0], "Low": [5.0], "Close": [5.0]},
            index=[pd.Timestamp("2026-08-12 16:01", tz=ET)])
        fetch_post_market = lambda ticker, date=None: stub_bars   # noqa: E731
        read_intents = lambda: {"SHORT:ZZZZ": {"action": "close"}}  # noqa: E731
        with tempfile.TemporaryDirectory() as tmp:
            EOD_SHORT_STATE = Path(tmp) / "eod_short_state.json"
            ex3 = _poll_until_resolved("ZZZZ", entry=10.0, p=Params(cover_by="23:59"),
                                       poll_interval=0, is_real=False)
        assert ex3["outcome"] == "MANUAL", f"expected an immediate manual close, got {ex3}"
        assert ex3["exit_price"] == 5.0, f"expected the stubbed last price, got {ex3}"
    finally:
        fetch_post_market, read_intents, EOD_SHORT_STATE = real_fpm, real_ri, real_state_path

    print("self-check passed: 3/3")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_check()
