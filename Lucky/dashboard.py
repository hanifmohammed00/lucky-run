"""Local dashboard: see what the runner is doing, and act on it.

    .venv/bin/python -m Lucky.dashboard        # http://localhost:8787

Read-only over the files the runner already writes (`live_state.json` while a
session is in flight, `trade_log.csv` / `eod_short_log.csv` once trades close).
The only thing it writes is `intents.json` - the close/ride buttons.

stdlib http.server on purpose: this serves one page to one person on
localhost. A web framework would be a dependency, a version to keep current,
and a config file, to do what 80 lines already do.

DRY RUN. "Close" marks the simulated position closed at that minute and the
end-of-day replay honours it. No broker is connected, so nothing here can
move real money.
The dashboard's own "Backfill missed sessions" button (see start_backfill())
is the exception to read-only: it kicks off Lucky.replay in a background
thread to fill in weekdays the live runner never ran. It never touches a day
that already has data - replay.py's own _already_done() guard covers that -
so the historical record can only grow, never be rewritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .eod_short import INTENT_PREFIX
from .kill_switch import KillParams, evaluate as kill_evaluate

log = logging.getLogger(__name__)

# The paper account the P&L is expressed against. P&L is (return % x
# POSITION_SIZE). A SIMULATION - no broker is wired up and nothing here places
# an order.
#
# $1,300 flat is not arbitrary: max_trades_per_day is 5, so 5 x 1,300 = 6,500
# is the largest flat size where a full day of fills still fits the account
# with no margin. Fully deployed at worst, never over. (Most sessions fill 2;
# the busiest in the sample filled 4.)
#
# Market impact is NOT what caps this - measured 2026-08-06, $2,500 was a 1.5%
# median share of the entry minute's dollar volume, so at $1,300 the fill is
# invisible on a typical name. What binds instead:
#   1. Capital. At $6,500 the account itself is the ceiling.
#   2. A thin tail no size fixes. 3 of 26 entry minutes were too small to fill
#      at any size (AMSS traded $0 in its entry minute). Simulation artifacts,
#      same family as the unmodelled trading halts.
#   3. Risk per stop: -11% of $1,300 = -$143, or 2.2% of the account.
ACCOUNT_START = 6_500.0
POSITION_SIZE = 1_300.0
MAX_FILLS = 5                    # mirrors Params.max_trades_per_day
BACKTEST_TARGET_RATE = 76.9      # what the 26-trade backtest expects

# n_min=20, confidence=95% - chosen from a false-kill / power calibration
# (see Lucky/kill_switch.py), not from how they score on any actual trade
# sequence. Changing this after seeing the running average defeats the
# entire point - see the module docstring.
KILL_PARAMS = KillParams(n_min=20, confidence=0.95)
STALE_AFTER_S = 150              # live_state older than this => runner not running

HERE = Path(__file__).parent
LIVE_STATE = DATA_DIR / "live_state.json"
INTENTS = DATA_DIR / "intents.json"


def load_trades() -> pd.DataFrame:
    """Every closed trade, long and short, oldest first.

    Three sources, most-authoritative last so `drop_duplicates(keep="last")`
    lets a curated row supersede the raw log it was derived from. That matters
    because `trade_log.csv` is an append-only forward record and must stay
    untouched - it still holds trades taken under superseded parameters, and
    two entries (DXST, EVTL) that `verify_gaps` has since ruled invalid.
    `week_trades.csv` is the current-config view, carrying a `source` column
    (live / replay / backtest) so a reconstruction is never read as a fill.
    """
    frames = []
    for path, side in [(DATA_DIR / "trade_log.csv", "long"),
                       (DATA_DIR / "eod_short_log.csv", "short")]:
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if d.empty:
            continue
        d["side"], d["source"] = side, "live"
        # "is_real"/"position_size" ADDED 2026-08-07 (eod_short.py) for the
        # one real position this project has had (MB, tracked by hand) -
        # trade_log.csv has neither column (nothing on the long side has
        # ever been real money), so both default in as absent/False.
        if "is_real" not in d.columns:
            d["is_real"] = False
        if "position_size" not in d.columns:
            d["position_size"] = float("nan")
        d["curated"] = False
        frames.append(d)
    path = DATA_DIR / "week_trades.csv"
    if path.exists():
        d = pd.read_csv(path)
        d["curated"] = True
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "side", "entry_price",
                                     "outcome", "return_pct", "source", "pnl"])
    t = pd.concat(frames, ignore_index=True)
    t["date"] = pd.to_datetime(t["date"]).dt.date.astype(str)
    t["return_pct"] = pd.to_numeric(t["return_pct"], errors="coerce")
    t = t.dropna(subset=["return_pct"])
    # A curated DAY replaces the raw log for that day outright - not row by
    # row. Dropping only matching rows would strand the entries the curated
    # view deliberately removes (DXST/EVTL, invalidated by verify_gaps; the
    # 08-05 trades taken under superseded parameters). Days the curated file
    # says nothing about - next week's live sessions - pass through untouched.
    covered = set(t.loc[t.curated, "date"])
    t = t[t.curated | ~t.date.isin(covered)]
    t["pnl"] = t.return_pct / 100 * POSITION_SIZE
    # a REAL trade of a KNOWN size gets its actual $ P&L, not the paper
    # POSITION_SIZE assumption every other row here uses.
    real_sized = t["is_real"].fillna(False) & t["position_size"].notna()
    t.loc[real_sized, "pnl"] = t.loc[real_sized, "return_pct"] / 100 * t.loc[real_sized, "position_size"]
    return t.drop(columns=["curated"]).sort_values(["date", "side", "ticker"]).reset_index(drop=True)


def equity_curve(trades: pd.DataFrame) -> list[dict]:
    """Per-day realised P&L and running equity from ACCOUNT_START."""
    if trades.empty:
        return []
    by_day = (trades.groupby("date")
              .agg(pnl=("pnl", "sum"), trades=("pnl", "size"),
                   wins=("return_pct", lambda s: int((s > 0).sum())))
              .reset_index().sort_values("date"))
    by_day["equity"] = ACCOUNT_START + by_day.pnl.cumsum()
    return by_day.to_dict("records")


def summarise(trades: pd.DataFrame) -> dict:
    """Headline numbers, plus the one divergence worth watching every day:
    live target rate against what the backtest promised."""
    if trades.empty:
        return {"trades": 0, "equity": ACCOUNT_START, "pnl": 0.0, "pnl_pct": 0.0,
                "win_rate": None, "avg_pct": None, "target_rate": None,
                "backtest_target_rate": BACKTEST_TARGET_RATE, "best": None, "worst": None}
    r = trades.return_pct
    hit_target = trades.outcome.astype(str).str.startswith("TP").sum()
    pnl = float(trades.pnl.sum())
    return {"trades": int(len(trades)), "equity": ACCOUNT_START + pnl, "pnl": pnl,
            "pnl_pct": pnl / ACCOUNT_START * 100,
            "win_rate": float((r > 0).mean() * 100), "avg_pct": float(r.mean()),
            "target_rate": float(hit_target / len(trades) * 100),
            "backtest_target_rate": BACKTEST_TARGET_RATE,
            "best": float(r.max()), "worst": float(r.min())}


ET = "America/New_York"


def _minutes_since(hhmm: str | None) -> float | None:
    """Minutes from a "HH:MM" or "HH:MM:SS" entry time (assumed today, ET)
    to now. Computed server-side on purpose - the browser's clock/timezone
    is not guaranteed to be ET, and this project already got bitten once by
    a timezone assumption (runner.py's _today())."""
    if not hhmm:
        return None
    try:
        fmt = "%H:%M:%S" if hhmm.count(":") == 2 else "%H:%M"
        t = dt.datetime.strptime(hhmm, fmt).time()
    except ValueError:
        return None
    now = pd.Timestamp.now(tz=ET)
    entered = now.normalize() + pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    return max(0.0, (now - entered).total_seconds() / 60)


def _market_status() -> str:
    """OPEN / CLOSED, regular session only (09:30-16:00 ET, weekdays) - no
    holiday calendar, same known gap as _next_open()'s in runner.py."""
    now = pd.Timestamp.now(tz=ET)
    if now.weekday() >= 5:
        return "CLOSED"
    open_t = now.normalize() + pd.Timedelta(hours=9, minutes=30)
    close_t = now.normalize() + pd.Timedelta(hours=16)
    return "OPEN" if open_t <= now < close_t else "CLOSED"


def _intraday_equity() -> list[dict]:
    """Today's running P&L, one point per poll (runner.py's
    _append_equity_snapshot(), added 2026-08-07). Empty before the first
    poll of a session that's actually using the new code - old sessions
    (and anything before today) have no file to read, which is a real gap,
    not a bug: this data didn't exist until now."""
    path = DATA_DIR / f"equity_intraday_{dt.date.today().strftime('%Y%m%d')}.csv"
    if not path.exists():
        return []
    try:
        d = pd.read_csv(path)
    except (pd.errors.EmptyDataError, ValueError):
        return []
    d["total_return_pct"] = d.realised_return_pct + d.unrealised_return_pct
    d["equity"] = ACCOUNT_START + d.total_return_pct / 100 * POSITION_SIZE
    return _records(d)


def _live_state() -> dict:
    """The runner's in-flight snapshot, with a staleness verdict. A missing or
    old file means no session is running - that is normal, not an error."""
    try:
        age = dt.datetime.now().timestamp() - LIVE_STATE.stat().st_mtime
        state = json.loads(LIVE_STATE.read_text())
    except (FileNotFoundError, ValueError, OSError):
        return {"running": False, "positions": [], "closed": [], "candidates": []}
    state["running"] = age < STALE_AFTER_S
    state["age_s"] = int(age)
    return state


EOD_SHORT_STATE = DATA_DIR / "eod_short_state.json"


def _eod_short_state() -> dict | None:
    """The EOD-short module's in-flight snapshot (added 2026-08-07 - see
    eod_short.py's EOD_SHORT_STATE docstring note). Separate file from
    live_state.json on purpose: eod_short.py runs as its own process,
    independent of the intraday runner, and the two must never race writing
    the same file. None (not a dict) when nothing is tracked or the last
    poll is stale - same STALE_AFTER_S convention as the long side."""
    try:
        age = dt.datetime.now().timestamp() - EOD_SHORT_STATE.stat().st_mtime
        state = json.loads(EOD_SHORT_STATE.read_text())
    except (FileNotFoundError, ValueError, OSError):
        return None
    if state.get("resolved") or age >= STALE_AFTER_S:
        return None
    # unrealised_usd is computed at the source (eod_short.py's
    # _write_state()) from the position_size actually passed to
    # track_manual_short()/run_eod_short() - None when that size isn't
    # known, rather than falling back to the paper POSITION_SIZE constant,
    # which would misrepresent a real position of unknown size as a fact.
    if "unrealised_usd" not in state:
        state["unrealised_usd"] = None
    state["held_min"] = _minutes_since(state.get("started_at"))
    return state


BACKFILL_STATE = DATA_DIR / "backfill_state.json"
# ADDED 2026-09-13: a day with zero qualifying candidates AND an EOD-short
# NO_DATA (a real holiday like Labor Day, or just a quiet day - Aug 28 turned
# out to be the latter, not the unrecoverable case it first looked like)
# leaves candidates_YYYYMMDD.csv unwritten and no trade_log/eod_short_log
# row - runner._finish() and replay.replay_day() both skip an empty write on
# purpose (see their own "if not cands.empty" guards). Without this ledger,
# such a day would look "missing" forever and the button would re-offer it
# on every single poll despite a backfill having genuinely checked it.
# Append-only text, not JSON, so it survives independently of BACKFILL_STATE
# (which gets fully overwritten every run).
BACKFILL_CHECKED_EMPTY = DATA_DIR / "backfill_checked_empty.txt"
_backfill_lock = threading.Lock()


def _checked_empty_stamps() -> set[str]:
    try:
        return {line.strip() for line in BACKFILL_CHECKED_EMPTY.read_text().splitlines() if line.strip()}
    except FileNotFoundError:
        return set()


def _mark_checked_empty(stamps: set[str]) -> None:
    if not stamps:
        return
    with BACKFILL_CHECKED_EMPTY.open("a") as f:
        f.writelines(s + "\n" for s in sorted(stamps))


def _existing_stamps() -> set[str]:
    """YYYYMMDD stamps we already have SOME record for: a candidates file, a
    probe file (a session that ran but found nothing tradeable), a row in
    trade_log.csv/eod_short_log.csv, or a day a backfill already confirmed is
    genuinely empty. Anything not in this set, on a weekday, is a day nothing
    has ever touched."""
    stamps = {p.stem.removeprefix("candidates_") for p in DATA_DIR.glob("candidates_*.csv")}
    stamps |= {p.stem.removeprefix("probe_observations_")
              for p in DATA_DIR.glob("probe_observations_*.csv")}
    for name in ("trade_log.csv", "eod_short_log.csv"):
        path = DATA_DIR / name
        if not path.exists():
            continue
        try:
            d = pd.read_csv(path, usecols=["date"])
        except (pd.errors.EmptyDataError, ValueError):
            continue
        stamps |= set(pd.to_datetime(d.date).dt.strftime("%Y%m%d"))
    stamps |= _checked_empty_stamps()
    return stamps


def missing_sessions() -> list[dt.date]:
    """Weekdays with no record at all, from the earliest stamp on file up to
    (not including) today. Same 'no holiday calendar' gap as runner.py's
    _next_open() and replay.py: a real holiday just comes back from
    replay_day() with zero candidates, which is harmless, not a crash."""
    stamps = _existing_stamps()
    if not stamps:
        return []
    start = dt.datetime.strptime(min(stamps), "%Y%m%d").date()
    end = dt.date.today() - dt.timedelta(days=1)
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d.strftime("%Y%m%d") not in stamps:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _write_backfill_state(payload: dict) -> None:
    try:
        BACKFILL_STATE.write_text(json.dumps(payload, default=str))
    except OSError:
        log.exception("backfill state write failed")


def _backfill_state() -> dict | None:
    try:
        return json.loads(BACKFILL_STATE.read_text())
    except (FileNotFoundError, ValueError):
        return None


def _run_backfill(days: list[dt.date]) -> None:
    """Runs in a background thread - network-bound and can take minutes.
    Retries the WHOLE range on a rate limit rather than just the one call
    that hit it: replay.replay_day()'s _already_done() guard makes that safe
    (a retry just skips every day the first pass already wrote) and simpler
    than resuming mid-day."""
    from yfinance.exceptions import YFRateLimitError

    from . import replay

    start, end = min(days), max(days)
    started = pd.Timestamp.now(tz=ET).isoformat()
    _write_backfill_state({"status": "running", "range": [str(start), str(end)],
                          "target_days": len(days), "started": started})
    for attempt in range(6):
        try:
            replay.replay_range(start, end)
            # anything still unaccounted for after a clean run genuinely had
            # nothing to log (see BACKFILL_CHECKED_EMPTY above) - not a
            # failure, just a day worth remembering as "checked"
            targeted = {d.strftime("%Y%m%d") for d in days}
            _mark_checked_empty(targeted - _existing_stamps())
            _write_backfill_state({
                "status": "done", "range": [str(start), str(end)], "target_days": len(days),
                "remaining": len(missing_sessions()), "started": started,
                "finished": pd.Timestamp.now(tz=ET).isoformat()})
            return
        except YFRateLimitError:
            wait = 20 * (attempt + 1)
            _write_backfill_state({
                "status": "running", "range": [str(start), str(end)], "target_days": len(days),
                "started": started, "note": f"rate limited, retrying in {wait}s (attempt {attempt + 1}/6)"})
            time.sleep(wait)
        except Exception as exc:
            log.exception("backfill failed")
            _write_backfill_state({"status": "error", "error": str(exc), "started": started,
                                  "finished": pd.Timestamp.now(tz=ET).isoformat()})
            return
    _write_backfill_state({"status": "error", "error": "gave up after repeated rate limits",
                          "started": started, "finished": pd.Timestamp.now(tz=ET).isoformat()})


def start_backfill() -> dict:
    """Kicks off _run_backfill() in a background thread if one isn't already
    running and there's actually something missing. Returns immediately -
    progress is read back from BACKFILL_STATE (and from missing_sessions()
    shrinking as each day is written, since replay writes as it goes)."""
    if not _backfill_lock.acquire(blocking=False):
        return {"started": False, "reason": "already running"}
    days = missing_sessions()
    if not days:
        _backfill_lock.release()
        return {"started": False, "reason": "nothing missing"}

    def _target():
        try:
            _run_backfill(days)
        finally:
            _backfill_lock.release()

    threading.Thread(target=_target, daemon=True).start()
    return {"started": True, "days": len(days)}


def _records(df: pd.DataFrame) -> list[dict]:
    """Rows as JSON-safe dicts. The frames concatenated in load_trades() have
    different columns, so the gaps come out as NaN - and json.dumps writes
    those as a bare `NaN`, which Python re-reads happily and every browser
    rejects as invalid JSON. Null them before they reach the wire.
    """
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def _activity_feed(live: dict, eod_short: dict | None) -> list[dict]:
    """A real, derived-not-fabricated event log: every entry and exit this
    session actually has a timestamp for. NOT a full system log (there is
    no per-poll "scan completed" event stored anywhere to read back) - this
    is built from live_state.json's positions/closed and eod_short_state,
    not from a separate activity-tracking file. Newest first."""
    events = []
    for p in live.get("positions", []):
        events.append({"time": p.get("entry_time"), "kind": "entry",
                       "text": f"Entered {p['ticker']} @ {p.get('entry'):.4f}" if p.get("entry") is not None
                               else f"Entered {p['ticker']}"})
    for c in live.get("closed", []):
        if c.get("entry_time"):
            events.append({"time": c["entry_time"], "kind": "entry",
                           "text": f"Entered {c['ticker']} @ {c.get('entry'):.4f}" if c.get("entry") is not None
                                   else f"Entered {c['ticker']}"})
        exit_t = c.get("closed_at") or c.get("exit_time")
        if exit_t:
            events.append({"time": exit_t, "kind": "exit",
                           "text": f"{c['ticker']} closed {c.get('outcome')} "
                                   f"({c.get('return_pct', 0):+.1f}%)"})
    if eod_short:
        events.append({"time": eod_short.get("started_at"), "kind": "entry",
                       "text": f"Shorted {eod_short['ticker']} @ {eod_short.get('entry_price'):.4f}"
                               + (" [REAL]" if eod_short.get("is_real") else "")})
    events = [e for e in events if e.get("time")]
    return sorted(events, key=lambda e: e["time"], reverse=True)[:20]


def build_state() -> dict:
    trades = load_trades()
    live = _live_state()
    # Overlay intents here rather than trusting the copy in live_state: that
    # one is only as fresh as the runner's last poll, so a just-pressed button
    # would sit dead for up to a poll interval before lighting up.
    try:
        intents = json.loads(INTENTS.read_text())
    except (FileNotFoundError, ValueError):
        intents = {}
    for p in live.get("positions", []):
        p["unrealised_usd"] = (None if p.get("unrealised_pct") is None
                               else p["unrealised_pct"] / 100 * POSITION_SIZE)
        p["intent"] = intents.get(p["ticker"], {}).get("action") or p.get("intent")
        p["held_min"] = _minutes_since(p.get("entry_time"))
    # a ticker moves here the moment TP/SL/EODT fires mid-session (runner.py's
    # live exit check, added 2026-08-07) - frozen at its exit, not a mark.
    for c in live.get("closed", []):
        c["pnl"] = None if c.get("return_pct") is None else c["return_pct"] / 100 * POSITION_SIZE
    # Two views, both from ACCOUNT_START. "all" includes the seeded
    # reconstructions; "live" is only what the runner actually filled, which
    # is the number the forward test is being run to produce - blending the
    # two lets a backtest that was fitted on those same days flatter every
    # forward session that follows it.
    real = trades[trades.source == "live"] if "source" in trades else trades
    real_long = real[real.side == "long"].sort_values(["date", "entry_time"]) if len(real) else real
    kill = kill_evaluate(list(real_long.return_pct), KILL_PARAMS)
    kill["n_min"] = KILL_PARAMS.n_min
    eod_short = _eod_short_state()

    def by_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
        return df[df.side == side] if "side" in df.columns and len(df) else df
    if eod_short:
        # namespaced (see eod_short.INTENT_PREFIX) so this can never collide
        # with a same-day long position's own close/ride intent
        eod_short["intent"] = intents.get(INTENT_PREFIX + eod_short["ticker"], {}).get("action")
    return {"live": live,
            "eod_short": eod_short,
            "market_status": _market_status(),
            "intraday_equity": _intraday_equity(),
            "activity": _activity_feed(live, eod_short),
            "views": {
                "all": {"stats": summarise(trades), "history": equity_curve(trades),
                        "long": summarise(by_side(trades, "long")),
                        "short": summarise(by_side(trades, "short"))},
                "live": {"stats": summarise(real), "history": equity_curve(real),
                         "long": summarise(by_side(real, "long")),
                         "short": summarise(by_side(real, "short"))},
            },
            "kill_switch": kill,
            "trades": _records(trades),
            "missing_sessions": [str(d) for d in missing_sessions()],
            "backfill": _backfill_state(),
            "config": {"account": ACCOUNT_START, "position_size": POSITION_SIZE,
                       "max_deploy_pct": POSITION_SIZE * MAX_FILLS / ACCOUNT_START * 100,
                       "risk_per_stop": POSITION_SIZE * 0.11}}


def record_intent(ticker: str, action: str, at: str | None = None) -> dict:
    """Persist a close/ride decision. `at` is the minute the runner truncates
    the trade at; defaults to now in exchange time."""
    if action not in {"close", "ride"}:
        raise ValueError(f"unknown action {action!r}")
    try:
        intents = json.loads(INTENTS.read_text())
    except (FileNotFoundError, ValueError):
        intents = {}
    now = pd.Timestamp.now(tz="America/New_York")
    intents[ticker] = {"action": action, "at": at or now.strftime("%H:%M"),
                       "ts": now.isoformat()}
    INTENTS.write_text(json.dumps(intents, indent=2))
    return intents[ticker]


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # everything here is live state that must never be served stale
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/api/state"):
            # allow_nan=False turns a NaN leak into a 500 with a traceback in
            # the log, instead of JSON the browser silently refuses to parse
            body = json.dumps(build_state(), default=str, allow_nan=False)
            self._send(body.encode(), "application/json")
        elif self.path in ("/", "/index.html"):
            self._send((HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        if self.path.startswith("/api/intent"):
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)))
                saved = record_intent(payload["ticker"], payload["action"])
            except (ValueError, KeyError) as exc:
                return self._send(json.dumps({"error": str(exc)}).encode(),
                                  "application/json", 400)
            log.info("intent: %s -> %s", payload["ticker"], payload["action"])
            return self._send(json.dumps(saved).encode(), "application/json")
        if self.path == "/api/backfill":
            result = start_backfill()
            log.info("backfill: %s", result)
            return self._send(json.dumps(result).encode(), "application/json")
        self._send(b"not found", "text/plain", 404)

    def log_message(self, *args) -> None:
        pass                                  # one user on localhost; the access log is noise


def _self_check() -> None:
    """P&L and the equity curve are the numbers being trusted; assert them."""
    t = pd.DataFrame({"ticker": ["A", "B", "C"], "date": ["2026-08-03"] * 2 + ["2026-08-04"],
                      "outcome": ["TP", "SL", "EODT"], "return_pct": [10.0, -5.0, 2.0],
                      "side": "long"})
    t["pnl"] = t.return_pct / 100 * POSITION_SIZE
    s = summarise(t)
    unit = POSITION_SIZE / 100                               # $ per 1% of return
    # isclose, not == : dollar amounts routed through pandas sums vs. plain
    # float arithmetic land a float-epsilon apart even when they're equal.
    assert math.isclose(s["pnl"], 7.0 * unit), s["pnl"]      # (10 - 5 + 2)%
    assert math.isclose(s["equity"], ACCOUNT_START + 7.0 * unit), s["equity"]
    assert abs(s["win_rate"] - 200 / 3) < 1e-9, s["win_rate"]
    assert abs(s["target_rate"] - 100 / 3) < 1e-9, s["target_rate"]

    curve = equity_curve(t)
    assert [c["date"] for c in curve] == ["2026-08-03", "2026-08-04"]
    assert math.isclose(curve[0]["pnl"], 5.0 * unit) and curve[0]["trades"] == 2, curve[0]
    assert math.isclose(curve[-1]["equity"], ACCOUNT_START + 7.0 * unit), curve[-1]

    # a full day of fills must never need more capital than the account has
    assert POSITION_SIZE * MAX_FILLS <= ACCOUNT_START, (POSITION_SIZE, MAX_FILLS, ACCOUNT_START)
    import gap_vwap_strategy as _S
    assert MAX_FILLS == _S.DEFAULT.max_trades_per_day, "fill cap drifted from the strategy"

    empty = summarise(pd.DataFrame(columns=["return_pct", "outcome", "pnl"]))
    assert empty["trades"] == 0 and empty["equity"] == ACCOUNT_START
    assert equity_curve(pd.DataFrame(columns=["date", "pnl", "return_pct"])) == []

    # the payload must survive a STRICT parser: pandas leaves NaN in the gaps
    # between differently-shaped frames, and json.dumps writes those as a bare
    # NaN that Python re-reads but no browser will
    ragged = pd.concat([t, pd.DataFrame([{"ticker": "D", "date": "2026-08-05",
                                          "return_pct": 1.0, "extra": 5}])],
                       ignore_index=True)
    json.dumps(_records(ragged), allow_nan=False)
    state = build_state()
    json.dumps(state, default=str, allow_nan=False)

    # the live view must never be flattered by a reconstruction
    all_n = state["views"]["all"]["stats"]["trades"]
    live_n = state["views"]["live"]["stats"]["trades"]
    assert live_n <= all_n, (live_n, all_n)
    assert live_n == sum(t.get("source") == "live" for t in state["trades"]), live_n

    # kill switch tracks real LONG fills only - a backtest-heavy week must
    # not feed it, or the "decide it before the data arrives" guarantee breaks
    ks = state["kill_switch"]
    live_long_n = sum(t.get("source") == "live" and t.get("side") == "long" for t in state["trades"])
    assert ks["n"] == live_long_n, (ks["n"], live_long_n, "kill switch counted a non-live trade")
    assert ks["n_min"] == KILL_PARAMS.n_min
    if live_long_n < KILL_PARAMS.n_min:
        assert ks["verdict"] == "TOO_EARLY", ks

    # missing_sessions must only ever name past weekdays - never today/future
    # (a session in progress isn't "missing") and never a weekend (nothing
    # runs on one, so it can never be backfilled and must not show as a gap)
    assert isinstance(state["missing_sessions"], list)
    for d in missing_sessions():
        assert d < dt.date.today(), (d, "missing_sessions must not include today or later")
        assert d.weekday() < 5, (d, "missing_sessions must not include a weekend")
    print("dashboard self-check passed: 5/5")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = ap.parse_args()
    _self_check()
    url = f"http://localhost:{args.port}"
    # bind now so the socket is already accepting connections (queued in the
    # OS backlog) by the time the browser tab requests it, below
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    log.info("dashboard on %s  (DRY RUN - no broker connected)", url)
    if not args.no_browser:
        # mirrors runner.py's serve_dashboard() - a missing/odd browser must
        # never be fatal, this is a convenience, not a requirement
        import webbrowser
        try:
            webbrowser.get("safari").open(url)
        except Exception:
            try:
                webbrowser.open(url)
            except Exception:
                log.info("could not open a browser - visit %s yourself", url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
