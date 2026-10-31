"""
polymarket_tables.py -- fills in the analysis-plan tables (A, B, C, C-D, E) for any group of markets.

Put this file next to polymarket_toolkit.py and run it cell by cell in VS Code (# %% cells),
or import it:  import polymarket_tables as tb

How it works (built for ~1 million markets):
  * Market size and category mix come from Postgres.
  * Trades are STREAMED one day at a time through DuckDB. Only small per-wallet / per-market
    totals are kept (data/agg/<group>/...), never the raw rows, so the all-markets group fits.
  * Liquidity uses one order-book snapshot per day (12:00 UTC by default), not every hour.
  * Every step saves its output and SKIPS work already done, so if it stops (Wi-Fi, laptop
    asleep) just run it again and it picks up where it left off.
  * Final tables are printed and saved as CSV in results/ for copying into the doc.

Access to Postgres and CloudFront ends Oct 29 2026, 11:59pm PDT. Run the "collect" cells before
then; the "tables" cells only read the saved summaries and work any time afterwards.
"""

# %% Setup
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import polymarket_toolkit as pt
from polymarket_toolkit import con, sql, sql_saved, load, save, is_saved, path_for, token_table, gini

START, END = "2026-06-01", "2026-10-05"          # analysis window (UTC), inclusive
END_EXCL = str((pd.Timestamp(END) + pd.Timedelta(days=1)).date())
SNAPSHOT_HOUR = 12                                # which hour's order-book snapshot to sample each day

AGG = pt.DATA_DIR / "agg"                         # small per-day summaries live here
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Group definitions: SQL WHERE clauses on markets m / events e.
# "Active in window" = the market was open at some point between START and END.
ACTIVE = "m.start_date < :end_excl AND m.end_date >= :start"
WORLD_CUP = ("(m.title ILIKE :wc1 OR e.title ILIKE :wc1 "
             "OR m.title ILIKE :wc2 OR e.title ILIKE :wc2)")
PARAMS = {"start": START, "end_excl": END_EXCL, "wc1": "%world cup%", "wc2": "%fifa%"}

GROUPS = {
    "all_active": ACTIVE,
    "worldcup_active": f"{ACTIVE} AND {WORLD_CUP}",
    # For "World Cup vs. everything else", add this group and use it instead of all_active in the tables:
    # "nonwc_active": f"{ACTIVE} AND NOT {WORLD_CUP}",
}

# Trades and snapshots are downloaded ONCE, for this group. Every other group is a subset of it,
# so its summaries are made by filtering (derive_group), with no second download.
PARENT = "all_active"


def _params_for(where):
    '''Only the parameters this WHERE clause uses.'''
    return {k: v for k, v in PARAMS.items() if f":{k}" in where}


def _save_csv(df, group, table):
    path = RESULTS / f"{group}_{table}.csv"
    df.to_csv(path)
    print(f"  → {path}")
    return df


# %% ---------------------------------------------------------------------------
# 1. COLLECT: market lists and resolutions (Postgres)
# -----------------------------------------------------------------------------

def group_markets(group, refresh=False):
    '''Markets in a group (saved as data/<group>.parquet).'''
    where = GROUPS[group]
    return sql_saved(group, f'''
        SELECT m.slug, m.title, e.title AS event_title, m.condition_id,
               m.token_ids, m.outcomes, m.start_date, m.end_date, m.volume_clob
        FROM markets m
        LEFT JOIN events e ON e.slug = m.event_slug
        WHERE {where}
    ''', refresh=refresh, **_params_for(where))


def group_resolutions(group, refresh=False):
    '''Resolutions for a group's markets, joined on the server (data/<group>_resolutions.parquet).'''
    where = GROUPS[group]
    return sql_saved(f"{group}_resolutions", f'''
        SELECT r.market_slug, r.resolved_yes, r.resolved_at
        FROM resolutions r
        JOIN markets m ON m.slug = r.market_slug
        LEFT JOIN events e ON e.slug = m.event_slug
        WHERE {where}
    ''', refresh=refresh, **_params_for(where))


# %% ---------------------------------------------------------------------------
# 2. COLLECT: stream wallet trades, keep only daily summaries
# -----------------------------------------------------------------------------
# Per day it writes three small files under data/agg/<group>/:
#   wm/<day>.parquet    one row per wallet x market x month: first trade time, volume, trades,
#                       size counts, BUY shares/cost
#   hist/<day>.parquet  trade-size histogram per market (log-spaced bins) for medians and the chart
#   ttr/<day>.parquet   volume per market by time before resolution and before end_date

TTR_SQL = '''CASE
    WHEN h < 24      THEN '5 last 24 hours'
    WHEN h < 48      THEN '4 1–2 days'
    WHEN h < 24*7    THEN '3 2–7 days'
    WHEN h < 24*30   THEN '2 8–30 days'
    ELSE                  '1 more than 30 days' END'''

KICKOFF_SQL = '''CASE
    WHEN h < 0       THEN '6 after end'
    WHEN h < 1       THEN '5 last hour'
    WHEN h < 6       THEN '4 1–6 hours'
    WHEN h < 24      THEN '3 6–24 hours'
    WHEN h < 24*7    THEN '2 1–7 days'
    ELSE                  '1 more than 7 days' END'''


def collect_trades(group, markets, res, start=START, end=END):
    '''Stream every day's trades for this group into small summaries (resumable).'''
    out = AGG / group
    for sub in ("wm", "hist", "ttr"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    con.register("mk", markets[["slug", "end_date"]])
    con.register("res", res[["market_slug", "resolved_at"]])

    days = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d")
    for i, day in enumerate(days, 1):
        if (out / "wm" / f"{day}.parquet").exists():
            continue                                     # done on an earlier run
        src = path_for(f"wallet_trades/trades-{day}.parquet")
        try:
            con.execute(f'''
                CREATE OR REPLACE TEMP TABLE day AS
                SELECT t.wallet, t.market, t.time, t.side, t.outcome, t.price, t.size, t.dollars,
                       mk.end_date
                FROM '{src}' t JOIN mk ON mk.slug = t.market
            ''')
        except Exception as e:
            print(f"  skipped {day}: {str(e).splitlines()[0][:120]}")
            continue

        con.execute(f'''COPY (
            SELECT wallet, market, date_trunc('month', time) AS month,
                   MIN(time)                         AS first_time,
                   SUM(dollars)                      AS volume,
                   COUNT(*)                          AS n_trades,
                   SUM((dollars < 10)::INT)          AS n_under_10,
                   SUM((dollars > 1000)::INT)        AS n_over_1000,
                   SUM(CASE WHEN side = 'BUY' AND outcome = 'Yes' THEN size END)         AS buy_yes_shares,
                   SUM(CASE WHEN side = 'BUY' AND outcome = 'Yes' THEN size * price END) AS buy_yes_cost,
                   SUM(CASE WHEN side = 'BUY' AND outcome = 'No'  THEN size END)         AS buy_no_shares,
                   SUM(CASE WHEN side = 'BUY' AND outcome = 'No'  THEN size * price END) AS buy_no_cost
            FROM day GROUP BY ALL
        ) TO '{(out / "wm" / f"{day}.parquet").as_posix()}' (FORMAT parquet)''')

        con.execute(f'''COPY (
            SELECT market, FLOOR(LOG10(dollars) * 50)::INT AS bin, COUNT(*) AS n
            FROM day WHERE dollars > 0 GROUP BY ALL
        ) TO '{(out / "hist" / f"{day}.parquet").as_posix()}' (FORMAT parquet)''')

        con.execute(f'''COPY (
            WITH t AS (
                SELECT d.market, d.dollars,
                       EXTRACT(EPOCH FROM (r.resolved_at - d.time)) / 3600 AS h_res,
                       EXTRACT(EPOCH FROM (d.end_date   - d.time)) / 3600 AS h_end
                FROM day d LEFT JOIN res r ON r.market_slug = d.market
            )
            SELECT market,
                   CASE WHEN h_res IS NULL THEN NULL ELSE {TTR_SQL.replace("h <", "h_res <")} END AS ttr_bucket,
                   CASE WHEN h_end IS NULL THEN NULL ELSE {KICKOFF_SQL.replace("h <", "h_end <")} END AS end_bucket,
                   SUM(dollars) AS volume, COUNT(*) AS n_trades
            FROM t GROUP BY ALL
        ) TO '{(out / "ttr" / f"{day}.parquet").as_posix()}' (FORMAT parquet)''')

        if i % 7 == 0 or i == len(days):
            print(f"  {group}: {i}/{len(days)} days")
    con.execute("DROP TABLE IF EXISTS day")


# %% ---------------------------------------------------------------------------
# 3. COLLECT: sample order-book snapshots (one hour per day)
# -----------------------------------------------------------------------------

def collect_liquidity(group, markets, start=START, end=END, hour=SNAPSHOT_HOUR):
    '''YES-token top of book at `hour` UTC each day for this group (resumable).'''
    out = AGG / group / "snap"
    out.mkdir(parents=True, exist_ok=True)
    tok = token_table(markets)
    con.register("tok", tok[tok["is_yes"]][["slug", "asset_id"]])

    days = pd.date_range(start, end, freq="D")
    for i, d in enumerate(days, 1):
        h = f"{d:%Y-%m-%d}T{hour:02d}"
        dest = out / f"{h}.parquet"
        if dest.exists() or pt.in_gap(h):
            continue
        src = path_for(f"snap-{h}.parquet")
        try:
            con.execute(f'''COPY (
                SELECT epoch_ms(s.ts_ms) AS hour, t.slug,
                       s.bids[1].price AS best_bid, s.asks[1].price AS best_ask,
                       s.bids[1].size  AS bid_size, s.asks[1].size  AS ask_size,
                       len(s.bids)     AS bid_levels, len(s.asks)   AS ask_levels
                FROM '{src}' s JOIN tok t ON t.asset_id = s.asset_id
            ) TO '{dest.as_posix()}' (FORMAT parquet)''')
        except Exception as e:
            print(f"  skipped {h}: {str(e).splitlines()[0][:120]}")
        if i % 14 == 0 or i == len(days):
            print(f"  {group} snapshots: {i}/{len(days)} days")


def derive_group(child, parent=PARENT):
    '''Make a subgroup's summaries by filtering the parent's (no downloading).'''
    con.register("child_mk", group_markets(child)[["slug"]])
    for sub, key in (("wm", "market"), ("hist", "market"), ("ttr", "market"), ("snap", "slug")):
        src = (AGG / parent / sub / "*.parquet").as_posix()
        dest_dir = AGG / child / sub
        dest_dir.mkdir(parents=True, exist_ok=True)
        for old in dest_dir.glob("*.parquet"):
            old.unlink()                                # always rebuilt from the parent's latest files
        con.execute(f'''COPY (SELECT f.* FROM '{src}' f JOIN child_mk c ON c.slug = f.{key})
                       TO '{(dest_dir / "all.parquet").as_posix()}' (FORMAT parquet)''')
    print(f"  {child}: derived from {parent}")


# %% ---------------------------------------------------------------------------
# 4. TABLES (read only the saved files; work after the deadline)
# -----------------------------------------------------------------------------

CATEGORY_PATTERNS = {        # first match wins; inspect the "Other" sample and adjust
    "Sports": r"world cup|fifa|\bnba\b|\bnfl\b|\bmlb\b|\bnhl\b|premier league|champions league|la liga|serie a"
              r"|bundesliga|\bufc\b|tennis|wimbledon|us open|\bgolf\b|\bpga\b|masters|formula 1|\bf1\b"
              r"|grand prix|super bowl|stanley cup|world series|olympic|\bvs\.?\b|top 20|top 10 at",
    "Crypto": r"bitcoin|\bbtc\b|ethereum|\beth\b|solana|\bsol\b|crypto|\bxrp\b|dogecoin|token|airdrop|memecoin",
    "Economics": r"\bfed\b|interest rate|rate cut|rate hike|inflation|\bcpi\b|\bgdp\b|recession|unemployment"
                 r"|jobs report|treasury|tariff|s&p|nasdaq|dow jones|stock|earnings|ipo",
    "Politics": r"election|president|senate|house|congress|governor|mayor|trump|biden|harris|vance|primary"
                r"|parliament|prime minister|vote|party|democrat|republican|minister|supreme court|nominee",
}

WC_CATEGORY_PATTERNS = {
    "Tournament winner": r"win the (?:2026 )?(?:fifa )?world cup",
    "Awards": r"golden ball|golden glove|best young|award|\bmvp\b",
    "Player props": r"top scorer|golden boot|\bgoals?\b|\bscore\b|assist|hat.?trick",
    "Game winner": r"\bvs\.?\b|\bbeat\b|\bdraw\b|\bwin on\b",
}
WC_STAGE_PATTERNS = {
    "Knockout": r"round of (?:32|16)|quarter.?final|semi.?final|\bfinal\b|third.place",
    "Group stage": r"\bgroup [a-l]\b|group stage|matchday",
}


def classify(markets, patterns, other="Other"):
    '''Category per market from keywords in its title or event title.'''
    text = (markets["title"].fillna("") + " | " + markets["event_title"].fillna("")).str.lower()
    cat = pd.Series(other, index=markets.index)
    unassigned = pd.Series(True, index=markets.index)
    for name, pattern in patterns.items():
        hit = unassigned & text.str.contains(pattern, regex=True)
        cat[hit] = name
        unassigned &= ~hit
    return cat


def _wm(group):
    return f"'{(AGG / group / 'wm' / '*.parquet').as_posix()}'"


def market_volume(group):
    '''Window trade volume and trades per market.'''
    return con.sql(f"SELECT market AS slug, SUM(volume) AS volume, SUM(n_trades) AS n_trades "
                   f"FROM {_wm(group)} GROUP BY 1").df()


def table_A(group, markets, res, wc=False):
    '''A. Market size + category mix + per-month created/resolved.'''
    start, end_excl = pd.Timestamp(START, tz="UTC"), pd.Timestamp(END_EXCL, tz="UTC")
    s = pd.to_datetime(markets["start_date"], utc=True)
    e = pd.to_datetime(markets["end_date"], utc=True)
    r = pd.to_datetime(res["resolved_at"], utc=True)
    length = (e - s).dt.total_seconds() / 86400

    size = pd.Series({
        "Markets active in window": len(markets),
        "Markets created in window": int(((s >= start) & (s < end_excl)).sum()),
        "Markets resolved in window": int(((r >= start) & (r < end_excl)).sum()),
        "Market length, median (days)": length.median(),
        "Market length, mean (days)": length.mean(),
    }, name="value")

    cats = classify(markets, WC_CATEGORY_PATTERNS if wc else CATEGORY_PATTERNS)
    vol = market_volume(group).set_index("slug")["volume"]
    mix = pd.DataFrame({"category": cats.values,
                        "volume": markets["slug"].map(vol).fillna(0).values})
    mix = mix.groupby("category").agg(n_markets=("volume", "size"), volume_usd=("volume", "sum"))
    mix["pct_of_markets"] = mix["n_markets"] / mix["n_markets"].sum()
    mix["pct_of_volume"] = mix["volume_usd"] / mix["volume_usd"].sum()
    mix = mix[["n_markets", "pct_of_markets", "volume_usd", "pct_of_volume"]]

    created = s[(s >= start) & (s < end_excl)].dt.tz_localize(None).dt.to_period("M").value_counts()
    resolved = r[(r >= start) & (r < end_excl)].dt.tz_localize(None).dt.to_period("M").value_counts()
    monthly = pd.DataFrame({"markets_created": created, "markets_resolved": resolved}).fillna(0).astype(int)
    monthly.index = monthly.index.astype(str)

    print(f"\n=== {group}: A. Market size ===");     print(_save_csv(size, group, "A_market_size"))
    print(f"\n=== {group}: Category mix ===");       print(_save_csv(mix, group, "A_category_mix"))
    return size, mix, monthly.sort_index()


def _median_from_hist(group):
    '''Approximate median trade size from the log histogram (bins are 1/50 of a decade).'''
    h = con.sql(f"SELECT bin, SUM(n) AS n FROM '{(AGG / group / 'hist' / '*.parquet').as_posix()}' "
                f"GROUP BY bin ORDER BY bin").df()
    cum = h["n"].cumsum() / h["n"].sum()
    b = h.loc[cum >= 0.5, "bin"].iloc[0]
    return 10 ** ((b + 0.5) / 50), h


def table_B(group, monthly_markets):
    '''B. Trading volume + per month.'''
    tot = con.sql(f"SELECT SUM(volume) AS v, SUM(n_trades) AS n FROM {_wm(group)}").df().iloc[0]
    median, _ = _median_from_hist(group)
    mv = market_volume(group)["volume"].sort_values(ascending=False)
    top_n = max(1, int(np.ceil(len(mv) * 0.01)))
    volume = pd.Series({
        "Total volume (USD)": tot["v"],
        "Total trades": tot["n"],
        "Average trade size (USD)": tot["v"] / tot["n"],
        "Median trade size (USD, approx.)": median,
        "Share of volume in top 1% of markets": mv.head(top_n).sum() / mv.sum(),
    }, name="value")

    m = con.sql(f'''SELECT strftime(month, '%Y-%m') AS month, SUM(n_trades) AS trades,
                           SUM(volume) AS volume_usd, COUNT(DISTINCT wallet) AS unique_traders
                    FROM {_wm(group)} GROUP BY 1 ORDER BY 1''').df().set_index("month")
    per_month = monthly_markets.join(m, how="outer").fillna(0)

    print(f"\n=== {group}: B. Trading volume ===");  print(_save_csv(volume, group, "B_trading_volume"))
    print(f"\n=== {group}: Per month ===");          print(_save_csv(per_month, group, "B_per_month"))
    return volume, per_month


def wallet_table(group):
    '''One row per wallet: volume, trades, markets.'''
    return con.sql(f'''SELECT wallet, SUM(volume) AS volume, SUM(n_trades) AS n_trades,
                              COUNT(DISTINCT market) AS n_markets
                       FROM {_wm(group)} GROUP BY wallet''').df()


def table_E(group, res):
    '''E. Trader behavior + profitability by size group.'''
    w = wallet_table(group)
    counts = con.sql(f"SELECT SUM(n_trades) n, SUM(n_under_10) u, SUM(n_over_1000) o FROM {_wm(group)}").df().iloc[0]
    top_n = max(1, int(np.ceil(len(w) * 0.01)))
    behavior = pd.Series({
        "Unique traders (wallets)": len(w),
        "Share of volume from top 1% of wallets": w["volume"].nlargest(top_n).sum() / w["volume"].sum(),
        "Gini coefficient (volume per wallet)": gini(w["volume"]),
        "% of trades under $10": counts["u"] / counts["n"],
        "% of trades over $1,000": counts["o"] / counts["n"],
        "% one-time traders (1 trade)": (w["n_trades"] == 1).mean(),
        "Markets per trader, median": w["n_markets"].median(),
    }, name="value")

    # Size groups by window volume
    w = w.sort_values("volume", ascending=False).reset_index(drop=True)
    share = (np.arange(len(w)) + 1) / len(w)
    w["size_group"] = np.select([share <= 0.01, share <= 0.10], ["Top 1%", "Next 9%"], "Bottom 90%")

    # Profit of BUYs held to resolution: shares of the winning side pay $1, minus what was paid
    con.register("resolved", res[["market_slug", "resolved_yes"]])
    p = con.sql(f'''
        SELECT wm.wallet,
               SUM(COALESCE(buy_yes_cost, 0) + COALESCE(buy_no_cost, 0)) AS buy_volume,
               SUM(CASE WHEN r.resolved_yes THEN COALESCE(buy_yes_shares, 0)
                        ELSE COALESCE(buy_no_shares, 0) END
                   - COALESCE(buy_yes_cost, 0) - COALESCE(buy_no_cost, 0))  AS profit
        FROM {_wm(group)} wm JOIN resolved r ON r.market_slug = wm.market
        GROUP BY wm.wallet
    ''').df()
    w = w.merge(p, on="wallet", how="left").fillna({"buy_volume": 0, "profit": 0})
    prof = w.groupby("size_group").agg(n_wallets=("wallet", "size"),
                                       volume_usd=("volume", "sum"),
                                       resolved_buy_volume_usd=("buy_volume", "sum"),
                                       profit_usd=("profit", "sum"))
    prof["profit_per_1_traded"] = prof["profit_usd"] / prof["resolved_buy_volume_usd"]
    prof = prof.reindex(["Top 1%", "Next 9%", "Bottom 90%"])

    print(f"\n=== {group}: E. Trader behavior ==="); print(_save_csv(behavior, group, "E_trader_behavior"))
    print(f"\n=== {group}: Profitability ===");     print(_save_csv(prof, group, "E_profitability"))
    return behavior, prof


def trade_size_chart(groups):
    '''Trade-size distribution (share of trades per size bin), one line per group.'''
    fig, ax = plt.subplots(figsize=(9, 4))
    for g, color in zip(groups, ["#2a78d6", "#eb6834", "#1baf7a"]):
        _, h = _median_from_hist(g)
        ax.plot(10 ** ((h["bin"] + 0.5) / 50), h["n"] / h["n"].sum(), color=color, lw=2, label=g)
    ax.set_xscale("log")
    ax.set_xlabel("trade size (USD, log scale)")
    ax.set_ylabel("share of trades")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS / "E_trade_size_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def _snaps(group):
    s = con.sql(f"SELECT * FROM '{(AGG / group / 'snap' / '*.parquet').as_posix()}'").df()
    s["spread"] = s["best_ask"] - s["best_bid"]
    return s


def table_C(group, res):
    '''C. Liquidity + spread by time until resolution (merged with C-D volume).'''
    s = _snaps(group)
    two_sided = s.dropna(subset=["spread"])
    per_market = two_sided.groupby("slug")["spread"].median().rename("median_spread")
    vol = market_volume(group).set_index("slug")["volume"]
    pm = pd.concat([per_market, vol], axis=1, join="inner")
    liquidity = pd.Series({
        "Bid-ask spread, median": two_sided["spread"].median(),
        "Bid-ask spread, mean": two_sided["spread"].mean(),
        "Depth: shares at best bid, median": s["bid_size"].median(),
        "Depth: shares at best ask, median": s["ask_size"].median(),
        "Depth: price levels per side, median": pd.concat([s["bid_levels"], s["ask_levels"]]).median(),
        "Spread vs. volume (Spearman correlation, per market)": pm["median_spread"].rank().corr(pm["volume"].rank()),   # Spearman = correlation of ranks
        "Snapshot rows with no bid or no ask (share)": s["spread"].isna().mean(),
    }, name="value")

    # Spread by time until resolution (from snapshots) + volume by time until resolution (from trades)
    t = two_sided.merge(res[["market_slug", "resolved_at"]], left_on="slug", right_on="market_slug")
    h = (pd.to_datetime(t["resolved_at"], utc=True) - pd.to_datetime(t["hour"], utc=True)).dt.total_seconds() / 3600
    t["bucket"] = pd.cut(h, [-np.inf, 24, 48, 24 * 7, 24 * 30, np.inf], right=False,
                         labels=["5 last 24 hours", "4 1–2 days", "3 2–7 days", "2 8–30 days", "1 more than 30 days"])
    spread_by = t[h >= 0].groupby("bucket", observed=False)["spread"].median().rename("median_spread")
    spread_by.index = spread_by.index.astype(str)
    vol_by = con.sql(f"SELECT ttr_bucket AS bucket, SUM(volume) AS volume_usd "
                     f"FROM '{(AGG / group / 'ttr' / '*.parquet').as_posix()}' "
                     f"WHERE ttr_bucket IS NOT NULL GROUP BY 1").df().set_index("bucket")["volume_usd"]
    life = pd.concat([spread_by, vol_by], axis=1).sort_index()
    life["pct_of_volume"] = life["volume_usd"] / life["volume_usd"].sum()
    life.index = [str(i)[2:] for i in life.index]          # drop the sort prefix

    print(f"\n=== {group}: C. Liquidity ===");          print(_save_csv(liquidity, group, "C_liquidity"))
    print(f"\n=== {group}: C–D. Over a market's life ==="); print(_save_csv(life, group, "CD_market_life"))
    return liquidity, life, pm


def spread_vs_volume_chart(pm, group):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(pm["volume"], pm["median_spread"], s=8, alpha=0.3, color="#2a78d6", edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("market trade volume in window (USD, log scale)")
    ax.set_ylabel("median bid-ask spread")
    ax.set_title(f"Spread vs. volume: {group}", loc="left")
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS / f"{group}_C_spread_vs_volume.png", dpi=150, bbox_inches="tight")
    plt.show()


def table_WC_extras(group, markets):
    '''World Cup only: stage mix, top events (games), top player-prop markets, volume before kickoff.'''
    cats = classify(markets, WC_CATEGORY_PATTERNS)
    vol = market_volume(group).set_index("slug")["volume"]
    m = markets.assign(category=cats.values, volume=markets["slug"].map(vol).fillna(0).values)

    games = m[m["category"] == "Game winner"].copy()
    games["stage"] = classify(games, WC_STAGE_PATTERNS, other="Unknown").values
    stage = games.groupby("stage").agg(n_markets=("volume", "size"), volume_usd=("volume", "sum"))
    top_events = (games.groupby("event_title")["volume"].sum().nlargest(10).rename("volume_usd").to_frame())
    top_props = (m[m["category"] == "Player props"].nlargest(10, "volume")[["title", "volume"]]
                 .rename(columns={"volume": "volume_usd"}).set_index("title"))

    k = con.sql(f"SELECT market, end_bucket, SUM(volume) AS volume_usd "
                f"FROM '{(AGG / group / 'ttr' / '*.parquet').as_posix()}' GROUP BY ALL").df()
    k = k[k["market"].isin(games["slug"])].groupby("end_bucket")["volume_usd"].sum().sort_index()
    kickoff = k.to_frame()
    kickoff["pct_of_game_volume"] = kickoff["volume_usd"] / kickoff["volume_usd"].sum()
    kickoff.index = [str(i)[2:] for i in kickoff.index]

    for title, df, key in [("Game stage", stage, "WC_stage"), ("Top 10 games", top_events, "WC_top_games"),
                           ("Top 10 player-prop markets", top_props, "WC_top_player_props"),
                           ("Volume before kickoff (end_date proxy)", kickoff, "WC_kickoff")]:
        print(f"\n=== {group}: {title} ==="); print(_save_csv(df, group, key))
    return stage, top_events, top_props, kickoff


def overlap(group_wc, group_all):
    '''World Cup traders: did they also trade other markets? Was their first trade (in the window) World Cup?'''
    con.register("wcw", wallet_table(group_wc)[["wallet"]])
    con.register("wcm", load(group_wc)[["slug"]])
    both = f"(SELECT * FROM {_wm(group_all)} UNION ALL SELECT * FROM {_wm(group_wc)})"
    r = con.sql(f'''
        WITH w AS (SELECT wallet, market, MIN(first_time) AS ft
                   FROM {both} JOIN wcw USING (wallet) GROUP BY ALL),
        per_wallet AS (
            SELECT wallet,
                   BOOL_OR(market NOT IN (SELECT slug FROM wcm)) AS traded_other,
                   arg_min(market, ft)                           AS first_market
            FROM w GROUP BY wallet)
        SELECT COUNT(*)                                                AS n,
               AVG(traded_other::INT)                                  AS other,
               AVG((first_market IN (SELECT slug FROM wcm))::INT)      AS first_wc
        FROM per_wallet
    ''').df().iloc[0]
    out = pd.Series({"World Cup traders": r["n"],
                     "% of World Cup traders who also traded other markets": r["other"],
                     "% whose first trade in the window was a World Cup market": r["first_wc"]},
                    name="value")
    print("\n=== World Cup overlap ==="); print(_save_csv(out, group_wc, "E_overlap"))
    return out


# %% ---------------------------------------------------------------------------
# RUN: collect (needs access, before Oct 29 11:59pm PDT). Safe to re-run: done work is skipped.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    for g in GROUPS:                                     # market lists + resolutions (quick)
        mk, rs = group_markets(g), group_resolutions(g)
        print(f"{g}: {len(mk):,} markets, {len(rs):,} resolutions")
    mk, rs = load(PARENT), load(f"{PARENT}_resolutions")
    collect_trades(PARENT, mk, rs)                       # the long part: one pass over every day
    collect_liquidity(PARENT, mk)
    for g in GROUPS:
        if g != PARENT:
            derive_group(g)

# %% RUN: tables (reads saved files only)
if __name__ == "__main__":
    for g in GROUPS:
        mk, rs = load(g), load(f"{g}_resolutions")
        size, mix, monthly = table_A(g, mk, rs, wc=g.startswith("worldcup"))
        table_B(g, monthly)
        table_E(g, rs)
        liq, life, pm = table_C(g, rs)
        spread_vs_volume_chart(pm, g)
    trade_size_chart(list(GROUPS))
    table_WC_extras("worldcup_active", load("worldcup_active"))
    overlap("worldcup_active", "all_active")
