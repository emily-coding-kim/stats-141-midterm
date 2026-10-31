"""
export_tableau.py -- Tableau-ready CSV files from the saved Polymarket summaries.

Run next to polymarket_tables.py and polymarket_toolkit.py, after the tables script collected data.
Works offline (reads data/ and data/agg/ only). Writes to tableau/:

  markets.csv              one row per active market: category, World Cup flag, volume, tier, rank, outcome
  category_summary.csv     per group x category: markets, traded markets, volume, shares
  monthly.csv              per group x month: markets created/resolved, trades, volume, unique traders
  concentration.csv        cumulative share of volume by top-x% of markets and of wallets (for curves)
  wallets_top1000.csv      top 1,000 wallets by volume: volume, trades, markets, winnings, profit
  wallet_groups.csv        per group x size group (top 1% / next 9% / bottom 90%): wallets, volume, winnings, profit
  sankey_links.csv         per group: Category -> wallet group, value = winnings (USD), for a Sankey
  sankey_paths.csv         per group: wallet group -> win size -> category, value = winnings (USD), n_wins

All money is USD; all dates UTC; window Jun 1 - Oct 5 2026.
"""

# %% Setup
from pathlib import Path

import numpy as np
import pandas as pd

import polymarket_tables as tb
from polymarket_toolkit import con, load

OUT = Path("tableau")
OUT.mkdir(exist_ok=True)
PARENT, WC = "all_active", "worldcup_active"
INCLUDE_UNTRADED = True        # False drops markets with no trades from markets.csv (much smaller file)


def write(df, name):
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"{path}: {len(df):,} rows, {path.stat().st_size / 1e6:.1f} MB")


def wm(group):
    return tb._wm(group)


wc_slugs = set(load(WC)["slug"])

# %% markets.csv: one row per active market
mk = load(PARENT)[["slug", "title", "event_title", "start_date", "end_date"]]
mv = tb.market_volume(PARENT)
res = load(f"{PARENT}_resolutions")
m = (mk.merge(mv, on="slug", how="left")
       .merge(res, left_on="slug", right_on="market_slug", how="left")
       .drop(columns="market_slug")
       .fillna({"volume": 0, "n_trades": 0}))
m["category"] = tb.classify(m, tb.CATEGORY_PATTERNS).values
m["world_cup"] = m["slug"].isin(wc_slugs)
m["wc_category"] = np.where(m["world_cup"], tb.classify(m, tb.WC_CATEGORY_PATTERNS).values, "")
m["traded"] = np.where(m["n_trades"] > 0, "Traded", "No trades")
cut = m.loc[m["volume"] > 0, "volume"].quantile(0.99)                       # top-1% threshold
m["volume_tier"] = pd.cut(m["volume"], [-1, 0, 1_000, 10_000, cut, np.inf],
                          labels=["1 No trades", "2 Under $1k", "3 $1k–$10k",
                                  f"4 $10k–${cut / 1e3:,.0f}k", "5 Top 1%"]).astype(str)
m["length_days"] = (pd.to_datetime(m["end_date"], utc=True)
                    - pd.to_datetime(m["start_date"], utc=True)).dt.total_seconds() / 86400
m = m.sort_values("volume", ascending=False).reset_index(drop=True)
m["volume_rank"] = np.arange(1, len(m) + 1)
m["cum_volume_share"] = m["volume"].cumsum() / m["volume"].sum()
m["resolved"] = m["resolved_yes"].notna()
for col in ("start_date", "end_date", "resolved_at"):
    m[col] = pd.to_datetime(m[col], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
write(m if INCLUDE_UNTRADED else m[m["n_trades"] > 0], "markets")

# %% category_summary.csv
def category_summary(df, group):
    s = df.groupby("category").agg(n_markets=("slug", "size"),
                                   traded_markets=("n_trades", lambda x: (x > 0).sum()),
                                   volume_usd=("volume", "sum"),
                                   trades=("n_trades", "sum")).reset_index()
    s["pct_of_markets"] = s["n_markets"] / s["n_markets"].sum()
    s["pct_of_volume"] = s["volume_usd"] / s["volume_usd"].sum()
    s["volume_per_market"] = s["volume_usd"] / s["n_markets"]
    s.insert(0, "group", group)
    return s

wc_m = m[m["world_cup"]].assign(category=lambda d: d["wc_category"])
write(pd.concat([category_summary(m, "All markets"), category_summary(wc_m, "World Cup")]),
      "category_summary")

# %% monthly.csv
def monthly(group, label, markets, resolutions):
    s = pd.to_datetime(markets["start_date"], utc=True)
    r = pd.to_datetime(resolutions["resolved_at"], utc=True)
    lo, hi = pd.Timestamp(tb.START, tz="UTC"), pd.Timestamp(tb.END_EXCL, tz="UTC")
    created = s[(s >= lo) & (s < hi)].dt.strftime("%Y-%m").value_counts().rename("markets_created")
    resolved = r[(r >= lo) & (r < hi)].dt.strftime("%Y-%m").value_counts().rename("markets_resolved")
    t = con.sql(f'''SELECT strftime(month, '%Y-%m') AS month, SUM(n_trades) AS trades,
                           SUM(volume) AS volume_usd, COUNT(DISTINCT wallet) AS unique_traders
                    FROM {wm(group)} GROUP BY 1''').df().set_index("month")
    out = pd.concat([created, resolved, t], axis=1).fillna(0).rename_axis("month").reset_index()
    out.insert(0, "group", label)
    return out.sort_values("month")

write(pd.concat([monthly(PARENT, "All markets", load(PARENT), load(f"{PARENT}_resolutions")),
                 monthly(WC, "World Cup", load(WC), load(f"{WC}_resolutions"))]), "monthly")

# %% Wallet table with winnings (shared by the next three files)
con.register("resolved", res[["market_slug", "resolved_yes"]])

def wallet_winnings(group):
    '''Per wallet x category: volume, trades, markets, winnings and cost of BUYs held to resolution.'''
    return con.sql(f'''
        SELECT wm.wallet, c.category,
               SUM(wm.volume)                 AS volume_usd,
               SUM(wm.n_trades)               AS trades,
               COUNT(DISTINCT wm.market)      AS markets,
               SUM(CASE WHEN r.resolved_yes IS NULL THEN 0
                        WHEN r.resolved_yes THEN COALESCE(wm.buy_yes_shares, 0)
                        ELSE COALESCE(wm.buy_no_shares, 0) END)                   AS winnings_usd,
               SUM(CASE WHEN r.resolved_yes IS NULL THEN 0
                        ELSE COALESCE(wm.buy_yes_cost, 0) + COALESCE(wm.buy_no_cost, 0) END) AS resolved_cost_usd
        FROM {wm(group)} wm
        LEFT JOIN resolved r ON r.market_slug = wm.market
        LEFT JOIN cats c     ON c.market = wm.market
        GROUP BY ALL
    ''').df()

def size_groups(per_wallet, col):
    s = per_wallet.sort_values(col, ascending=False)
    share = np.arange(1, len(s) + 1) / len(s)
    return pd.Series(np.select([share <= 0.01, share <= 0.10], ["1 Top 1%", "2 Next 9%"], "3 Bottom 90%"),
                     index=s.index)

wallet_rows, group_rows, curve_rows, link_rows, path_rows = [], [], [], [], []

GROUP_ORDER = ["Rest of top 1%", "Next 9%", "Bottom 90%", "Not ranked"]
wallet_labels = None      # set from the ALL-markets ranking, then reused for the World Cup view


def money(v):
    if v >= 1e9: return f"${v / 1e9:,.2f}B"
    if v >= 1e6: return f"${v / 1e6:,.1f}M"
    if v >= 1e3: return f"${v / 1e3:,.0f}k"
    return f"${v:,.0f}"


def add_node_labels(df, col, order=None):
    '''Add <col>_label = "NN name · $total": sorts correctly A-Z and shows each node's winnings.'''
    tot = df.groupby(col)["value"].sum()
    order = order or list(tot.sort_values(ascending=False).index)
    order = [k for k in order if k in tot.index]
    rank = {k: i for i, k in enumerate(order, 1)}
    name = lambda k: k[2:] if k[:1].isdigit() and k[1:2] == " " else k      # drop "1 " sort prefixes
    df[f"{col}_label"] = df[col].map(lambda k: f"{rank[k]:02d} {name(k)} · {money(tot[k])}")
    return df

WIN_SIZE_SQL = '''CASE
    WHEN payout < 100     THEN '1 Under $100'
    WHEN payout < 1000    THEN '2 $100–$1k'
    WHEN payout < 10000   THEN '3 $1k–$10k'
    WHEN payout < 100000  THEN '4 $10k–$100k'
    ELSE                       '5 $100k+' END'''
for group, label in ((PARENT, "All markets"), (WC, "World Cup")):
    # Categories: the general ones for all markets, the World Cup ones for World Cup markets
    cats = (m[m["world_cup"]][["slug", "wc_category"]].rename(columns={"wc_category": "category"})
            if group == WC else m[["slug", "category"]])
    con.register("cats", cats.rename(columns={"slug": "market"}))
    wcat = wallet_winnings(group)
    pw = wcat.groupby("wallet").agg(volume_usd=("volume_usd", "sum"), trades=("trades", "sum"),
                                    winnings_usd=("winnings_usd", "sum"),
                                    resolved_cost_usd=("resolved_cost_usd", "sum"))
    pw["markets"] = con.sql(f"SELECT wallet, COUNT(DISTINCT market) AS n FROM {wm(group)} GROUP BY wallet"
                            ).df().set_index("wallet")["n"]
    pw["profit_usd"] = pw["winnings_usd"] - pw["resolved_cost_usd"]
    pw["size_group"] = size_groups(pw, "volume_usd")

    top = pw.sort_values("volume_usd", ascending=False).head(1000).reset_index()
    top.insert(0, "group", label)
    top.insert(1, "volume_rank", np.arange(1, len(top) + 1))
    wallet_rows.append(top)

    g = pw.groupby("size_group").agg(wallets=("trades", "size"), volume_usd=("volume_usd", "sum"),
                                     trades=("trades", "sum"), winnings_usd=("winnings_usd", "sum"),
                                     profit_usd=("profit_usd", "sum"),
                                     resolved_cost_usd=("resolved_cost_usd", "sum")).reset_index()
    g["profit_per_1_traded"] = g["profit_usd"] / g["resolved_cost_usd"]
    g["pct_of_volume"] = g["volume_usd"] / g["volume_usd"].sum()
    g["pct_of_winnings"] = g["winnings_usd"] / g["winnings_usd"].sum()
    g.insert(0, "group", label)
    group_rows.append(g)

    # Concentration curves, downsampled to 1,001 points each (0%, 0.1%, ..., 100%)
    mvol = (m[m["world_cup"]] if group == WC else m).loc[lambda d: d["volume"] > 0, "volume"]
    for entity, vals in (("Markets", mvol), ("Wallets", pw["volume_usd"])):
        v = np.sort(vals.to_numpy())[::-1]
        cum = np.concatenate([[0], np.cumsum(v) / v.sum()])
        pct = np.linspace(0, 1, 1001)
        curve_rows.append(pd.DataFrame({
            "group": label, "entity": entity, "top_pct_of_entities": pct,
            "cum_share_of_volume": np.interp(pct * len(v), np.arange(len(v) + 1), cum)}))

    # Sankey: category -> winning-wallet group, value = winnings.
    # Winners are ranked ONCE, on all markets (the first pass of this loop); the World Cup view reuses
    # those labels, so "#1" is the same wallet in both views.
    if wallet_labels is None:
        winners = pw[pw["winnings_usd"] > 0]
        wg = size_groups(winners, "winnings_usd").str[2:]
        top10 = winners.sort_values("winnings_usd", ascending=False).index[:10]
        wg[wg.isin(["Top 1%"])] = "Rest of top 1%"
        wg[top10] = [f"#{i} {w[:6]}…{w[-4:]}" for i, w in enumerate(top10, 1)]
        wallet_labels = wg
        WALLET_ORDER = list(wg[top10]) + GROUP_ORDER
    wg = wallet_labels.reindex(pw.index).fillna("Not ranked")
    wcat["target"] = wcat["wallet"].map(wg)
    lk = (wcat.dropna(subset=["target"]).groupby(["category", "target"], as_index=False)["winnings_usd"]
              .sum().rename(columns={"category": "source", "winnings_usd": "value"}))
    lk = lk[lk["value"] > 0]
    lk.insert(0, "group", label)
    lk["share_of_group_winnings"] = lk["value"] / lk["value"].sum()
    add_node_labels(lk, "source")
    add_node_labels(lk, "target", WALLET_ORDER)
    link_rows.append(lk)

    # 3-column Sankey: wallet group -> size of each win -> category, value = winnings.
    # A "win" = one wallet's payout from one resolved market (its winning shares x $1).
    con.register("wgroups", wg.rename("wallet_group").rename_axis("wallet").reset_index()
                            .loc[lambda d: d["wallet_group"] != "Not ranked"])
    paths = con.sql(f'''
        WITH wins AS (
            SELECT wm.wallet, wm.market, c.category,
                   SUM(CASE WHEN r.resolved_yes THEN COALESCE(wm.buy_yes_shares, 0)
                            ELSE COALESCE(wm.buy_no_shares, 0) END) AS payout
            FROM {wm(group)} wm
            JOIN resolved r ON r.market_slug = wm.market
            JOIN cats c     ON c.market = wm.market
            GROUP BY ALL
            HAVING payout > 0)
        SELECT g.wallet_group, {WIN_SIZE_SQL} AS win_size, w.category,
               SUM(w.payout) AS value, COUNT(*) AS n_wins
        FROM wins w JOIN wgroups g USING (wallet)
        GROUP BY ALL
    ''').df()
    paths.insert(0, "group", label)
    paths["share_of_group_winnings"] = paths["value"] / paths["value"].sum()
    add_node_labels(paths, "wallet_group", WALLET_ORDER)
    add_node_labels(paths, "win_size", sorted(paths["win_size"].unique()))
    add_node_labels(paths, "category")
    path_rows.append(paths)

write(pd.concat(wallet_rows), "wallets_top1000")
write(pd.concat(group_rows), "wallet_groups")
write(pd.concat(curve_rows), "concentration")
write(pd.concat(link_rows), "sankey_links")
write(pd.concat(path_rows), "sankey_paths")
print("\nDone. In Tableau: Connect → To a File → Text file, and pick a CSV from the tableau/ folder.")
