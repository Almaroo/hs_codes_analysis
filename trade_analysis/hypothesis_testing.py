import numpy as np
import polars as pl


def _segment_slope(years: list, values: list) -> float:
    if len(years) < 2:
        return float("nan")
    return float(np.polyfit(years, values, 1)[0])


def _level_around_cutoff(series: pl.DataFrame, cutoff_year: int, col: str, n: int = 2):
    before = (
        series
        .filter(pl.col("time_period") < cutoff_year)
        .sort("time_period", descending=True)
        .head(n)
        [col]
    )
    after = (
        series
        .filter(pl.col("time_period") >= cutoff_year)
        .sort("time_period")
        .head(n)
        [col]
    )
    level_before = before.mean() if len(before) > 0 else None
    level_after = after.mean() if len(after) > 0 else None
    return level_before, level_after


def _direction(slope_change: float, threshold: float) -> str:
    if slope_change <= -threshold:
        return "declining"
    if slope_change >= threshold:
        return "increasing"
    return "stable"


def screen_share_breaks(
    shares_df: pl.DataFrame,
    partner_code: str = "CN",
    cutoff_year: int = 2020,
    threshold: float = 0.5,
) -> pl.DataFrame:
    partner_df = (
        shares_df
        .filter(pl.col("partner_code") == partner_code)
        .sort("time_period")
    )

    product_codes = partner_df["product_code"].unique().sort().to_list()
    rows = []

    for pc in product_codes:
        series = partner_df.filter(pl.col("product_code") == pc)
        if len(series) == 0:
            continue

        product_name = series["product_name"][0]
        shares_pct = (series["share"] * 100).to_list()
        years = series["time_period"].to_list()

        before_idx = [i for i, y in enumerate(years) if y < cutoff_year]
        after_idx = [i for i, y in enumerate(years) if y >= cutoff_year]

        slope_before = _segment_slope(
            [years[i] for i in before_idx],
            [shares_pct[i] for i in before_idx],
        )
        slope_after = _segment_slope(
            [years[i] for i in after_idx],
            [shares_pct[i] for i in after_idx],
        )
        slope_change = slope_after - slope_before

        level_before, level_after = _level_around_cutoff(
            series.with_columns((pl.col("share") * 100).alias("share_pct")),
            cutoff_year,
            "share_pct",
        )
        level_change = (
            (level_after - level_before)
            if level_before is not None and level_after is not None
            else None
        )

        rows.append({
            "product_code": pc,
            "product_name": product_name,
            "slope_before": round(slope_before, 4),
            "slope_after": round(slope_after, 4),
            "slope_change": round(slope_change, 4),
            "level_before": round(level_before, 2) if level_before is not None else None,
            "level_after": round(level_after, 2) if level_after is not None else None,
            "level_change": round(level_change, 2) if level_change is not None else None,
            "direction": _direction(slope_change, threshold),
            "is_meaningful": abs(slope_change) >= threshold,
        })

    return pl.DataFrame(rows).sort("slope_change")


def screen_hhi_breaks(
    hhi_df: pl.DataFrame,
    cutoff_year: int = 2020,
    threshold: float = 50,
) -> pl.DataFrame:
    product_codes = hhi_df["product_code"].unique().sort().to_list()
    rows = []

    for pc in product_codes:
        series = hhi_df.filter(pl.col("product_code") == pc).sort("time_period")
        if len(series) == 0:
            continue

        hhi_vals = series["hhi"].to_list()
        years = series["time_period"].to_list()

        before_idx = [i for i, y in enumerate(years) if y < cutoff_year]
        after_idx = [i for i, y in enumerate(years) if y >= cutoff_year]

        slope_before = _segment_slope(
            [years[i] for i in before_idx],
            [hhi_vals[i] for i in before_idx],
        )
        slope_after = _segment_slope(
            [years[i] for i in after_idx],
            [hhi_vals[i] for i in after_idx],
        )
        slope_change = slope_after - slope_before

        level_before, level_after = _level_around_cutoff(series, cutoff_year, "hhi")
        level_change = (
            (level_after - level_before)
            if level_before is not None and level_after is not None
            else None
        )

        rows.append({
            "product_code": pc,
            "slope_before": round(slope_before, 2),
            "slope_after": round(slope_after, 2),
            "slope_change": round(slope_change, 2),
            "level_before": round(level_before, 2) if level_before is not None else None,
            "level_after": round(level_after, 2) if level_after is not None else None,
            "level_change": round(level_change, 2) if level_change is not None else None,
            "direction": _direction(slope_change, threshold),
            "is_meaningful": abs(slope_change) >= threshold,
        })

    return (
        pl.DataFrame(rows)
        .sort("slope_change")
    )


def compare_breakpoints(
    shares_df: pl.DataFrame,
    hhi_df: pl.DataFrame,
    partner_code: str = "CN",
) -> pl.DataFrame:
    share_2020 = screen_share_breaks(shares_df, partner_code, cutoff_year=2020, threshold=0.0)
    share_2022 = screen_share_breaks(shares_df, partner_code, cutoff_year=2022, threshold=0.0)
    hhi_2020 = screen_hhi_breaks(hhi_df, cutoff_year=2020, threshold=0.0)
    hhi_2022 = screen_hhi_breaks(hhi_df, cutoff_year=2022, threshold=0.0)

    share_joined = (
        share_2020
        .select("product_code", "product_name", pl.col("slope_change").alias("share_slope_chg_2020"))
        .join(
            share_2022.select("product_code", pl.col("slope_change").alias("share_slope_chg_2022")),
            on="product_code",
            how="inner",
        )
        .with_columns(
            (pl.col("share_slope_chg_2022").abs() > pl.col("share_slope_chg_2020").abs())
            .alias("share_stronger_2022")
        )
    )

    hhi_joined = (
        hhi_2020
        .select("product_code", pl.col("slope_change").alias("hhi_slope_chg_2020"))
        .join(
            hhi_2022.select("product_code", pl.col("slope_change").alias("hhi_slope_chg_2022")),
            on="product_code",
            how="inner",
        )
        .with_columns(
            (pl.col("hhi_slope_chg_2022").abs() > pl.col("hhi_slope_chg_2020").abs())
            .alias("hhi_stronger_2022")
        )
    )

    return (
        share_joined
        .join(hhi_joined, on="product_code", how="inner")
        .sort("product_code")
    )
