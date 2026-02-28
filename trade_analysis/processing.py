import numpy as np
import polars as pl


def compute_shares(
    df: pl.DataFrame,
    aggregate_code: str = "EXT_EU27_2020",
) -> pl.DataFrame:
    denominator_df = (
        df
        .filter(
            pl.col("partner_code") == aggregate_code
        )
        .select(
            pl.col("time_period"),
            pl.col("product_code"),
            pl.col("value").alias("ext_eu27_total"),
        )
    )

    return (
        df
        .remove(
            pl.col("partner_code") == aggregate_code,
            pl.col("product_code").str == "TOTAL",
        )
        .join(
            denominator_df,
            on=["time_period", "product_code"],
            how="left",
        )
        .with_columns(
            (pl.col("value") / pl.col("ext_eu27_total")).alias("share"),
        )
        .drop(
            pl.col("ext_eu27_total")
        )
        .with_columns(
            [
                # Year-over-year growth ratio
                (pl.col("value") /
                 pl.col("value").shift(1).over("partner_code", "product_code"))
                .alias("yoy_ratio"),

                # Year-over-year percentage change
                ((pl.col("value") - pl.col("value").shift(1).over("partner_code", "product_code")) /
                 pl.col("value").shift(1).over("partner_code", "product_code") * 100)
                .alias("yoy_change_percent"),

                # 3-year centered moving average
                pl.col("value")
                .rolling_mean(window_size=3, center=True)
                .over("partner_code", "product_code")
                .alias("ma_3y"),

                ((pl.col("share") >= 0.01).alias("is_significant")),
            ]
        )
        .with_columns(
            ((pl.col("is_significant").shift(1).over("partner_code", "product_code")).alias("was_significant")),
        )
    )

def compute_product_weights(
    shares_df: pl.DataFrame,
    baseline_end: int = 2019,
) -> pl.DataFrame:
    baseline = shares_df.filter(
        (pl.col("time_period") <= baseline_end)
        & (pl.col("product_code").cast(pl.Utf8).str.len_chars() > 2)
    )

    totals = (
        baseline
        .group_by("product_code")
        .agg(
            pl.col("value").sum().alias("total_value"),
            pl.col("product_name").first().alias("product_name"),
        )
    )

    grand_total = totals["total_value"].sum()

    return (
        totals
        .with_columns(
            (pl.col("total_value") / grand_total * 100).alias("weight_pct"),
        )
        .select("product_code", "product_name", "total_value", "weight_pct")
        .sort("weight_pct", descending=True)
    )


def compute_hhi(
    df: pl.DataFrame,
) -> pl.DataFrame:
    return (
        df
        .group_by(["time_period", "product_code"])
        .agg(
            (
                (pl.col("share") ** 2).sum().alias("hhi") * 10_000,
            ))
    )
