from pathlib import Path

import polars as pl


def load_trade_csv_v1(path: Path) -> pl.DataFrame:
    """Read a v1 trade CSV and return a clean DataFrame.

    Splits composite ``partner`` / ``product`` columns into
    ``_code`` / ``_name`` pairs, renames ``TIME_PERIOD`` →
    ``time_period`` and ``OBS_VALUE`` → ``value``, and drops TOTAL
    product rows.
    """
    raw = pl.read_csv(path)

    return (
        raw
        .select(
            pl.col("partner"),
            pl.col("product"),
            pl.col("TIME_PERIOD").alias("time_period"),
            pl.col("OBS_VALUE").alias("value"),
        )
        .with_columns(
            pl.col("partner").str.split(":").list.get(0).alias("partner_code"),
            pl.col("partner").str.split(":").list.get(1).alias("partner_name"),
            pl.col("product").str.split(":").list.get(0).alias("product_code"),
            pl.col("product").str.split(":").list.get(1).alias("product_name"),
        )
        .drop(
            pl.col("partner"),
            pl.col("product"),
        )
        .remove(
            (pl.col("product_code") == "TOTAL"),
        )
    )


def load_trade_csv_v2(path: Path) -> pl.DataFrame:
    """Read a v2 trade CSV and return a clean DataFrame.

    The v2 format has separate code/name columns and duplicate header
    names (e.g. two ``TIME_PERIOD`` columns), so the file is read
    without headers and mapped by position.

    Produces the same output schema as :func:`load_trade_csv_v1`.
    """
    # Positional mapping from v2 header:
    #  0  STRUCTURE        7  partner   14 INDICATORS
    #  1  STRUCTURE_ID     8  PARTNER   15 TIME_PERIOD (value)
    #  2  STRUCTURE_NAME   9  product   16 TIME_PERIOD (label, empty)
    #  3  freq            10  PRODUCT   17 OBS_VALUE
    #  4  Frequency       11  flow      18 Observation Value (empty)
    #  5  reporter        12  FLOW
    #  6  REPORTER        13  indicators
    raw = pl.read_csv(path, has_header=False, skip_rows=1)
    col = raw.columns

    return (
        raw
        .select(
            pl.col(col[7]).alias("partner_code"),
            pl.col(col[8]).alias("partner_name"),
            pl.col(col[9]).alias("product_code"),
            pl.col(col[10]).alias("product_name"),
            pl.col(col[15]).cast(pl.Int64).alias("time_period"),
            pl.col(col[17]).cast(pl.Float64).alias("value"),
        )
        .remove(
            (pl.col("product_code").str == "TOTAL"),
        )
    )
