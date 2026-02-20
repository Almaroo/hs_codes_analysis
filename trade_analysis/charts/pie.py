import polars as pl
import matplotlib.pyplot as plt


def plot_pie(
    df: pl.DataFrame,
    product_code: str,
    year: int,
    significance_threshold: float = 0.01,
) -> None:
    """Pie chart of partner shares for a single product/year.

    Partners whose share is below *significance_threshold* are grouped
    into a single "Rest" slice.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`trade_analysis.processing.compute_shares`.
    product_code:
        HS product code to chart.
    year:
        Calendar year to display.
    significance_threshold:
        Minimum share to show as an individual slice (default 1 %).
    """
    data = (
        df
        .filter(
            (pl.col("product_code") == product_code)
            & (pl.col("time_period") == year)
        )
        .sort(pl.col("share"), descending=True)
    )

    if len(data) == 0:
        print(f"No data for product {product_code}, year {year}")
        return

    significant = data.filter(pl.col("share") >= significance_threshold)
    non_significant = data.filter(pl.col("share") < significance_threshold)

    if len(significant) == 0:
        return

    partners = significant["partner_code"].to_list()
    shares = (significant["share"] * 100).to_list()

    if len(non_significant) > 0:
        rest_share = non_significant["share"].sum() * 100
        partners.append("Rest")
        shares.append(rest_share)

    product_name = data["product_name"][0]

    plt.figure(figsize=(10, 8))
    plt.pie(shares, labels=partners, autopct="%1.1f%%", startangle=90)
    plt.title(f"{product_name} ({product_code}) - {year}")
    plt.tight_layout()
    plt.show()
