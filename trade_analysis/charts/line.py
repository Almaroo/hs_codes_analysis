import polars as pl
import matplotlib.pyplot as plt


def plot_share_over_time(
    df: pl.DataFrame,
    product_code: str,
    partner_code: str = "CN",
) -> None:
    """Line chart of a partner's import share over time.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`trade_analysis.processing.compute_shares`.
    product_code:
        HS product code to chart.
    partner_code:
        Partner country code (default ``"CN"`` for China).
    """
    partner_data = (
        df
        .filter(
            (pl.col("product_code") == product_code)
            & (pl.col("partner_code") == partner_code)
        )
        .sort(pl.col("time_period"))
    )

    if len(partner_data) == 0:
        print(f"No data for product {product_code}, partner {partner_code}")
        return

    years = partner_data["time_period"].to_list()
    shares = (partner_data["share"] * 100).to_list()
    product_name = partner_data["product_name"][0]
    partner_name = partner_data["partner_name"][0]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Year")
    ax.set_ylabel("Share (%)", color="tab:blue")
    ax.plot(years, shares, color="tab:blue", marker="o")
    plt.title(f"{partner_name}'s Share - {product_name} ({product_code})")
    plt.grid(True, alpha=0.3)
    plt.xticks(years, rotation=45)
    plt.tight_layout()
    plt.show()
