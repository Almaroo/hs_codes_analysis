from typing import Optional

import polars as pl
import matplotlib.pyplot as plt


def plot_bar(
    df: pl.DataFrame,
    product_code: int,
    year: int,
    significance_threshold: float = 0.01,
    hhi_df: Optional[pl.DataFrame] = None,
    print_data: bool = False,
) -> None:
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
    values = significant["value"].to_list()

    if len(non_significant) > 0:
        rest_value = non_significant["value"].sum()
        partners.append("Rest")
        values.append(rest_value)

    product_name = data["product_name"][0]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [plt.cm.tab10(i % 10) for i in range(len(partners))]
    bars = ax.bar(partners, values, color=colors)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Partner")
    ax.set_ylabel("Value")

    title = f"{product_name} ({product_code}) - {year}"
    if hhi_df is not None:
        hhi_row = hhi_df.filter(
            (pl.col("product_code") == product_code)
            & (pl.col("time_period") == year)
        )
        if len(hhi_row) > 0:
            hhi_val = hhi_row["hhi"][0]
            title += f"\nHHI = {hhi_val:.4f}"

    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    if print_data:
        print(title)
        print(f"Partner\tValue")
        for p, v in zip(partners, values):
            print(f"{p}\t{v}")
