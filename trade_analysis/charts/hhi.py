import textwrap

import polars as pl
import matplotlib.pyplot as plt


def plot_hhi_over_time(
    hhi_df: pl.DataFrame,
    product_code: int,
    product_name: str = "",
    print_data: bool = False,
) -> None:
    data = (
        hhi_df
        .filter(pl.col("product_code") == product_code)
        .sort(pl.col("time_period"))
    )

    if len(data) == 0:
        print(f"No HHI data for product {product_code}")
        return

    years = data["time_period"].to_list()
    hhi_values = data["hhi"].to_list()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(years, hhi_values, color="tab:red", marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("HHI")
    ax.set_ylim(0, None)

    label = product_name or product_code
    title = f"Market Concentration (HHI) - {label} ({product_code})"
    ax.set_title("\n".join(textwrap.wrap(title, width=60)))

    ax.axhline(y=2500, color="grey", linestyle="--", alpha=0.5, label="Highly concentrated (2500)")
    ax.axhline(y=1500, color="grey", linestyle=":", alpha=0.5, label="Moderately concentrated (1500)")
    ax.legend()

    plt.grid(True, alpha=0.3)
    plt.xticks(years, rotation=45)
    plt.tight_layout()
    plt.show()

    if print_data:
        label = product_name or product_code
        title = f"Market Concentration (HHI) - {label} ({product_code})"
        print(title)
        print(f"Year\tHHI")
        for y, h in zip(years, hhi_values):
            print(f"{y}\t{h:.4f}")
