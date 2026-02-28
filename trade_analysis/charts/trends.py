import numpy as np
import polars as pl
import matplotlib.pyplot as plt


def plot_segmented_trend(
    years: list,
    values: list,
    cutoff_year: int,
    title: str = "",
    ylabel: str = "",
) -> None:
    before_y = [y for y in years if y < cutoff_year]
    before_v = [v for y, v in zip(years, values) if y < cutoff_year]
    after_y = [y for y in years if y >= cutoff_year]
    after_v = [v for y, v in zip(years, values) if y >= cutoff_year]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(years, values, "o-", color="grey", alpha=0.5, label="Observed")

    if len(before_y) >= 2:
        coeffs = np.polyfit(before_y, before_v, 1)
        fit_y = np.array(before_y)
        fit_v = np.polyval(coeffs, fit_y)
        ax.plot(fit_y, fit_v, "-", color="tab:blue", linewidth=2, label=f"Pre-{cutoff_year} slope: {coeffs[0]:+.2f}/yr")

    if len(after_y) >= 2:
        coeffs = np.polyfit(after_y, after_v, 1)
        fit_y = np.array(after_y)
        fit_v = np.polyval(coeffs, fit_y)
        ax.plot(fit_y, fit_v, "-", color="tab:red", linewidth=2, label=f"Post-{cutoff_year} slope: {coeffs[0]:+.2f}/yr")

    ax.axvline(x=cutoff_year, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_hypothesis_summary(
    summary_df: pl.DataFrame,
    metric_label: str = "Slope change",
    title: str = "",
    top_n: int = 20,
    threshold: float | None = None,
) -> None:
    df = (
        summary_df
        .sort(pl.col("slope_change").abs(), descending=True)
        .head(top_n)
        .sort("slope_change")
    )

    labels = df["product_code"].cast(pl.Utf8).to_list()
    if "product_name" in df.columns:
        names = df["product_name"].to_list()
        labels = [f"{c} – {n[:30]}" for c, n in zip(labels, names)]

    values = df["slope_change"].to_list()
    colors = ["tab:blue" if v <= 0 else "tab:red" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.35)))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel(metric_label)
    ax.set_title(title)
    ax.axvline(x=0, color="black", linewidth=0.8)

    if threshold is not None:
        ax.axvline(x=threshold, color="grey", linestyle="--", alpha=0.5, label=f"Threshold ±{threshold}")
        ax.axvline(x=-threshold, color="grey", linestyle="--", alpha=0.5)
        ax.legend()

    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
