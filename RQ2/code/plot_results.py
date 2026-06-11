"""
Plot Init-md and Rfn-md across datasets, filling the area between the two
lines based on which method has the smaller std at each dataset.
Reads results/0/aggregates/aggregated.csv produced by aggregate_results_0.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_md_with_error_bars(
    csv_path: str = "results/0/aggregates/aggregated.csv",
    output_path: str | None = None,
    sort_by: str | None = "Init-md",
    font_family: str = "Times New Roman",
):
    """
    Create a line plot with two lines (Init-md and Rfn-md) across datasets.
    Between the two lines, fill each per-dataset segment with light red when
    Init-sd is the smaller std, and light green when Rfn-sd is the smaller std.

    Args:
        csv_path: Path to aggregated CSV file.
        output_path: Where to save the figure. Defaults to alongside the CSV.
        sort_by: Column to sort datasets by for readability. Set to None to
            preserve the file order.
        font_family: Font family used for every text element in the chart.
            Examples on macOS: "Helvetica", "Arial", "Avenir Next",
            "Times New Roman", "Georgia", "Courier New", "Menlo".
    """
    plt.rcParams["font.family"] = font_family

    csv_file = Path(csv_path)
    df = pd.read_csv(csv_file)

    numeric_cols = ["Init-sd", "Init-md", "Rfn-sd", "Rfn-md"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    df = df[df["dataset"] != "javagc"].reset_index(drop=True)
    df = df[df["dataset"] != "nasa93dem"].reset_index(drop=True)

    df["dataset"] = df["dataset"].astype(str).str[:10] + "..."

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(by=sort_by).reset_index(drop=True)

    x = list(range(len(df)))

    fig, ax = plt.subplots(figsize=(max(14, len(df) * 0.18), 7))

    for xi, init_sd, rfn_sd in zip(x, df["Init-sd"], df["Rfn-sd"]):
        if init_sd < rfn_sd:
            band_color = "lightcoral"
        elif rfn_sd < init_sd:
            band_color = "lightgreen"
        else:
            continue
        ax.axvspan(xi - 0.5, xi + 0.5, color=band_color, alpha=0.4, linewidth=0)

    ax.plot(
        x,
        df["Init-md"],
        label="Initial Settings Median Win Score",
        marker="o",
        markersize=4,
        linewidth=1.2,
        alpha=0.9,
        color="tab:blue",
    )
    ax.plot(
        x,
        df["Rfn-md"],
        label="Refined Settings Median Win Score",
        marker="s",
        markersize=4,
        linewidth=1.2,
        alpha=0.9,
        color="tab:orange",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=90, fontsize=16)
    ax.set_yticklabels([0,20,40,60,80,100], fontsize=16)
    ax.set_ylabel("Median of  Win()  scores over 20 repeats", fontsize=18)
    ax.set_ybound(0, 100)
    ax.set_xlim(-0.5, len(df) - 0.5)
    # title_suffix = f" (sorted by {sort_by})" if sort_by else ""
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Make the legend bigger

    # Make the lines thicker
    for line in ax.get_lines():
        line.set_linewidth(4)

    line_handles, line_labels = ax.get_legend_handles_labels()

    fill_handles = [
        mpatches.Patch(color="lightcoral", alpha=0.5, label="Initial settings More Stable"),
        mpatches.Patch(color="lightgreen", alpha=0.5, label="Refined settings More Stable"),
    ]
    ax.legend(handles=line_handles + fill_handles, loc="center right", fontsize=20)
    fig.tight_layout()

    if output_path is None:
        output_path = csv_file.with_name("md_with_area.png")
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    plot_md_with_error_bars()
