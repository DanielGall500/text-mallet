import pandas as pd
from plotnine import (
    aes,
    annotate,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_density,
    geom_vline,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
)


class ShannonVisualiser:
    """
    A class for visualising mutual information distributions.

    Args:
        data: Dictionary containing mutual information data or pandas DataFrame
        style: Plot style theme (default: 'minimal')
    """

    def __init__(self, data: pd.DataFrame | None = None, style: str = "minimal"):
        self.data = data
        self.style = style

    def prepare_data(
        self,
        mutual_info: list[list[float]],
        labels: list[str] | None = None,
        flatten: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare mutual information data for plotting.

        Args:
            mutual_info: List of mutual information scores (one per text/sample)
            labels: Optional labels for each text/sample
            flatten: Whether to flatten all surprisals into single distribution

        Returns:
            DataFrame ready for plotting
        """
        if flatten:
            flat_mutual_info = [s for sublist in mutual_info for s in sublist]
            df = pd.DataFrame({"mutual_info": flat_mutual_info})

            if labels:
                text_labels = []
                for i, sublist in enumerate(mutual_info):
                    label = labels[i] if i < len(labels) else f"Text {i + 1}"
                    text_labels.extend([label] * len(sublist))
                df["group"] = text_labels
        else:
            records = []
            for i, surp_list in enumerate(mutual_info):
                label = labels[i] if labels and i < len(labels) else f"Text {i + 1}"
                for s in surp_list:
                    records.append({"mutual_info": s, "group": label})
            df = pd.DataFrame(records)

        self.data = df
        return df

    def plot_density(
        self,
        fill: str = "#4A90D9",
        alpha: float = 0.35,
        title: str | None = None,
        subtitle: str | None = None,
        xlabel: str = "Mutual Information (bits)",
        ylabel: str = "Density",
        show_median: bool = False,
        show_mean: bool = False,
        show_stats_box: bool = True,
        color_scheme: str = "blue",  # "blue", "teal", "amber", "slate"
    ):
        if self.data is None:
            raise ValueError("No data available. Use prepare_data() first.")

        # ── Colour presets ──────────────────────────────────────────────────────
        palettes = {
            "blue": {
                "fill": "#4A90D9",
                "median": "#E74C3C",
                "mean": "#F39C12",
                "edge": "#2C5F8A",
            },
            "teal": {
                "fill": "#1ABC9C",
                "median": "#E74C3C",
                "mean": "#F39C12",
                "edge": "#0E7A63",
            },
            "amber": {
                "fill": "#E67E22",
                "median": "#2980B9",
                "mean": "#8E44AD",
                "edge": "#A04000",
            },
            "slate": {
                "fill": "#7F8C8D",
                "median": "#E74C3C",
                "mean": "#F39C12",
                "edge": "#4D5656",
            },
        }
        pal = palettes.get(color_scheme, palettes["blue"])
        fill = pal["fill"]

        vals = self.data["mutual_info"]
        median_val = vals.median()
        mean_val = vals.mean()
        std_val = vals.std()
        n = len(vals)

        if subtitle is None:
            pass

        plot = (
            ggplot(self.data, aes(x="mutual_info"))
            + geom_density(
                fill=fill,
                color=pal["edge"],
                alpha=alpha,
                size=0.8,
                bounds=(self.data["mutual_info"].min() - 0.5, float("inf")),
            )
            + labs(title=title, x=xlabel, y=ylabel, caption=subtitle)
            + scale_x_continuous(
                limits=(vals.min() - 1, vals.max() + 1),
                expand=(0, 0),
            )
            + scale_y_continuous(expand=(0.01, 0))
            + theme_minimal(base_size=13, base_family="DejaVu Sans")
            + theme(
                plot_caption=element_text(
                    size=10, color="#6B7280", ha="left", margin={"t": 6}
                ),
                axis_title=element_text(size=12, color="#374151"),
                axis_text=element_text(size=10, color="#6B7280"),
                axis_line=element_line(color="#D1D5DB", size=0.6),
                axis_ticks=element_line(color="#D1D5DB", size=0.5),
                panel_grid_major_y=element_line(color="#F3F4F6", size=0.5),
                panel_grid_major_x=element_blank(),
                panel_grid_minor=element_blank(),
                panel_background=element_rect(fill="white"),
                plot_background=element_rect(fill="white"),
                panel_border=element_blank(),
            )
        )

        if show_median:
            plot += geom_vline(
                xintercept=median_val,
                linetype="dashed",
                color=pal["median"],
                size=0.9,
                alpha=0.85,
            )
            plot += annotate(
                "text",
                x=median_val + 0.15,
                y=0.001,
                label=f"Median\n{median_val:.2f}",
                color=pal["median"],
                size=8,
                ha="left",
                va="bottom",
                fontweight="semibold",
            )

        if show_mean:
            plot += geom_vline(
                xintercept=mean_val,
                linetype="dotted",
                color=pal["mean"],
                size=0.9,
                alpha=0.85,
            )
            plot += annotate(
                "text",
                x=mean_val + 0.15,
                y=0.001,
                label=f"Mean\n{mean_val:.2f}",
                color=pal["mean"],
                size=8,
                ha="left",
                va="bottom",
                fontweight="semibold",
            )

        if show_stats_box:
            x_max = vals.max() + 1
            stats_text = (
                f"n = {n:,}\n"
                f"μ = {mean_val:.2f}\n"
                f"Md = {median_val:.2f}\n"
                f"σ = {std_val:.2f}"
            )
            plot += annotate(
                "label",
                x=x_max - 0.3,
                y=0.085,
                label=stats_text,
                size=8,
                ha="right",
                va="top",
                color="#1F2937",
                fill="white",
                label_padding=0.4,
                label_size=0.3,
                label_r=0.15,
                alpha=0.9,
            )

        return plot
