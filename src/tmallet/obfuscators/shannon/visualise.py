from typing import List, Optional
import matplotlib.cm as cm
from IPython.display import HTML
import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_density,
    geom_vline,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme_minimal,
    theme,
    element_text,
    element_line,
    element_blank,
    element_rect,
    annotate,
)


class ShannonVisualiser:
    """
    A class for visualising mutual information distributions.

    Args:
        data: Dictionary containing mutual information data or pandas DataFrame
        style: Plot style theme (default: 'minimal')
    """

    def __init__(self, data: Optional[pd.DataFrame] = None, style: str = "minimal"):
        self.data = data
        self.style = style

    def prepare_data(
        self,
        mutual_info: List[List[float]],
        labels: Optional[List[str]] = None,
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
        title: str = None,
        subtitle: str = None,
        xlabel: str = "Mutual Information (bits)",
        ylabel: str = "Density",
        show_median: bool = False,
        show_mean: bool = False,
        show_stats_box: bool = True,
        color_scheme: str = "blue",  # "blue", "teal", "amber", "slate"
        # : tuple = (9, 5.5),
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

        # ── Derived statistics ───────────────────────────────────────────────────
        print("Data:")
        print("Size: ", len(self.data))
        print(self.data.head())
        vals = self.data["mutual_info"]
        median_val = vals.median()
        mean_val = vals.mean()
        std_val = vals.std()
        n = len(vals)

        if subtitle is None:
            pass
            # subtitle = f"n = {n:,}  ·  Mean = {mean_val:.2f}  ·  Std = {std_val:.2f}"

        # ── Base plot ────────────────────────────────────────────────────────────
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
                # ── Title & caption ──
                # plot_title=element_text(size=15, weight="bold", margin={"b": 4}),
                plot_caption=element_text(
                    size=10, color="#6B7280", ha="left", margin={"t": 6}
                ),
                # ── Axes ──
                axis_title=element_text(size=12, color="#374151"),
                axis_text=element_text(size=10, color="#6B7280"),
                axis_line=element_line(color="#D1D5DB", size=0.6),
                axis_ticks=element_line(color="#D1D5DB", size=0.5),
                # ── Grid ──
                panel_grid_major_y=element_line(color="#F3F4F6", size=0.5),
                panel_grid_major_x=element_blank(),
                panel_grid_minor=element_blank(),
                # ── Panel ──
                panel_background=element_rect(fill="white"),
                plot_background=element_rect(fill="white"),
                panel_border=element_blank(),
                # ── Margins ──
                # plot_margin={"t": 0.25, "r": 0.3, "b": 0.2, "l": 0.1},
                # plot_margin=(0.25, 0.3, 0.2, 0.1),
                # figure_size=figsize,
            )
        )

        # ── Median line ──────────────────────────────────────────────────────────
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

        # ── Mean line ────────────────────────────────────────────────────────────
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

        # ── Stats box (top-right corner) ─────────────────────────────────────────
        if show_stats_box:
            x_max = vals.max() + 1
            # Infer a safe y-position from the data range later; use a high fraction
            stats_text = (
                f"n = {n:,}\n"
                f"μ = {mean_val:.2f}\n"
                f"Md = {median_val:.2f}\n"
                f"σ = {std_val:.2f}"
            )
            # We annotate at a high y that the KDE won't reach at the tail
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

    """
    def plot_density(
        self,
        fill: str = "#2ecc71",
        alpha: float = 0.5,
        title: str = "Distribution of Word-Level Mutual Information",
        xlabel: str = "Mutual Information",
        ylabel: str = "Density",
        show_median: bool = False,
    ):
        if self.data is None:
            raise ValueError("No data available. Use prepare_data() first.")

        # bounds required so that it doesn't try to fill in negative space
        # as if any values are negative
        plot = (
            ggplot(self.data, aes(x="mutual_info"))
            + geom_density(fill=fill, alpha=alpha, bounds=(0, float('inf')))
            + labs(title=title, x=xlabel, y=ylabel)
            + scale_x_continuous(limits=(-5, 15))
            + theme_minimal()
        )

        if show_median:
            median_val = self.data["mutual_info"].median()
            plot += geom_vline(xintercept=median_val, linetype="dashed", color="red")
            plot += annotate(
                "text",
                x=median_val,
                y=0,  # vertical position (0 at baseline, adjust if needed)
                label=f"Median = {median_val:.2f}",
                color="red",
                va="bottom",  # vertical alignment
                ha="left",  # horizontal alignment
            )

        return plot
    """

    def display_sentence_heatmap(self, words, mutual_info, colormap="Reds"):
        # Handle both single sentences and multiple sentences
        if isinstance(words[0], list):
            # Multiple sentences: words and mutual_info are lists of lists
            sentences_html = []
            for sentence_words, sentence_mutual_info in zip(words, mutual_info):
                sentence_html = self._generate_sentence_html(
                    sentence_words, sentence_mutual_info, colormap
                )
                sentences_html.append(sentence_html)
            html_str = (
                "<div style='margin: 15px 0;'>" + "".join(sentences_html) + "</div>"
            )
        else:
            # Single sentence
            html_str = self._generate_sentence_html(words, mutual_info, colormap)

        return html_str

    def _generate_sentence_html(self, words, mutual_info, colormap):
        words = list(words)
        mutual_info = np.array(mutual_info)

        # Normalize mutual_info to 0-1
        norm = (mutual_info - mutual_info.min()) / (
            mutual_info.max() - mutual_info.min() + 1e-10
        )

        # Pick a colormap
        cmap = cm.get_cmap(colormap)

        # Generate HTML spans with background colors
        spans = []
        for w, v in zip(words, norm):
            r, g, b, a = cmap(v)
            color = f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a:.2f})"
            spans.append(
                f"<span style='background-color:{color};padding:2px 4px;margin:0 1px;border-radius:3px;transition:all 0.3s;' "
                f"onmouseover='this.style.transform=\"scale(1.1)\"' "
                f"onmouseout='this.style.transform=\"scale(1)\"'>{w}</span>"
            )

        # Wrap spans in a styled container
        sentence_html = f"""
        <div style="
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 18px;
            line-height: 2;
            padding: 15px;
            border-radius: 8px;
            background-color: #f9f9f9;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: fit-content;
            margin: 10px auto;
        ">
            {" ".join(spans)}
        </div>
        """
        return sentence_html
