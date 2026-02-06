from typing import List, Optional
import matplotlib.cm as cm
from IPython.display import HTML
import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_density,
    labs,
    theme_minimal,
    geom_vline,
    scale_x_continuous,
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
                print(surp_list)
                for s in surp_list:
                    records.append({"mutual_info": s, "group": label})
            df = pd.DataFrame(records)

        self.data = df
        return df

    def plot_density(
        self,
        fill: str = "#2ecc71",
        alpha: float = 0.5,
        title: str = "Distribution of Word-Level Mutual Information",
        xlabel: str = "Mutual Information",
        ylabel: str = "Density",
        show_median: bool = False,
    ):
        """
        Create a density plot of mutual information values.

        Args:
            fill: Fill color
            alpha: Transparency level
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label

        Returns:
            plotnine plot object
        """
        if self.data is None:
            raise ValueError("No data available. Use prepare_data() first.")

        plot = (
            ggplot(self.data, aes(x="mutual_info"))
            + geom_density(fill=fill, alpha=alpha)
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

    def display_sentence_heatmap(self, words, mutual_info, colormap="Reds"):
        # Handle both single sentences and multiple sentences
        if isinstance(words[0], list):
            # Multiple sentences: words and mutual_info are lists of lists
            sentences_html = []
            for sentence_words, sentence_mutual_info in zip(words, mutual_info):
                sentence_html = self._generate_sentence_html(sentence_words, sentence_mutual_info, colormap)
                sentences_html.append(sentence_html)
            html_str = "<div style='margin: 15px 0;'>" + "".join(sentences_html) + "</div>"
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
