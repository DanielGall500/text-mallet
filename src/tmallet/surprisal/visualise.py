from typing import List, Dict, Optional
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_density,
    labs,
    theme_minimal,
    scale_fill_brewer,
    geom_vline,
    scale_x_continuous,
    annotate,
)


class SurprisalVisualiser:
    """
    A class for visualizing surprisal distributions using plotnine.

    Args:
        data: Dictionary containing surprisal data or pandas DataFrame
        style: Plot style theme (default: 'minimal')

    Example:
        >>> plotter = SurprisalPlotter(surprisal_data)
        >>> plot = plotter.plot_histogram()
        >>> plot.save('surprisal_dist.png')
    """

    def __init__(self, data: Optional[pd.DataFrame] = None, style: str = "minimal"):
        self.data = data
        self.style = style

    def prepare_data(
        self,
        surprisals: List[List[float]],
        labels: Optional[List[str]] = None,
        flatten: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare surprisal data for plotting.

        Args:
            surprisals: List of surprisal lists (one per text/sample)
            labels: Optional labels for each text/sample
            flatten: Whether to flatten all surprisals into single distribution

        Returns:
            DataFrame ready for plotting
        """
        if flatten:
            flat_surprisals = [s for sublist in surprisals for s in sublist]
            df = pd.DataFrame({"surprisal": flat_surprisals})

            if labels:
                text_labels = []
                for i, sublist in enumerate(surprisals):
                    label = labels[i] if i < len(labels) else f"Text {i + 1}"
                    text_labels.extend([label] * len(sublist))
                df["group"] = text_labels
        else:
            records = []
            for i, surp_list in enumerate(surprisals):
                label = labels[i] if labels and i < len(labels) else f"Text {i + 1}"
                for s in surp_list:
                    records.append({"surprisal": s, "group": label})
            df = pd.DataFrame(records)

        self.data = df
        return df

    def plot_density(
        self,
        fill: str = "#2ecc71",
        alpha: float = 0.5,
        title: str = "Surprisal Density",
        xlabel: str = "Surprisal",
        ylabel: str = "Density",
        show_median: bool = False,
    ):
        """
        Create a density plot of surprisal values.

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
            ggplot(self.data, aes(x="surprisal"))
            + geom_density(fill=fill, alpha=alpha)
            + labs(title=title, x=xlabel, y=ylabel)
            + scale_x_continuous(limits=(-5, 15))
            + theme_minimal()
        )

        if show_median:
            median_val = self.data["surprisal"].median()
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

    def plot_grouped_density(
        self,
        alpha: float = 0.5,
        title: str = "Surprisal Density by Group",
        xlabel: str = "Surprisal",
        ylabel: str = "Density",
    ):
        """
        Create overlaid density plots by group.

        Args:
            alpha: Transparency level
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label

        Returns:
            plotnine plot object
        """
        if self.data is None or "group" not in self.data.columns:
            raise ValueError("No grouped data available.")

        plot = (
            ggplot(self.data, aes(x="surprisal", fill="group"))
            + geom_density(alpha=alpha)
            + labs(title=title, x=xlabel, y=ylabel)
            + theme_minimal()
            + scale_fill_brewer(type="qual", palette="Set2")
        )

        return plot
