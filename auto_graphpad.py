"""Auto settings for GraphPad style in matplotlib and seaborn."""

from itertools import cycle
from typing import Any, Sequence

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.axes import Axes

CUSTOM_PALETTE_SNSSTYLE = cycle(
    (
        "#BCE4F9",
        "#7BD3F2",
        "#F0AECB",
        "#E88791",
        "#C4DDB1",
        "#97D9A8",
        "#CCCCE6",
        "#9D9DCF",
        "#8FB0DD",
        "#F0CEA4",
        "#F8A450",
    )
)


def add_custom_fonts(*args) -> None:
    """Add custom fonts to the font manager."""
    for font in args:
        fm.fontManager.addfont(font)


def auto_style(
    rc_mplstyle: dict[str, Any] | None = None,
    fname_mplstyle: str | None = "./GraphPadPrism.mplstyle",
    palette_snsstyle: str | Sequence[str] | None = "bright",
    n_colors: int | None = None,
) -> None:
    """Initialize the matplotlib and seaborn styles.

    Args
    ----
    rc_mplstyle : dict[str, Any] | None
        The matplotlib style for rcParams.
    fname_mplstyle : str | None
        The matplotlib style for figure.
    palette_snsstyle : str | Sequence[str] | None
        The seaborn palette style.
    n_colors : int | None
        The number of colors to generate.

    Returns
    -------
    None

    Notes
    -----
    Please put this function on the top of the script to enable global settings.
    """

    if palette_snsstyle is not None:
        sns.set_palette(palette_snsstyle, n_colors=n_colors)
    if fname_mplstyle is not None:
        plt.style.use(fname_mplstyle)
    if rc_mplstyle is not None:
        plt.style.use(rc_mplstyle)


def auto_ticks(
    ax: Axes,
    *,
    left: float | None = None,
    right: float | None = None,
    bottom: float | None = None,
    top: float | None = None,
) -> None:
    """Set the major and minor ticks of an axis automatically.

    Args
    ----
    ax : Axes
        The axis to be set.
    left : float | None, optional
        The left limit of the x-axis, by default None.
    right : float | None, optional
        The right limit of the x-axis, by default None.
    bottom : float | None, optional
        The bottom limit of the y-axis, by default None.
    top : float | None, optional
        The top limit of the y-axis, by default None.

    Notes
    -----
    Please put this function after the data are passed to the axis.
    """

    # Set the major and minor ticks of the x-axis
    if left is not None:
        ax.set_xlim(left=left)
    if right is not None:
        ax.set_xlim(right=right)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, min_n_ticks=3))
    x_tick_num = len(ax.get_xticks())
    if x_tick_num >= 6:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    elif x_tick_num >= 5:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    else:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    # For viewing incomplete ticks
    if left is None and (ax.get_xticks()[0] != ax.get_xlim()[0]):
        ax.set_xlim(left=ax.get_xticks()[0])
    if right is None and (ax.get_xticks()[-1] != ax.get_xlim()[-1]):
        ax.set_xlim(right=ax.get_xticks()[-1])

    # Set the major and minor ticks of the y-axis
    if bottom is not None:
        ax.set_ylim(bottom=bottom)
    if top is not None:
        ax.set_ylim(top=top)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, min_n_ticks=3))
    y_tick_num = len(ax.get_yticks())
    if y_tick_num >= 6:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    elif y_tick_num >= 5:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    else:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    # For viewing incomplete ticks
    if bottom is None and (ax.get_yticks()[0] != ax.get_ylim()[0]):
        ax.set_ylim(bottom=ax.get_yticks()[0])
    if top is None and (ax.get_yticks()[-1] != ax.get_ylim()[-1]):
        ax.set_ylim(top=ax.get_yticks()[-1])


if __name__ == "__main__":
    auto_style()
