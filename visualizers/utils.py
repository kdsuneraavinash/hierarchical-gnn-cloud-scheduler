import io

from matplotlib import axes
import matplotlib.pyplot as plt
import pygraphviz as pgv


def get_color(color_id: int) -> str:
    color_map = ["#FADFA1", "#7EACB5", "#E6B9A6", "#939185", "#FFF078", "#939185"]
    return color_map[color_id % len(color_map)]


def draw_agraph(ax: axes.Axes, a: pgv.AGraph) -> None:
    """
    Draw the provided AGraph on the provided Axes.
    """

    a.layout(prog="dot")
    buffer = io.BytesIO()
    d_bytes = a.draw(format="png")
    assert d_bytes is not None
    buffer.write(d_bytes)
    buffer.seek(0)
    ax.imshow(plt.imread(buffer))
    ax.axis("off")
