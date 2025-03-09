import io

import matplotlib.pyplot as plt
import pygraphviz as pgv


def get_color(color_id: int) -> str:
    color_map = ["#FADFA1", "#7EACB5", "#E6B9A6", "#939185", "#FFF078", "#939185"]
    return color_map[color_id % len(color_map)]


def draw_agraph(ax: plt.Axes, a: pgv.AGraph) -> None:
    """
    Draw the provided AGraph on the provided Axes.
    """

    a.layout(prog="dot")
    buffer = io.BytesIO()
    buffer.write(a.draw(format="png"))
    buffer.seek(0)
    ax.imshow(plt.imread(buffer))
    ax.axis("off")
