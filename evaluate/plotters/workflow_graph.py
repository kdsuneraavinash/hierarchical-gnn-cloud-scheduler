import networkx as nx
from matplotlib import pyplot as plt

from dataset.models import Solution
from evaluate.plotters.color import draw_agraph, get_color


def plot_workflow_graphs(ax: plt.Axes, solution: Solution):
    """
    Plot the workflows on the provided DiGraph.
    """

    g_w: nx.DiGraph = nx.DiGraph()
    for task in solution.dataset.tasks:
        node_label = f"T{task.id}\n{task.length} MI\n{task.req_memory_mb // 1024} GB"
        node_color = get_color(task.workflow_id)
        g_w.add_node(str(task.id), label=node_label, fillcolor=node_color, style="filled", fontname="Arial")
        for child_id in task.child_ids:
            g_w.add_edge(str(task.id), str(child_id), color="black")

    a_w = nx.nx_agraph.to_agraph(g_w)
    draw_agraph(ax, a_w)
