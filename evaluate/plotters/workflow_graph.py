import networkx as nx
import pygraphviz as pgv

from dataset.models import Task
from evaluate.plotters.color import get_color


def plot_workflow_graphs(g: nx.DiGraph, tasks: list[Task]) -> pgv.AGraph:
    """
    Plot the workflows on the provided DiGraph.
    """

    for task in tasks:
        node_label = f"T{task.id}\n{task.length} MI\n{task.req_memory_mb // 1024} GB"
        node_color = get_color(task.workflow_id)
        g.add_node(str(task.id), label=node_label, fillcolor=node_color, style="filled", fontname="Arial")
        for child_id in task.child_ids:
            g.add_edge(str(task.id), str(child_id), color="black")

    return nx.nx_agraph.to_agraph(g)
