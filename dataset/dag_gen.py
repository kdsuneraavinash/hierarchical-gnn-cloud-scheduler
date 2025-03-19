from abc import abstractmethod
from collections import defaultdict

import numpy as np

from dataset.utils import random_list

MIN_DAG_SIZE = 25


# Base DAG
# ----------------------------------------------------------------------------------------------------------------------


class BaseDagGen:
    possibilities: dict[int, list[tuple[int, ...]]]

    def __init__(self):
        self.possibilities = {}

    def generate(self, n: int, rng: np.random.RandomState) -> dict[int, set[int]]:
        if len(self.possibilities) == 0:
            self.possibilities = self._possibilities(100)
        if n not in self.possibilities:
            raise ValueError(f"{n=} is not supported by this dag gen")
        curr_possibilities = self.possibilities[n]
        curr_possibility = curr_possibilities[rng.randint(0, len(curr_possibilities))]
        graph = {i: set() for i in range(n)}
        self._generate(graph, rng, *curr_possibility)
        return graph

    @abstractmethod
    def _possibilities(self, max_range: int) -> dict[int, list[tuple[int, ...]]]:
        raise NotImplementedError()

    @abstractmethod
    def _generate(self, graph: dict[int, set[int]], rng: np.random.RandomState, *args):
        raise NotImplementedError()


# Epigenomics DAG
# ----------------------------------------------------------------------------------------------------------------------


class EpigenomicsDagGen(BaseDagGen):
    def _possibilities(self, max_range: int) -> dict[int, list[tuple[int, ...]]]:
        possibilities: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        for len_a in range(1, max_range):
            for sum_a in range(len_a, max_range):
                dag_size = 3 + 4 * sum_a + 2 * len_a
                if dag_size > max_range:
                    continue
                possibilities[dag_size].append((len_a, sum_a, 0))
                possibilities[dag_size + 1].append((len_a, sum_a, 1))
        return possibilities

    def _generate(self, graph: dict[int, set[int]], rng: np.random.RandomState, *args):
        len_a, sum_a, pad = args
        a = random_list(len_a, sum_a, rng)

        combiner = sum(a) * 4 + 2 * len(a)
        for i, ai in enumerate(a):
            start = sum(a[:i]) * 4 + 2 * i
            end = start + 4 * a[i] + 2 - 1
            for j in range(ai):
                graph[start].add(start + j * 4 + 1)
                graph[start + j * 4 + 1].add(start + j * 4 + 2)
                graph[start + j * 4 + 2].add(start + j * 4 + 3)
                graph[start + j * 4 + 3].add(start + j * 4 + 4)
                graph[start + j * 4 + 4].add(end)
            graph[end].add(combiner)
        graph[combiner].add(combiner + 1)
        graph[combiner + 1].add(combiner + 2)

        if pad > 0:
            graph[combiner + 2].add(combiner + 3)


# Inspiral DAG
# ----------------------------------------------------------------------------------------------------------------------


class InspiralDagGen(BaseDagGen):
    def _possibilities(self, max_range: int) -> dict[int, list[tuple[int, ...]]]:
        possibilities: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        for a in range(1, max_range):
            for len_b in range(1, max_range):
                for sum_b in range(len_b, max_range):
                    dag_size = 2 * (a + len_b + sum_b) + 1
                    if dag_size > max_range:
                        continue
                    possibilities[dag_size].append((a, len_b, sum_b, 0))
                    possibilities[dag_size + 1].append((a, len_b, sum_b, 1))
        return possibilities

    def _generate(self, graph: dict[int, set[int]], rng: np.random.RandomState, *args):
        a, len_b, sum_b, pad = args
        b = random_list(len_b, sum_b, rng)

        for i in range(a):
            graph[i].add(i + a)

        global_end = 2 * a + 2 * sum(b) + 2 * len(b)
        connector_range_min = 1
        for i, b_i in enumerate(b):
            connector_range = np.random.randint(connector_range_min, a + 1)
            connector_end = 2 * a + i
            if len(b) == 1:
                connector_range_start = a
                connector_range_end = 2 * a
            elif i == 0:
                connector_range_start = a
                connector_range_end = connector_range_start + connector_range
                connector_range_min = 2 * a - connector_range_end + 1
            else:
                connector_range_end = 2 * a
                connector_range_start = connector_range_end - connector_range
            for connector in range(connector_range_start, connector_range_end):
                graph[connector].add(connector_end)

            start = 2 * a + 2 * sum(b[:i]) + len(b) + i
            end = start + 2 * b[i]
            for j in range(b_i):
                graph[connector_end].add(start + j)
                graph[start + j].add(start + b_i + j)
                graph[start + b_i + j].add(end)
            graph[end].add(global_end)

        if pad > 0:
            graph[global_end].add(global_end + 1)


if __name__ == "__main__":
    import networkx
    from matplotlib import pyplot as plt
    from visualizers.utils import draw_agraph

    _, ax = plt.subplots(figsize=(16, 5))

    graph = InspiralDagGen().generate(32, np.random.RandomState())
    g = networkx.DiGraph()
    for i in sorted(graph.keys()):
        g.add_node(i)
    for a in list(graph.keys()):
        for b in list(graph[a]):
            g.add_edge(a, b)
    a = networkx.nx_agraph.to_agraph(g)
    draw_agraph(ax, a)
