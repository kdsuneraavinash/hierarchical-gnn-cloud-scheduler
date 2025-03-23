from algorithms.base_mo import BaseMoScheduler, SolutionStoreType
from algorithms.pymoo_problem import PyMooOptimizationProblem, decode_pymoo_x
from dataset.models import Dataset
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions


class Nsga3Scheduler(BaseMoScheduler):
    def __init__(self, store: SolutionStoreType, index: int) -> None:
        super().__init__("NSGA-III", store, index)

    def get_pareto_solutions(self, dataset: Dataset) -> list[list[tuple[int, int]]]:
        problem = PyMooOptimizationProblem(dataset)
        ref_dirs = get_reference_directions("uniform", n_dim=3, n_partitions=12)
        algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)
        res = minimize(problem, algorithm, termination=("n_gen", 5), seed=1)

        return [decode_pymoo_x(dataset, sol) for sol in res.X]  # type: ignore
