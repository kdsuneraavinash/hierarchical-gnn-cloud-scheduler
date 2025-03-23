from algorithms.base_mo import BaseMoScheduler, SolutionStoreType
from algorithms.pymoo_problem import PyMooOptimizationProblem, decode_pymoo_x
from dataset.models import Dataset
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions


class MoeaDScheduler(BaseMoScheduler):
    def __init__(self, store: SolutionStoreType, index: int) -> None:
        super().__init__("MOEA/D", store, index)

    def get_pareto_solutions(self, dataset: Dataset) -> list[list[tuple[int, int]]]:
        problem = PyMooOptimizationProblem(dataset)
        ref_dirs = get_reference_directions("uniform", n_dim=3, n_partitions=12)
        algorithm = MOEAD(ref_dirs)
        res = minimize(problem, algorithm, termination=("n_gen", 5), seed=1)

        return [decode_pymoo_x(dataset, sol) for sol in res.X]  # type: ignore
