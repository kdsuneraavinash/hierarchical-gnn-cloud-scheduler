from algorithms.base_mo import BaseMoScheduler, SolutionStore
from algorithms.pymoo_problem import PyMooOptimizationProblem, decode_pymoo_x
from dataset.models import Dataset
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


class Nsga2Scheduler(BaseMoScheduler):
    def __init__(self, store: SolutionStore, index: int) -> None:
        super().__init__("NSGA-II", store, index)

    def get_pareto_solutions(self, dataset: Dataset) -> list[list[tuple[int, int]]]:
        problem = PyMooOptimizationProblem(dataset)
        algorithm = NSGA2(pop_size=100)
        algorithm.setup(problem, termination=("n_gen", 20))
        res = algorithm.run()

        res = minimize(problem, algorithm, termination=("n_gen", 10), seed=1)

        return [decode_pymoo_x(dataset, sol) for sol in res.X]  # type: ignore
