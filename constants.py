from dataset.real_world import RealWorldDatasetArgs
from dataset.synthetic import SyntheticDatasetArgs


N_TASK = 100
N_VM = 5
N_HOST = 4

F_TASK = 7  # Number of task features
F_VM = 4  # Number of VM features
OBS_SIZE = (N_TASK * F_TASK) + (N_TASK * N_VM * F_VM) + N_TASK + (N_TASK * N_VM) + (N_TASK * N_TASK)
ACT_SIZE = N_TASK * N_VM + N_VM

TEST_SEED = 100_000  # Seed used for testing
EVALUATION_SEED = 200_000  # Seed used for evaluation

MAKESPAN_PREFERENCE = 1
ENERGY_CONSUMPTION_PREFERENCE = 1
LATENCY_SCORE_PREFERENCE = 0

REAL_WORLD_DATASET_ARGS = RealWorldDatasetArgs(
    task_count=100,
    max_vm_count=5,
    max_host_count=4,
    makespan_preference=MAKESPAN_PREFERENCE,
    energy_consumption_preference=ENERGY_CONSUMPTION_PREFERENCE,
    latency_score_preference=LATENCY_SCORE_PREFERENCE,
    dag_structure="Montage",
)

SYNTHETIC_DATASET_ARGS = SyntheticDatasetArgs(
    task_count=100,
    max_vm_count=5,
    max_host_count=4,
    max_tasks_per_workflow=20,
    makespan_preference=MAKESPAN_PREFERENCE,
    energy_consumption_preference=ENERGY_CONSUMPTION_PREFERENCE,
    latency_score_preference=LATENCY_SCORE_PREFERENCE,
)

CHART_AXIS_PAD = 0.1
