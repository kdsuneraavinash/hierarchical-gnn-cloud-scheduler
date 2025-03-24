# --- Default sizes ---

DEFAULT_TASK_COUNT = 100
DEFAULT_MAX_VM_COUNT = 10
DEFAULT_MAX_HOST_COUNT = 4
DEFAULT_MAX_TASKS_PER_WORKFLOW = 20

# --- Buffer size/Tensor size constraints ---

N_TASK = DEFAULT_TASK_COUNT
N_VM = DEFAULT_MAX_VM_COUNT
F_TASK = 6  # Number of task features
F_VM = 4  # Number of VM features
OBS_SIZE = (N_TASK * F_TASK) + (N_TASK * N_VM * F_VM) + N_TASK + (N_TASK * N_VM) + (N_TASK * N_TASK)
ACT_SIZE = N_TASK * N_VM + N_VM

# --- Random Seeds ---

TEST_SEED = 100_000  # Seed used for testing
EVALUATION_SEED = 200_000  # Seed used for evaluation

# --- Preferences ---

MAKESPAN_PREFERENCE = 1
ENERGY_CONSUMPTION_PREFERENCE = 1
LATENCY_SCORE_PREFERENCE = 1

# --- Other Constants ---

CHART_AXIS_PAD = 0.1
