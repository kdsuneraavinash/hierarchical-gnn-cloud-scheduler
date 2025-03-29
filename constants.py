# --- Default sizes ---

DEFAULT_TASK_COUNT = 50
DEFAULT_MAX_VM_COUNT = 5
DEFAULT_MAX_HOST_COUNT = 4
DEFAULT_MAX_TASKS_PER_WORKFLOW = 10

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
SCH_COLORS = {
    "HEFT": "#1f77b4",  # muted blue
    "FERPTS": "#ff7f0e",  # orange
    "Random": "#2ca02c",  # green
    "Least Loaded First": "#d62728",  # red
    "Round--Robin": "#9467bd",  # purple
    "Weighted Dynamic": "#8c564b",  # brown
    "MOHEFT": "#e377c2",  # pink
    "NSGA-II": "#7f7f7f",  # gray
    "NSGA-III": "#bcbd22",  # olive
    "MOEA/D": "#17becf",  # cyan
    "Proposed": "#1a55FF",  # strong blue
    "Proposed-Syn": "#1a55FF",  # strong blue
    "Proposed-Real": "#FF1493",  # deep pink
}
