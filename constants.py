N_TASK = 200  # Number of tasks
N_VM = 20  # Number of VMs
N_WORKFLOW_TASK = 20
N_HOST = 10

F_TASK = 10  # Number of task features
F_VM = 10  # Number of VM features
OBS_SIZE = (N_TASK * F_TASK) + (N_TASK * N_VM * F_VM) + N_TASK + N_VM + (N_TASK * N_TASK)
ACT_SIZE = N_TASK * N_VM + N_VM

TEST_SEED = 100_000  # Seed used for testing
EVALUATION_SEED = 200_000  # Seed used for evaluation

MAKESPAN_PREFERENCE = 1  # Preference for makespan optimization
ENERGY_CONSUMPTION_PREFERENCE = 1  # Preference for Energy consumption optimization
SLA_PENALTY_PREFERENCE = 1  # Preference for SLA penalty optimization
