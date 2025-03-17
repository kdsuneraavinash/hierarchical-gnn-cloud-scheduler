from itertools import product
from constants import N_HOST, N_TASK, N_VM, N_WORKFLOW_TASK
from dataset.real_world import RealWorldDatasetArgs
from train import Args, train


for m, e, s in product(range(3), range(3), range(3)):
    if m + e + s == 0:
        continue
    train(
        Args(
            exp_name=f"gnn_[{m}][{e}][{s}]",
            track=True,
            wandb_project_name="hierarchical-cloud-task-scheduling",
            wandb_entity="kdsuneraavinash-shared-team",
            test_iterations=4,
            dataset=RealWorldDatasetArgs(
                task_count=N_TASK,
                max_vm_count=N_VM,
                max_host_count=N_HOST,
                min_tasks_per_workflow=N_WORKFLOW_TASK,
                makespan_preference=m,
                energy_consumption_preference=e,
                latency_score_preference=s,
            ),
            test_dataset=RealWorldDatasetArgs(
                task_count=N_TASK,
                max_vm_count=N_VM,
                max_host_count=N_HOST,
                min_tasks_per_workflow=N_WORKFLOW_TASK,
                makespan_preference=m,
                energy_consumption_preference=e,
                latency_score_preference=s,
            ),
        )
    )
