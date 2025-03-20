from itertools import product
from dataset.generator import DatasetArgs
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
            dataset=DatasetArgs.real_world().with_priority(m, e, s),
            test_dataset=DatasetArgs.real_world().with_priority(m, e, s),
        )
    )
