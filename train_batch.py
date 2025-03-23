from itertools import product
from train import Args, train


for m, e, s in product(range(3), range(3), range(3)):
    if m + e + s == 0:
        continue
    train(
        Args(
            exp_name=f"mlp_syn_[{m}][{e}][{s}]",
            agent_type="mlp",
            dataset_type="synthetic",
            makespan_pref=m,
            energy_pref=e,
            latency_pref=s,
        )
    )
