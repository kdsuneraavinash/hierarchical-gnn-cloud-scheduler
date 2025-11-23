import json
from pathlib import Path
import yaml

# ---------- Load Data ----------
data = json.loads(
    """
{"vms": [{"speed": 1188}, {"speed": 1818}, {"speed": 3072}, {"speed": 3072}, {"speed": 1818}], "tasks": [{"id": 0, "len": 7704, "deps": [], "vm": 3}, {"id": 1, "len": 7334, "deps": [0], "vm": 3}, {"id": 2, "len": 3999, "deps": [1], "vm": 3}, {"id": 3, "len": 9139, "deps": [], "vm": 1}, {"id": 4, "len": 3014, "deps": [3], "vm": 1}, {"id": 5, "len": 5876, "deps": [4], "vm": 1}, {"id": 6, "len": 3725, "deps": [2], "vm": 3}, {"id": 7, "len": 2000, "deps": [6], "vm": 3}, {"id": 8, "len": 8014, "deps": [7], "vm": 3}, {"id": 9, "len": 7877, "deps": [5], "vm": 1}, {"id": 10, "len": 7996, "deps": [9], "vm": 1}, {"id": 11, "len": 761, "deps": [8, 10], "vm": 3}, {"id": 12, "len": 3883, "deps": [10], "vm": 1}, {"id": 13, "len": 119, "deps": [11, 12], "vm": 3}, {"id": 14, "len": 4557, "deps": [13], "vm": 3}, {"id": 15, "len": 12425, "deps": [12], "vm": 1}, {"id": 16, "len": 1209, "deps": [14, 15], "vm": 3}, {"id": 17, "len": 6753, "deps": [16], "vm": 3}, {"id": 18, "len": 11055, "deps": [15], "vm": 1}, {"id": 19, "len": 2782, "deps": [17, 18], "vm": 3}, {"id": 20, "len": 6964, "deps": [19], "vm": 3}, {"id": 21, "len": 6968, "deps": [18], "vm": 1}, {"id": 22, "len": 7600, "deps": [21], "vm": 1}, {"id": 23, "len": 3300, "deps": [20, 22], "vm": 3}, {"id": 24, "len": 8476, "deps": [22], "vm": 1}, {"id": 25, "len": 4450, "deps": [24, 23], "vm": 3}, {"id": 26, "len": 7777, "deps": [25], "vm": 3}, {"id": 27, "len": 8887, "deps": [24], "vm": 1}, {"id": 28, "len": 8328, "deps": [27], "vm": 1}, {"id": 29, "len": 11377, "deps": [26, 28], "vm": 3}, {"id": 30, "len": 2541, "deps": [28], "vm": 1}, {"id": 31, "len": 1778, "deps": [30], "vm": 1}, {"id": 32, "len": 5313, "deps": [31], "vm": 1}, {"id": 33, "len": 8950, "deps": [29], "vm": 3}, {"id": 34, "len": 6983, "deps": [33], "vm": 3}, {"id": 35, "len": 1486, "deps": [34], "vm": 3}, {"id": 36, "len": 9635, "deps": [32], "vm": 1}, {"id": 37, "len": 4967, "deps": [36], "vm": 1}, {"id": 38, "len": 11166, "deps": [35, 37], "vm": 3}, {"id": 39, "len": 14753, "deps": [37], "vm": 1}, {"id": 40, "len": 5397, "deps": [38, 39], "vm": 3}, {"id": 41, "len": 5036, "deps": [40], "vm": 3}, {"id": 42, "len": 5545, "deps": [41], "vm": 3}, {"id": 43, "len": 11289, "deps": [40, 39], "vm": 1}, {"id": 44, "len": 2723, "deps": [40, 42], "vm": 3}, {"id": 45, "len": 164, "deps": [44], "vm": 3}, {"id": 46, "len": 5104, "deps": [42, 43, 45], "vm": 3}, {"id": 47, "len": 8380, "deps": [46], "vm": 3}, {"id": 48, "len": 3080, "deps": [47], "vm": 3}, {"id": 49, "len": 2905, "deps": [48], "vm": 3}]}
"""
)

vms = data["vms"]
tasks = data["tasks"]

out_dir = Path("yamls")
out_dir.mkdir(exist_ok=True)


# ---------- Helper to create node ----------
def make_node_yaml(idx):
    node = {
        "apiVersion": "v1",
        "kind": "Node",
        "metadata": {"name": f"vm-{idx}"},
        "spec": {"taints": []},  # empty spec for demo
    }
    return node


# ---------- Helper to create pod ----------
def make_pod_yaml(task):
    vm_index = task["vm"]
    vm_name = f"vm-{vm_index}"
    sleep_time = round(task["len"] / vms[vm_index]["speed"], 3)

    deps = task.get("deps", [])
    deps_str = ",".join(map(lambda d: f"task-{d}", deps)) if deps else ""

    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"task-{task['id']}",
            "annotations": {"deps": deps_str},
        },
        "spec": {
            "schedulerName": "python-scheduler",
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": f"task-{task['id']}",
                    "image": "busybox",
                    "command": ["sh", "-c", f"sleep {sleep_time}"],
                }
            ],
        },
    }
    return pod


# ---------- Generate Node YAMLs ----------
# for i in range(len(vms)):
#     node_yaml = make_node_yaml(i)
#     with open(out_dir / f"node-{i}.yaml", "w") as f:
#         yaml.dump(node_yaml, f, sort_keys=False)

# ---------- Generate Pod YAMLs ----------
for t in tasks:
    pod_yaml = make_pod_yaml(t)
    with open(out_dir / f"pod-task-{t['id']}.yaml", "w") as f:
        yaml.dump(pod_yaml, f, sort_keys=False)

print(f"✅ Generated {len(tasks)} pod YAMLs and {len(vms)} node YAMLs in {out_dir}/")
