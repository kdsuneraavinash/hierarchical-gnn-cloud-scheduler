from collections import defaultdict, deque
import json
import random
import time

from kubernetes import client, config, watch

try:
    config.load_incluster_config()
except:
    config.load_kube_config()

v1 = client.CoreV1Api()

# ----------------------------------------------------------------------------------------------------------------------


def topological_sort(tasks):
    indeg = defaultdict(int)
    graph = defaultdict(list)
    id_to_task = {i: f"task-{t['id']}" for i, t in enumerate(tasks)}

    # Compute in-degrees and graph
    for i, t in enumerate(tasks):
        indeg[i] = len(t["deps"])
        for d in t["deps"]:
            graph[d].append(i)

    # Initialize queue with tasks having no dependencies
    queue = deque([i for i in range(len(tasks)) if indeg[i] == 0])
    order = []

    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nxt in graph[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    if len(order) != len(tasks):
        raise ValueError("Cycle detected in dependencies")

    return [id_to_task[i] for i in order]


data = json.loads(
    """
{"vms": [{"speed": 1188}, {"speed": 1818}, {"speed": 3072}, {"speed": 3072}, {"speed": 1818}], "tasks": [{"id": 0, "len": 7704, "deps": [], "vm": 1}, {"id": 1, "len": 7334, "deps": [0], "vm": 3}, {"id": 2, "len": 3999, "deps": [0, 1], "vm": 1}, {"id": 3, "len": 9139, "deps": [1], "vm": 3}, {"id": 4, "len": 3014, "deps": [2, 3], "vm": 1}, {"id": 5, "len": 5876, "deps": [3, 4], "vm": 3}, {"id": 6, "len": 3725, "deps": [4], "vm": 1}, {"id": 7, "len": 2000, "deps": [5, 6], "vm": 3}, {"id": 8, "len": 8014, "deps": [6, 7], "vm": 1}, {"id": 9, "len": 7877, "deps": [7], "vm": 3}, {"id": 10, "len": 7996, "deps": [8, 9], "vm": 1}, {"id": 11, "len": 761, "deps": [9, 10], "vm": 3}, {"id": 12, "len": 3883, "deps": [10], "vm": 1}, {"id": 13, "len": 119, "deps": [11, 12], "vm": 3}, {"id": 14, "len": 4557, "deps": [12, 13], "vm": 1}, {"id": 15, "len": 12425, "deps": [13], "vm": 3}, {"id": 16, "len": 1209, "deps": [14, 15], "vm": 1}, {"id": 17, "len": 6753, "deps": [16, 15], "vm": 3}, {"id": 18, "len": 11055, "deps": [16], "vm": 1}, {"id": 19, "len": 2782, "deps": [17, 18], "vm": 3}, {"id": 20, "len": 6964, "deps": [18, 19], "vm": 1}, {"id": 21, "len": 6968, "deps": [19], "vm": 3}, {"id": 22, "len": 7600, "deps": [20, 21], "vm": 1}, {"id": 23, "len": 3300, "deps": [21, 22], "vm": 3}, {"id": 24, "len": 8476, "deps": [22], "vm": 1}, {"id": 25, "len": 4450, "deps": [24, 23], "vm": 3}, {"id": 26, "len": 7777, "deps": [24, 25], "vm": 1}, {"id": 27, "len": 8887, "deps": [25], "vm": 3}, {"id": 28, "len": 8328, "deps": [26, 27], "vm": 1}, {"id": 29, "len": 11377, "deps": [27, 28], "vm": 3}, {"id": 30, "len": 2541, "deps": [28], "vm": 1}, {"id": 31, "len": 1778, "deps": [29, 30], "vm": 3}, {"id": 32, "len": 5313, "deps": [30, 31], "vm": 1}, {"id": 33, "len": 8950, "deps": [31], "vm": 3}, {"id": 34, "len": 6983, "deps": [32, 33], "vm": 1}, {"id": 35, "len": 1486, "deps": [33, 34], "vm": 3}, {"id": 36, "len": 9635, "deps": [34], "vm": 1}, {"id": 37, "len": 4967, "deps": [35, 36], "vm": 3}, {"id": 38, "len": 11166, "deps": [36, 37], "vm": 1}, {"id": 39, "len": 14753, "deps": [37], "vm": 3}, {"id": 40, "len": 5397, "deps": [38, 39], "vm": 1}, {"id": 41, "len": 5036, "deps": [40, 39], "vm": 3}, {"id": 42, "len": 5545, "deps": [40, 41], "vm": 1}, {"id": 43, "len": 11289, "deps": [40, 41], "vm": 3}, {"id": 44, "len": 2723, "deps": [40, 42], "vm": 1}, {"id": 45, "len": 164, "deps": [43, 44], "vm": 3}, {"id": 46, "len": 5104, "deps": [42, 43, 44, 45], "vm": 1}, {"id": 47, "len": 8380, "deps": [45, 46], "vm": 3}, {"id": 48, "len": 3080, "deps": [46, 47], "vm": 1}, {"id": 49, "len": 2905, "deps": [48, 47], "vm": 3}]}
"""
)
tasks = data["tasks"]
vm_names = [
    "scheduler-sim-worker",
    "scheduler-sim-worker2",
    "scheduler-sim-worker4",
    "scheduler-sim-worker5",
    "scheduler-sim-worker5",
]
target_vm = {f"task-{t['id']}": vm_names[t["vm"]] for t in tasks}
ordered_tasks = topological_sort(tasks)
# ----------------------------------------------------------------------------------------------------------------------


def get_schedulable_nodes():
    """Return all node names."""
    nodes = v1.list_node().items
    node_names = [n.metadata.name for n in nodes]
    return node_names


def get_pending_pods():
    """Return pods that are pending and use the custom scheduler."""
    pods = v1.list_pod_for_all_namespaces().items
    pending = [
        p
        for p in pods
        if p.status.phase == "Pending" and p.spec.scheduler_name == "python-scheduler" and p.spec.node_name is None
    ]
    return pending


def bind_pod(pod, node_name):
    """Bind the given pod to the selected node."""
    target = client.V1ObjectReference(kind="Node", api_version="v1", name=node_name)
    meta = client.V1ObjectMeta(name=pod.metadata.name, namespace=pod.metadata.namespace)
    body = client.V1Binding(target=target, metadata=meta)

    v1.create_namespaced_binding(namespace=pod.metadata.namespace, body=body)
    print(f"✅ Bound pod {pod.metadata.name} → node {node_name}")


def dependencies_satisfied(pod):
    deps_str = pod.metadata.annotations.get("deps", "")
    deps = [d.strip() for d in deps_str.split(",") if d.strip()]
    for dep in deps:
        try:
            dep_pod = v1.read_namespaced_pod(dep, pod.metadata.namespace)
            if dep_pod.status.phase != "Succeeded":
                return False
        except client.exceptions.ApiException:
            return False
    return True


def main():
    print("🎯 Python scheduler started.")
    nodes = get_schedulable_nodes()
    print(f"Detected nodes: {nodes}")

    while True:
        pods = get_pending_pods()
        if not pods:
            time.sleep(1)
            continue

        suitable_pods = [pod for pod in pods if dependencies_satisfied(pod)]
        if not suitable_pods:
            time.sleep(1)
            continue

        pod = min(suitable_pods, key=lambda p: ordered_tasks.index(p.metadata.name))
        node = target_vm[pod.metadata.name]
        try:
            bind_pod(pod, node)
        except Exception as e:
            print(f"❌ Failed to bind {pod.metadata.name}: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()
