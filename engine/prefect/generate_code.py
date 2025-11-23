import json
from textwrap import indent
from collections import defaultdict, deque


def topological_sort(tasks):
    indeg = defaultdict(int)
    graph = defaultdict(list)
    id_to_task = {i: t for i, t in enumerate(tasks)}

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


def find_terminal_tasks(tasks):
    """Return a list of task IDs that no other task depends on."""
    dependents = set()
    for t in tasks:
        dependents.update(t["deps"])
    all_ids = set(range(len(tasks)))
    terminals = all_ids - dependents
    return sorted(terminals)


def generate_prefect_code(data: dict) -> str:
    vms = data["vms"]
    tasks = data["tasks"]

    # Ensure each task has an explicit id for reference
    for i, t in enumerate(tasks):
        t["id"] = i

    # Sort tasks topologically
    sorted_tasks = topological_sort(tasks)
    terminal_tasks = find_terminal_tasks(tasks)

    lines = []
    lines.append("from time import sleep")
    lines.append("from prefect import flow, task\n")

    # --- Task definitions ---
    for task in sorted_tasks:
        lines.append(f"@task")
        lines.append(f"def task_{task['id']}():")
        lines.append(f"    sleep({task['len'] / vms[task['vm']]['speed']:.3f})")
        lines.append(f'    print("Task {task["id"]} completed")\n')

    # --- Flow definition ---
    lines.append("@flow")
    lines.append("def workflow():")
    body = []

    for task in sorted_tasks:
        if task["deps"]:
            deps = ", ".join([f"t{d}" for d in task["deps"]])
            body.append(f"t{task['id']} = task_{task['id']}.submit(wait_for=[{deps}])")
        else:
            body.append(f"t{task['id']} = task_{task['id']}.submit()")

    for t_id in terminal_tasks:
        body.append(f"t{t_id}.result()")

    lines.append(indent("\n".join(body), "    "))
    lines.append("\n\nworkflow()")

    return "\n".join(lines)


if __name__ == "__main__":
    json_str = """
{"vms": [{"speed": 1818}, {"speed": 1818}, {"speed": 1818}, {"speed": 1818}, {"speed": 1818}], "tasks": [{"id": 0, "len": 5571, "deps": [47], "vm": 3}, {"id": 1, "len": 5233, "deps": [0, 3], "vm": 3}, {"id": 2, "len": 12324, "deps": [0, 44], "vm": 0}, {"id": 3, "len": 5205, "deps": [2, 18], "vm": 3}, {"id": 4, "len": 10119, "deps": [0], "vm": 1}, {"id": 5, "len": 6390, "deps": [27, 4], "vm": 0}, {"id": 6, "len": 2302, "deps": [1, 13, 3, 5], "vm": 1}, {"id": 7, "len": 477, "deps": [6, 15], "vm": 3}, {"id": 8, "len": 10224, "deps": [16, 7], "vm": 0}, {"id": 9, "len": 1172, "deps": [28], "vm": 3}, {"id": 10, "len": 2259, "deps": [9], "vm": 3}, {"id": 11, "len": 11726, "deps": [10, 30], "vm": 3}, {"id": 12, "len": 11390, "deps": [10, 45], "vm": 0}, {"id": 13, "len": 7668, "deps": [10, 20], "vm": 1}, {"id": 14, "len": 821, "deps": [10], "vm": 2}, {"id": 15, "len": 7444, "deps": [11, 12, 13, 14, 22], "vm": 3}, {"id": 16, "len": 5224, "deps": [12, 15], "vm": 0}, {"id": 17, "len": 9281, "deps": [16, 25], "vm": 1}, {"id": 18, "len": 388, "deps": [46], "vm": 3}, {"id": 19, "len": 8173, "deps": [1, 18], "vm": 3}, {"id": 20, "len": 6513, "deps": [19, 29], "vm": 1}, {"id": 21, "len": 6552, "deps": [20, 5], "vm": 0}, {"id": 22, "len": 6539, "deps": [11, 21], "vm": 3}, {"id": 23, "len": 9468, "deps": [19, 4], "vm": 1}, {"id": 24, "len": 3990, "deps": [39, 22, 23], "vm": 1}, {"id": 25, "len": 987, "deps": [24], "vm": 1}, {"id": 26, "len": 3636, "deps": [25, 17], "vm": 1}, {"id": 27, "len": 7028, "deps": [37], "vm": 0}, {"id": 28, "len": 5394, "deps": [19, 27], "vm": 3}, {"id": 29, "len": 3432, "deps": [28, 23], "vm": 1}, {"id": 30, "len": 3106, "deps": [10, 29], "vm": 3}, {"id": 31, "len": 10491, "deps": [30, 7], "vm": 3}, {"id": 32, "len": 12022, "deps": [6, 31], "vm": 1}, {"id": 33, "len": 15907, "deps": [32, 40], "vm": 3}, {"id": 34, "len": 1964, "deps": [33, 41], "vm": 3}, {"id": 35, "len": 9075, "deps": [49, 34], "vm": 0}, {"id": 36, "len": 5234, "deps": [42, 35], "vm": 3}, {"id": 37, "len": 8291, "deps": [2], "vm": 0}, {"id": 38, "len": 5439, "deps": [37, 31], "vm": 3}, {"id": 39, "len": 1777, "deps": [32, 38], "vm": 1}, {"id": 40, "len": 6775, "deps": [38, 39], "vm": 3}, {"id": 41, "len": 12515, "deps": [40, 33], "vm": 3}, {"id": 42, "len": 9665, "deps": [41, 34], "vm": 3}, {"id": 43, "len": 2764, "deps": [42, 26], "vm": 1}, {"id": 44, "len": 6547, "deps": [], "vm": 0}, {"id": 45, "len": 3485, "deps": [44, 21], "vm": 0}, {"id": 46, "len": 11313, "deps": [0, 44], "vm": 3}, {"id": 47, "len": 7559, "deps": [44], "vm": 3}, {"id": 48, "len": 2942, "deps": [45, 46, 47], "vm": 4}, {"id": 49, "len": 13438, "deps": [48, 8], "vm": 0}]}
"""
    data = json.loads(json_str)
    code = generate_prefect_code(data)
    print(code)
