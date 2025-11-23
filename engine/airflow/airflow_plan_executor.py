import json
from pathlib import Path
from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

dataset_file = Path(__file__).parent / "dataset.json"
with dataset_file.open("r") as fr:
    dataset = json.load(fr)

tasks: list[dict] = dataset["tasks"]
vms: list[dict] = dataset["vms"]

with DAG(dag_id="vm_task_workflow_loops", tags=["vm", "loop-load", "dag"]) as dag:
    airflow_tasks = {}
    for tid, task in enumerate(tasks):
        exec_time = task["len"] / vms[task["vm"]]["speed"]
        airflow_tasks[tid] = BashOperator(
            task_id=f"task_{tid}",
            bash_command=f"sleep {exec_time:.4f}",
            queue=f"vm{task['vm']}",
        )

    for tid, task in enumerate(tasks):
        for dep in task["deps"]:
            if dep in airflow_tasks:
                airflow_tasks[dep] >> airflow_tasks[tid]
