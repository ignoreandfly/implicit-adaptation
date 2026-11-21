import importlib

TASKS = ["gsm8k"]


def get_task(name):
    if name not in TASKS:
        raise ValueError(f"unknown task {name}; have {TASKS}")
    return importlib.import_module(f"tasks.{name}")
