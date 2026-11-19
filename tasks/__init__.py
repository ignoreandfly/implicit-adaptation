import importlib

TASKS = ["arithmetic"]  # add "last_letter", "coin_flip" as they are implemented


def get_task(name):
    if name not in TASKS:
        raise ValueError(f"unknown task {name}; have {TASKS}")
    return importlib.import_module(f"tasks.{name}")
