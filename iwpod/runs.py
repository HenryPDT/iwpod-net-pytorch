"""Experiment run directories: out/train/<name>[_2..], never overwrite.

Mirrors pixeltable-yolox's get_unique_output_name: the first run keeps the
bare name, reruns auto-increment. Also records the exact CLI invocation and a
snapshot of the resolved config for reproducibility.
"""
import os
import shlex
import sys


def get_unique_output_name(base_dir, name):
    """Returns (base_dir, unique_name); appends _2, _3, ... on collision."""
    full_path = os.path.join(base_dir, name)
    if not os.path.exists(full_path):
        return base_dir, name
    counter = 2
    while True:
        new_name = f"{name}_{counter}"
        if not os.path.exists(os.path.join(base_dir, new_name)):
            return base_dir, new_name
        counter += 1


def resolve_run_dir(base_dir, name):
    base, unique = get_unique_output_name(base_dir, name)
    run_dir = os.path.join(base, unique)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, unique


def default_task_dir(task, weights_path):
    """Default output dir for read-only commands: out/<task>/<weights-stem>/.

    Keeps artifacts out of the weights' directory (eval/infer must never
    write next to checkpoints) while staying predictable. Explicit CLI flags
    always override this.
    """
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    return os.path.join("out", task, stem)


def write_train_command(run_dir, prog="iwpod"):
    """Save the exact CLI invocation (yolox-style train_command.txt)."""
    try:
        cmd = shlex.join([prog, *sys.argv[1:]])
    except Exception:
        cmd = "(unavailable)"
    try:
        with open(os.path.join(run_dir, "train_command.txt"), "w") as f:
            f.write("# Training command used for this experiment:\n")
            f.write(f"{cmd}\n")
    except OSError:
        pass


def write_config_snapshot(run_dir, cfg):
    import yaml
    try:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except OSError:
        pass
