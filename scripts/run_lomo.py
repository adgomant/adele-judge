#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy

from adele_judge.config import load_config, save_config
from adele_judge.data import load_and_construct_targets
from adele_judge.splits import enumerate_lomo_models
from adele_judge.utils import ensure_dir, project_output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    base_config = load_config(args.config, args.override)
    models = enumerate_lomo_models(load_and_construct_targets(base_config))
    root = ensure_dir(project_output_dir(base_config) / "lomo")
    for model_id in models:
        fold_config = deepcopy(base_config)
        safe_model = model_id.replace("/", "_").replace(" ", "_")
        fold_config["project"]["run_name"] = f"{base_config['project']['run_name']}_lomo_{safe_model}"
        fold_config["project"]["output_dir"] = str(root / safe_model)
        fold_config["split"]["mode"] = "leave_one_model_out"
        fold_config["split"]["held_out_model"] = model_id
        config_path = root / safe_model / "config.yaml"
        save_config(fold_config, config_path)
        cmd = [sys.executable, "scripts/prepare_dataset.py", "--config", str(config_path)]
        if not args.prepare_only:
            cmd = [sys.executable, "scripts/train_judge.py", "--config", str(config_path)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
