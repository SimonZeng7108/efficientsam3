"""
Multi-node training entry point for srun-based DDP.

Usage (single node):
    PYTHONPATH=sam3:. srun --gpus=4 --ntasks-per-node=4 \
        python stage3/train_stage3_srun.py -c CONFIG

Usage (multi-node):
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=29720
    PYTHONPATH=sam3:. srun --nodes=4 --gpus-per-node=4 --ntasks-per-node=4 \
        python stage3/train_stage3_srun.py -c CONFIG

Each srun task = one GPU process.  SLURM_PROCID -> RANK,
SLURM_LOCALID -> LOCAL_RANK, SLURM_NTASKS -> WORLD_SIZE.
"""

import os
import sys
from argparse import ArgumentParser

os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "29720"

from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf, open_dict

from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers


def main():
    initialize_config_module("sam3.train", version_base="1.2")
    register_omegaconf_resolvers()

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    cfg = compose(config_name=args.config)

    checkpoint_dir = cfg.trainer.checkpoint.save_dir
    if checkpoint_dir:
        auto_resume_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.exists(auto_resume_path):
            with open_dict(cfg.trainer.checkpoint):
                cfg.trainer.checkpoint.resume_from = auto_resume_path

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        makedir(cfg.launcher.experiment_log_dir)
        with g_pathmgr.open(
            os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
        ) as f:
            f.write(OmegaConf.to_yaml(cfg))
        print("###################### Train App Config ####################")
        print(OmegaConf.to_yaml(cfg))
        print("############################################################")
        print(f"Experiment Log Dir:\n{cfg.launcher.experiment_log_dir}")

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


if __name__ == "__main__":
    main()
