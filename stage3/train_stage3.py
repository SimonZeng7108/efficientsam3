from argparse import ArgumentParser

from hydra import initialize_config_module

from sam3.train.train import main
from sam3.train.utils.train_utils import register_omegaconf_resolvers


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Config path under sam3.train configs.",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="0 to run in the allocated job, 1 to submit through Submitit.",
    )
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--qos", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume Stage 3 training from an existing checkpoint path.",
    )
    return parser


if __name__ == "__main__":
    initialize_config_module("sam3.train", version_base="1.2")
    parser = build_parser()
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    register_omegaconf_resolvers()
    main(args)
