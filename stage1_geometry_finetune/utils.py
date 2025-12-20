"""
Utility functions for Stage 2 training.
"""

import argparse


def add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments to parser."""
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE",
                        help='path to config file')
    parser.add_argument('--opts', nargs='+', default=None,
                        help="Modify config options by adding 'KEY VALUE' pairs")
    parser.add_argument('--batch-size', type=int, help="batch size per GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained', type=str, help='path to pretrained model')
    parser.add_argument('--resume', type=str, help='path to resume checkpoint')
    parser.add_argument('--sam3-checkpoint', type=str, help='path to SAM3 checkpoint')
    parser.add_argument('--teacher-embed-path', type=str, help='path to teacher embeddings')
    parser.add_argument('--accumulation-steps', type=int, help='gradient accumulation steps')
    parser.add_argument('--use-checkpoint', action='store_true', help='use gradient checkpointing')
    parser.add_argument('--disable-amp', action='store_true', help='disable AMP')
    parser.add_argument('--only-cpu', action='store_true', help='CPU only')
    parser.add_argument('--output', type=str, default='output_geometry_finetune', help='output directory')
    parser.add_argument('--tag', type=str, default='default', help='experiment tag')
    parser.add_argument('--eval', action='store_true', help='evaluate only')
    parser.add_argument('--throughput', action='store_true', help='test throughput')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    return parser
