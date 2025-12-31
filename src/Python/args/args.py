import argparse

from Python.constants.constants import (
    DEFAULT_LOAD_PATH,
    DEFAULT_SAVE_PATH,
    GAME_GRID_SIZE,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    elif v.lower() in ("no", "false", "f", "0", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(
        prog='Learn2Slither',
    )

    parser.add_argument(
        "--menu",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to show the menu. Defaults to True.",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        required=False,
        default=100,
        help="Number of sessions to run",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        required=False,
        default=DEFAULT_SAVE_PATH,
        help=f"Path to save the neural network model. Defaults to {DEFAULT_SAVE_PATH}.",
    )

    parser.add_argument(
        "--load-path",
        type=str,
        required=False,
        default=DEFAULT_LOAD_PATH,
        help="Path to load the neural network model from. Defaults to None.",
    )

    parser.add_argument(
        "--learn",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to enable learning mode. Defaults to True.",
    )

    parser.add_argument(
        "-hs",
        "--human-speed",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to enable human speed mode. Defaults to True.",
    )

    parser.add_argument(
        "--visuals",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to enable visualizations. Defaults to True.",
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=GAME_GRID_SIZE,
        help=f"Size of the grid. Defaults to {GAME_GRID_SIZE}.",
    )

    parser.add_argument(
        "--pve",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to enable player vs ai mode. Defaults to False.",
    )

    return parser.parse_args()
