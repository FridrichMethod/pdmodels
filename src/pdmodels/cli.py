import argparse
from collections.abc import Callable

from pdmodels import af2ig, esmfold, mpnn


def register_subparser(
    subparsers: argparse._SubParsersAction,
    name: str,
    setup_func: Callable,
    cli_func: Callable,
    **kwargs,
) -> None:
    """Helper function to register a subparser."""
    parser = subparsers.add_parser(name, **kwargs)
    setup_func(parser)
    parser.set_defaults(func=cli_func)


def main():
    parser = argparse.ArgumentParser(prog="pdmodels", description="Protein Design Models CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add af2ig
    register_subparser(
        subparsers,
        "af2ig",
        af2ig.setup_parser,
        af2ig.cli,
        help="AlphaFold2 initial guess",
    )

    # Add esmfold
    register_subparser(subparsers, "esmfold", esmfold.setup_parser, esmfold.cli, help="ESMFold")

    # Add mpnn
    register_subparser(subparsers, "mpnn", mpnn.setup_parser, mpnn.cli, help="MPNN sampling")

    args = parser.parse_args()
    args.func(args)
