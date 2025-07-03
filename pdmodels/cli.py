import argparse

from pdmodels import af2ig, esmfold


def main():
    parser = argparse.ArgumentParser(
        prog="pdmodels", description="Protein Design Models CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add af2ig
    parser_af2ig = subparsers.add_parser("af2ig", help="AlphaFold2 Initial Guess")
    af2ig.setup_parser(parser_af2ig)
    parser_af2ig.set_defaults(func=af2ig.cli)

    # Add esmfold
    parser_esmfold = subparsers.add_parser("esmfold", help="ESMFold")
    esmfold.setup_parser(parser_esmfold)
    parser_esmfold.set_defaults(func=esmfold.cli)

    args = parser.parse_args()
    args.func(args)
