"""Command line: validate, populate, and generate."""

from __future__ import annotations

import argparse
import sys

from apex.generate import generate
from apex.populate import populate
from apex.schema import SpecError, load_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apex",
        description="Generate synthetic tables with referential integrity.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate_cmd = sub.add_parser("validate", help="Validate a populated spec")
    validate_cmd.add_argument("spec")

    generate_cmd = sub.add_parser("generate", help="Generate CSV tables from a populated spec")
    generate_cmd.add_argument("spec")
    generate_cmd.add_argument("-o", "--out", default="out")
    generate_cmd.add_argument("--seed", type=int, default=None)
    generate_cmd.add_argument("--customers", type=int, default=None)

    populate_cmd = sub.add_parser(
        "populate",
        help="Fill the master YAML with an OpenAI-compatible model",
    )
    populate_cmd.add_argument("-i", "--input", required=True)
    populate_cmd.add_argument("-o", "--output", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            spec = load_spec(args.spec)
            areas = ", ".join(spec.request.functional_areas)
            print(f"valid spec: domain={spec.domain} areas={areas}")
            return 0
        if args.command == "generate":
            spec = load_spec(args.spec)
            manifest = generate(
                spec,
                args.out,
                seed=args.seed,
                n_customers=args.customers,
            )
            print(f"wrote {args.out} ({manifest['n_customers']} customers, seed {manifest['seed']})")
            for table in manifest["tables"]:
                print(f"  {table['path']}: {table['rows']} rows")
            return 0
        if args.command == "populate":
            spec = populate(args.input, args.output)
            print(f"wrote {args.output} (domain={spec.domain})")
            return 0
    except SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
