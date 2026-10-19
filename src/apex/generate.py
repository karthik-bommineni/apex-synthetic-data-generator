"""Generate CSV tables from a populated spec, parents before children."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from apex.sampler import Sampler
from apex.schema import (
    Column,
    Spec,
    SpecError,
    Table,
    column_generation_order,
    dependency_order,
    history_bounds,
    validate_semantics,
)


def generate(
    spec: Spec,
    out_dir: str | Path,
    *,
    seed: int | None = None,
    n_customers: int | None = None,
) -> dict[str, Any]:
    working = spec.model_copy(deep=True)
    if n_customers is not None:
        if n_customers <= 0:
            raise SpecError("n_customers must be positive")
        working.request.n_customers = n_customers
    used_seed = working.seed if seed is None else seed
    tables = validate_semantics(working)
    order = dependency_order(tables)
    rng = np.random.default_rng(used_seed)
    history_start, history_end = history_bounds(working)
    sampler = Sampler(rng, history_start, history_end)

    data: dict[str, list[dict[str, Any]]] = {}
    indexes: dict[str, dict[Any, dict[str, Any]]] = {}
    for name in order:
        rows = _generate_table(name, tables, data, indexes, working, sampler)
        pk = tables[name].primary_key[0]
        index: dict[Any, dict[str, Any]] = {}
        for row in rows:
            key = row[pk]
            if key in index:
                raise SpecError(f"duplicate primary key in {name}: {key}")
            index[key] = row
        data[name] = rows
        indexes[name] = index

    return _write_output(working, tables, order, data, Path(out_dir), used_seed)


def _generate_table(
    name: str,
    tables: dict[str, Table],
    data: dict[str, list[dict[str, Any]]],
    indexes: dict[str, dict[Any, dict[str, Any]]],
    spec: Spec,
    sampler: Sampler,
) -> list[dict[str, Any]]:
    table = tables[name]
    filled: set[str] = set()
    parent_keys: list[Any] | None = None
    if table.per_parent is not None:
        parent = tables[table.per_parent.parent]
        parent_pk = parent.primary_key[0]
        link = _parent_link(table, table.per_parent.parent)
        parent_keys = []
        low = table.per_parent.min
        high = table.per_parent.max
        for parent_row in data[table.per_parent.parent]:
            copies = int(sampler.rng.integers(low, high + 1))
            parent_keys.extend([parent_row[parent_pk]] * copies)
        row_count = len(parent_keys)
        filled.add(link.name)
    else:
        row_count = _resolve_count(name, table, data, spec)

    rows: list[dict[str, Any]] = [{} for _ in range(row_count)]
    if parent_keys is not None:
        link = _parent_link(table, table.per_parent.parent)  # type: ignore[union-attr]
        for index, key in enumerate(parent_keys):
            rows[index][link.name] = key

    for column in column_generation_order(name, table):
        if column.name in filled:
            continue
        if column.strategy == "foreign_key":
            assert column.references is not None
            parent_name = column.references.split(".", 1)[0]
            parent_pk = tables[parent_name].primary_key[0]
            keys = [row[parent_pk] for row in data[parent_name]]
            chosen = sampler.pick(keys, row_count, replace=not column.unique)
            for index, value in enumerate(chosen):
                rows[index][column.name] = value
        elif column.strategy == "consistent_with":
            assert column.consistent_with is not None
            target_table, target_column = column.consistent_with.split(".", 1)
            for row in rows:
                row[column.name] = _lookup(
                    row, name, target_table, target_column, indexes, tables
                )
        else:
            groups = [row[column.within] for row in rows] if column.within else None
            source = None
            if column.strategy == "correlate_with":
                assert column.correlate_with is not None
                source = [row[column.correlate_with] for row in rows]
            values = sampler.draw(column, row_count, groups=groups, source=source)
            for index, value in enumerate(values):
                rows[index][column.name] = value
        filled.add(column.name)

    missing = [column.name for column in table.columns if column.name not in filled]
    if missing:
        raise SpecError(f"{name} did not generate columns: {', '.join(missing)}")
    return rows


def _parent_link(table: Table, parent: str) -> Column:
    matches = [
        column
        for column in table.columns
        if column.references and column.references.split(".", 1)[0] == parent
    ]
    if len(matches) != 1:
        raise SpecError(f"expected one foreign key to {parent}")
    return matches[0]


def _resolve_count(
    name: str,
    table: Table,
    data: dict[str, list[dict[str, Any]]],
    spec: Spec,
) -> int:
    rows = table.rows
    if rows is None:
        raise SpecError(f"{name} is missing a row count")
    if rows.request_key is not None:
        count = int(getattr(spec.request, rows.request_key))
    elif rows.count is not None:
        count = rows.count
    else:
        assert rows.fraction_of is not None and rows.fraction is not None
        parent_n = len(data[rows.fraction_of])
        count = int(round(rows.fraction * parent_n))
        count = max(rows.min_rows, count)
        if rows.max_rows is not None:
            count = min(count, rows.max_rows)
        if parent_n == 0:
            count = 0
    for column in table.columns:
        if not (column.unique and column.references):
            continue
        parent_name = column.references.split(".", 1)[0]
        available = len(data[parent_name])
        if count > available:
            if rows.min_rows > available and available > 0:
                raise SpecError(
                    f"{name} needs {rows.min_rows} unique keys from {parent_name}, "
                    f"which has {available} rows"
                )
            count = available
    return count


def _lookup(
    row: dict[str, Any],
    source: str,
    target_table: str,
    target_column: str,
    indexes: dict[str, dict[Any, dict[str, Any]]],
    tables: dict[str, Table],
) -> Any:
    for column in tables[source].columns:
        if not column.references:
            continue
        mid_table = column.references.split(".", 1)[0]
        mid_row = indexes[mid_table][row[column.name]]
        if mid_table == target_table:
            return mid_row[target_column]
        mid_spec = tables[mid_table]
        for mid_column in mid_spec.columns:
            if not mid_column.references:
                continue
            next_table = mid_column.references.split(".", 1)[0]
            if next_table == target_table:
                parent = indexes[target_table][mid_row[mid_column.name]]
                return parent[target_column]
    raise SpecError(f"{source} cannot resolve {target_table}.{target_column}")


def _write_output(
    spec: Spec,
    tables: dict[str, Table],
    order: list[str],
    data: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    used_seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_tables: list[dict[str, Any]] = []
    for name in order:
        table = tables[name]
        area = table.area or "shared"
        relative = f"{area}/{name}.csv"
        path = out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [column.name for column in table.columns]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in data[name]:
                writer.writerow(
                    {column.name: _format_cell(row[column.name], column) for column in table.columns}
                )
        manifest_tables.append(
            {
                "name": name,
                "area": area,
                "path": relative,
                "rows": len(data[name]),
                "primary_key": list(table.primary_key),
                "foreign_keys": [
                    {"column": column.name, "references": column.references}
                    for column in table.columns
                    if column.references
                ],
                "consistent_with": [
                    {"column": column.name, "source": column.consistent_with}
                    for column in table.columns
                    if column.consistent_with
                ],
            }
        )
    manifest = {
        "domain": spec.domain,
        "seed": used_seed,
        "n_customers": spec.request.n_customers,
        "tables": manifest_tables,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _format_cell(value: Any, column: Column) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if column.type == "integer" or (
        isinstance(value, int) and column.type != "float"
    ):
        return str(int(value))
    if column.type == "float" or isinstance(value, float):
        precision = column.precision if column.precision is not None else 4
        return f"{float(value):.{precision}f}"
    return str(value)
