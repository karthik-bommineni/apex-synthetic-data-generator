"""Specification models for the synthetic data generator.

A populated spec describes tables, distributions, and foreign keys. It does not
contain row data. The generator samples rows from this description.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Strategy = Literal[
    "sequential",
    "uniform",
    "normal",
    "lognormal",
    "strictly_increasing",
    "strictly_decreasing",
    "categorical",
    "bernoulli",
    "datetime",
    "foreign_key",
    "consistent_with",
    "correlate_with",
    "constant",
]

ColumnType = Literal[
    "id",
    "integer",
    "float",
    "string",
    "category",
    "boolean",
    "datetime",
    "fk",
]

RANGE_STRATEGIES = {
    "uniform",
    "normal",
    "lognormal",
    "strictly_increasing",
    "strictly_decreasing",
    "correlate_with",
}

MONOTONE_STRATEGIES = {"strictly_increasing", "strictly_decreasing", "sequential"}


class SpecError(ValueError):
    """The spec is syntactically loaded but not usable for generation."""


class Guardrail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str | None = None
    column_glob: str | None = None
    min: float | None = None
    max: float | None = None
    allow_negative_override: bool = False

    @model_validator(mode="after")
    def needs_matcher(self) -> Guardrail:
        if self.column is None and self.column_glob is None:
            raise ValueError("guardrail requires column or column_glob")
        if self.min is None and self.max is None:
            raise ValueError("guardrail requires min or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("guardrail min cannot exceed max")
        return self


class RowSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_key: str | None = None
    count: int | None = Field(default=None, ge=0)
    fraction_of: str | None = None
    fraction: float | None = None
    min_rows: int = Field(default=0, ge=0)
    max_rows: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def exactly_one_mode(self) -> RowSource:
        modes = [
            self.request_key is not None,
            self.count is not None,
            self.fraction_of is not None,
        ]
        if sum(modes) != 1:
            raise ValueError("rows must set exactly one of request_key, count, or fraction_of")
        if self.fraction_of is not None and self.fraction is None:
            raise ValueError("fraction_of requires fraction")
        if self.fraction is not None and not (0 < self.fraction <= 1):
            raise ValueError("fraction must be in (0, 1]")
        if self.max_rows is not None and self.max_rows < self.min_rows:
            raise ValueError("max_rows cannot be below min_rows")
        return self


class PerParent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent: str
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> PerParent:
        if self.min > self.max:
            raise ValueError("per_parent min cannot exceed max")
        return self


class Column(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: ColumnType
    strategy: Strategy
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    mu: float | None = None
    sigma: float | None = None
    precision: int | None = Field(default=None, ge=0, le=8)
    values: list[Any] | None = None
    weights: list[float] | None = None
    references: str | None = None
    consistent_with: str | None = None
    correlate_with: str | None = None
    correlation: float | None = None
    scope: Literal["table", "within"] | None = None
    within: str | None = None
    start: int | None = None
    prefix: str | None = None
    width: int | None = Field(default=None, ge=0, le=18)
    constant: Any | None = None
    p: float | None = None
    allows_negative: bool = False
    unique: bool = False

    @model_validator(mode="after")
    def check_strategy(self) -> Column:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"{self.name}: min cannot exceed max")
        if self.references is not None and _split_ref(self.references) is None:
            raise ValueError(f"{self.name}: references must be table.column")
        if self.consistent_with is not None and _split_ref(self.consistent_with) is None:
            raise ValueError(f"{self.name}: consistent_with must be table.column")
        if self.references is not None and self.strategy != "foreign_key":
            raise ValueError(f"{self.name}: references is only valid for foreign_key")
        if self.consistent_with is not None and self.strategy != "consistent_with":
            raise ValueError(f"{self.name}: consistent_with target requires that strategy")
        if self.strategy == "foreign_key" and not self.references:
            raise ValueError(f"{self.name}: foreign_key requires references")
        if self.strategy == "consistent_with" and not self.consistent_with:
            raise ValueError(f"{self.name}: consistent_with requires a target")
        if self.strategy == "correlate_with":
            if not self.correlate_with:
                raise ValueError(f"{self.name}: correlate_with requires a source column")
            if self.correlation is None or not -1 <= self.correlation <= 1:
                raise ValueError(f"{self.name}: correlation must be between -1 and 1")
            if self.min is None or self.max is None:
                raise ValueError(f"{self.name}: correlate_with requires min and max")
        if self.strategy in {"uniform", "normal", "lognormal"} or (
            self.strategy in {"strictly_increasing", "strictly_decreasing"}
            and self.type != "datetime"
        ):
            if self.min is None or self.max is None:
                raise ValueError(f"{self.name}: {self.strategy} requires min and max")
        if self.strategy == "normal":
            if self.mean is None or self.std is None:
                raise ValueError(f"{self.name}: normal requires mean and std")
            if self.std <= 0:
                raise ValueError(f"{self.name}: std must be positive")
            if self.min is not None and self.max is not None and not (
                self.min <= self.mean <= self.max
            ):
                raise ValueError(f"{self.name}: mean must lie inside min and max")
        if self.strategy == "lognormal":
            if self.mu is None or self.sigma is None:
                raise ValueError(f"{self.name}: lognormal requires mu and sigma")
            if self.sigma <= 0:
                raise ValueError(f"{self.name}: sigma must be positive")
            median = math.exp(self.mu)
            if self.min is not None and self.max is not None and not (
                self.min <= median <= self.max
            ):
                raise ValueError(
                    f"{self.name}: lognormal median exp(mu) must lie inside min and max"
                )
        if self.strategy == "categorical":
            if not self.values:
                raise ValueError(f"{self.name}: categorical requires values")
            if self.weights is not None:
                if len(self.weights) != len(self.values):
                    raise ValueError(f"{self.name}: weights length must match values")
                if any(weight <= 0 for weight in self.weights):
                    raise ValueError(f"{self.name}: weights must be positive")
        if self.strategy == "bernoulli":
            if self.p is None or not 0 <= self.p <= 1:
                raise ValueError(f"{self.name}: bernoulli requires p in [0, 1]")
        if self.strategy == "constant" and self.constant is None:
            raise ValueError(f"{self.name}: constant requires a value")
        if self.strategy == "datetime" and self.type != "datetime":
            raise ValueError(f"{self.name}: datetime strategy requires datetime type")
        if self.strategy in MONOTONE_STRATEGIES and self.scope is None and self.within:
            self.scope = "within"
        if self.strategy in {"strictly_increasing", "strictly_decreasing"} and self.scope is None:
            self.scope = "table"
        if self.scope == "within" and not self.within:
            raise ValueError(f"{self.name}: scope within requires within")
        if self.within and self.strategy not in MONOTONE_STRATEGIES:
            raise ValueError(f"{self.name}: within is only valid for ordered strategies")
        return self


class Table(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: str | None = None
    primary_key: list[str] = Field(min_length=1, max_length=1)
    rows: RowSource | None = None
    per_parent: PerParent | None = None
    columns: list[Column] = Field(min_length=1)

    @model_validator(mode="after")
    def check_table(self) -> Table:
        if (self.rows is None) == (self.per_parent is None):
            raise ValueError("table needs exactly one of rows or per_parent")
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("duplicate column names")
        pk = self.primary_key[0]
        if pk not in names:
            raise ValueError(f"primary key {pk} is missing from columns")
        if self.per_parent is not None:
            linked = [
                column
                for column in self.columns
                if column.references
                and column.references.split(".", 1)[0] == self.per_parent.parent
            ]
            if len(linked) != 1:
                raise ValueError(
                    f"per_parent {self.per_parent.parent} needs exactly one foreign key to that table"
                )
            if linked[0].unique and self.per_parent.max > 1:
                raise ValueError("unique foreign key cannot be used when per_parent max is above 1")
        return self


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    n_customers: int = Field(gt=0)
    functional_areas: list[str] = Field(min_length=1)
    history_start: str
    history_end: str
    seed: int


class FunctionalArea(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tables: dict[str, Table] = Field(default_factory=dict)


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_version: Literal[1] = 1
    domain: str
    seed: int
    request: Request
    guardrails: list[Guardrail] = Field(default_factory=list)
    entities: dict[str, Table]
    functional_areas: dict[str, FunctionalArea]


def _split_ref(reference: str) -> tuple[str, str] | None:
    table, sep, column = reference.partition(".")
    if not sep or not table or not column or "." in column:
        return None
    return table, column


def iter_tables(spec: Spec) -> dict[str, Table]:
    """Return every table keyed by name, with functional-area folders filled in."""
    tables: dict[str, Table] = {}
    for name, table in spec.entities.items():
        tables[name] = table
    for area, functional_area in spec.functional_areas.items():
        for name, table in functional_area.tables.items():
            if name in tables:
                raise SpecError(f"duplicate table name: {name}")
            if table.area is None:
                table.area = area
            tables[name] = table
    return tables


def _matches_guardrail(column: Column, guardrail: Guardrail) -> bool:
    if guardrail.column is not None and column.name == guardrail.column:
        return True
    if guardrail.column_glob is not None and fnmatch(column.name, guardrail.column_glob):
        return True
    return False


def _guardrail_applies(column: Column) -> bool:
    if column.type in {"datetime", "boolean", "category", "string", "fk", "id"}:
        return False
    return column.strategy in RANGE_STRATEGIES


def _can_resolve(source: str, target: str, tables: dict[str, Table]) -> bool:
    if source == target:
        return True
    for column in tables[source].columns:
        if not column.references:
            continue
        mid = column.references.split(".", 1)[0]
        if mid == target:
            return True
        for mid_column in tables[mid].columns:
            if mid_column.references and mid_column.references.split(".", 1)[0] == target:
                return True
    return False


def dependency_order(tables: dict[str, Table]) -> list[str]:
    """Topologically sort tables so parents are generated before children."""
    dependents: dict[str, list[str]] = {name: [] for name in tables}
    indegree = {name: 0 for name in tables}
    for name, table in tables.items():
        deps: set[str] = set()
        for column in table.columns:
            for reference in (column.references, column.consistent_with):
                if reference:
                    deps.add(reference.split(".", 1)[0])
        if table.per_parent is not None:
            deps.add(table.per_parent.parent)
        if table.rows is not None and table.rows.fraction_of is not None:
            deps.add(table.rows.fraction_of)
        deps.discard(name)
        for dep in deps:
            if dep not in tables:
                raise SpecError(f"table {name} references unknown table {dep}")
            dependents[dep].append(name)
            indegree[name] += 1
    ready = [name for name in tables if indegree[name] == 0]
    order: list[str] = []
    while ready:
        ready.sort()
        current = ready.pop(0)
        order.append(current)
        for child in dependents[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(tables):
        raise SpecError("table references contain a cycle")
    return order


def column_generation_order(table_name: str, table: Table) -> list[Column]:
    """Order columns so each derived column is generated after its inputs."""
    by_name = {column.name: column for column in table.columns}
    names = [column.name for column in table.columns]
    dependents: dict[str, list[str]] = {name: [] for name in names}
    indegree = {name: 0 for name in names}
    for column in table.columns:
        deps: set[str] = set()
        if column.strategy == "correlate_with" and column.correlate_with:
            deps.add(column.correlate_with)
        if column.within:
            deps.add(column.within)
        if column.strategy == "consistent_with":
            for other in table.columns:
                if other.strategy == "foreign_key":
                    deps.add(other.name)
        deps.discard(column.name)
        for dep in deps:
            if dep not in by_name:
                raise SpecError(f"{table_name}.{column.name} depends on unknown column {dep}")
            dependents[dep].append(column.name)
            indegree[column.name] += 1
    ready = [name for name in names if indegree[name] == 0]
    order: list[str] = []
    while ready:
        ready.sort(key=names.index)
        current = ready.pop(0)
        order.append(current)
        for child in dependents[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(names):
        raise SpecError(f"column dependencies in {table_name} contain a cycle")
    return [by_name[name] for name in order]


def validate_semantics(spec: Spec) -> dict[str, Table]:
    """Cross-check references, guardrails, areas, and the history window."""
    if spec.domain != spec.request.domain:
        raise SpecError("spec domain and request domain must match")
    try:
        start = date.fromisoformat(spec.request.history_start)
        end = date.fromisoformat(spec.request.history_end)
    except ValueError as exc:
        raise SpecError("history_start and history_end must be YYYY-MM-DD dates") from exc
    if end < start:
        raise SpecError("history_end cannot be earlier than history_start")

    requested = list(spec.request.functional_areas)
    if len(requested) != len(set(requested)):
        raise SpecError("functional_areas contains a duplicate")
    requested_set = set(requested)

    for name, table in spec.entities.items():
        if not table.area:
            raise SpecError(f"entity {name} requires area")
        if table.area not in requested_set:
            raise SpecError(f"entity {name} area {table.area} is not in the request")

    for area in spec.functional_areas:
        if area not in requested_set:
            raise SpecError(f"functional area {area} is not in the request")

    tables = iter_tables(spec)
    covered = {table.area for table in spec.entities.values()} | set(spec.functional_areas)
    for area in requested:
        if area not in covered:
            raise SpecError(f"requested functional area {area} is missing from the spec")

    for name, table in tables.items():
        if not table.area:
            raise SpecError(f"table {name} requires area")
        pk_name = table.primary_key[0]
        pk_column = next(column for column in table.columns if column.name == pk_name)
        if pk_column.strategy != "sequential":
            raise SpecError(f"{name}.{pk_name} primary key must use strategy sequential")
        for column in table.columns:
            if column.min is not None and column.max is not None and column.min > column.max:
                raise SpecError(f"{name}.{column.name}: min cannot exceed max")
            if column.references:
                ref = _split_ref(column.references)
                assert ref is not None
                ref_table, ref_column = ref
                if ref_table not in tables:
                    raise SpecError(f"{name}.{column.name} references unknown table {ref_table}")
                if ref_column not in tables[ref_table].primary_key:
                    raise SpecError(
                        f"{name}.{column.name} references {column.references}, which is not a primary key"
                    )
            if column.consistent_with:
                ref = _split_ref(column.consistent_with)
                assert ref is not None
                ref_table, ref_column = ref
                if ref_table not in tables:
                    raise SpecError(f"{name}.{column.name} consistent_with unknown table {ref_table}")
                target_columns = {item.name for item in tables[ref_table].columns}
                if ref_column not in target_columns:
                    raise SpecError(
                        f"{name}.{column.name} consistent_with unknown column {column.consistent_with}"
                    )
                if not _can_resolve(name, ref_table, tables):
                    raise SpecError(
                        f"{name}.{column.name} cannot reach {ref_table} through foreign keys"
                    )
            if column.correlate_with:
                source = next((item for item in table.columns if item.name == column.correlate_with), None)
                if source is None:
                    raise SpecError(f"{name}.{column.name} correlates with unknown column")
                if source.type not in {"integer", "float"}:
                    raise SpecError(f"{name}.{column.name} must correlate with a numeric column")
            if column.within:
                if column.within not in {item.name for item in table.columns}:
                    raise SpecError(f"{name}.{column.name} groups by unknown column {column.within}")
            if _guardrail_applies(column):
                for guardrail in spec.guardrails:
                    if not _matches_guardrail(column, guardrail):
                        continue
                    skip_min = guardrail.allow_negative_override and column.allows_negative
                    if guardrail.min is not None and not skip_min:
                        if column.min is None or column.min < guardrail.min:
                            raise SpecError(
                                f"{name}.{column.name} violates guardrail min {guardrail.min}"
                            )
                    if guardrail.max is not None and (column.max is None or column.max > guardrail.max):
                        raise SpecError(
                            f"{name}.{column.name} violates guardrail max {guardrail.max}"
                        )
        if table.rows is not None and table.rows.request_key is not None:
            if not hasattr(spec.request, table.rows.request_key):
                raise SpecError(
                    f"{name} rows request_key {table.rows.request_key} is not on the request"
                )

    dependency_order(tables)
    for name, table in tables.items():
        column_generation_order(name, table)
    return tables


def parse_spec(data: dict[str, Any]) -> Spec:
    try:
        spec = Spec.model_validate(data)
    except ValidationError as exc:
        raise SpecError(str(exc)) from exc
    validate_semantics(spec)
    return spec


def load_spec(path: str | Path) -> Spec:
    raw = Path(path).read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise SpecError("spec must be a YAML mapping")
    return parse_spec(loaded)


def history_bounds(spec: Spec) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(spec.request.history_start)
    end_day = date.fromisoformat(spec.request.history_end)
    end = datetime(end_day.year, end_day.month, end_day.day, 23, 59, 59)
    return start, end
