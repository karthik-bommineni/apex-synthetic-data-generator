"""Seeded column samplers. Every numeric draw stays inside its declared range."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from apex.schema import Column, SpecError


def round_clip(value: float, column: Column) -> float | int:
    precision = column.precision
    if column.type == "integer" and precision is None:
        precision = 0
    if precision is not None:
        value = float(f"{value:.{precision}f}")
    if column.min is not None and value < column.min:
        value = float(column.min if precision is None else f"{column.min:.{precision}f}")
    if column.max is not None and value > column.max:
        value = float(column.max if precision is None else f"{column.max:.{precision}f}")
    if precision == 0 or column.type == "integer":
        return int(round(value))
    return value


class Sampler:
    def __init__(
        self,
        rng: np.random.Generator,
        history_start: datetime,
        history_end: datetime,
    ) -> None:
        self.rng = rng
        self.history_start = history_start
        self.history_end = history_end

    def draw(
        self,
        column: Column,
        n: int,
        *,
        groups: list[Any] | None = None,
        source: list[Any] | None = None,
    ) -> list[Any]:
        if n == 0:
            return []
        if column.strategy in {"foreign_key", "consistent_with"}:
            raise SpecError(f"{column.name} is resolved from parent tables, not sampled")
        if column.strategy == "sequential":
            return self._sequential(column, n, groups)
        if column.strategy == "uniform":
            return self._uniform(column, n)
        if column.strategy == "normal":
            return self._normal(column, n)
        if column.strategy == "lognormal":
            return self._lognormal(column, n)
        if column.strategy == "strictly_increasing":
            return self._monotone(column, n, groups, decreasing=False)
        if column.strategy == "strictly_decreasing":
            return self._monotone(column, n, groups, decreasing=True)
        if column.strategy == "categorical":
            return self._categorical(column, n)
        if column.strategy == "bernoulli":
            return [bool(flag) for flag in (self.rng.random(n) < column.p)]
        if column.strategy == "datetime":
            return self._datetimes(n)
        if column.strategy == "correlate_with":
            if source is None:
                raise SpecError(f"{column.name} is missing its correlate_with source")
            return self._correlate(column, source)
        if column.strategy == "constant":
            return [column.constant] * n
        raise SpecError(f"unknown strategy {column.strategy}")

    def pick(self, values: list[Any], n: int, *, replace: bool) -> list[Any]:
        if n == 0:
            return []
        if not values:
            raise SpecError("cannot sample foreign keys from an empty parent table")
        if replace:
            indexes = self.rng.integers(0, len(values), size=n)
        else:
            if n > len(values):
                raise SpecError("not enough parent keys for a unique foreign key")
            indexes = self.rng.choice(len(values), size=n, replace=False)
        return [values[int(index)] for index in indexes]

    def sample_unique_ints(self, lo: int, hi: int, n: int) -> np.ndarray:
        if n == 0:
            return np.array([], dtype=np.int64)
        span = hi - lo + 1
        if lo > hi or span < n:
            raise SpecError(f"cannot draw {n} unique values from [{lo}, {hi}]")
        if n == span:
            return np.arange(lo, hi + 1, dtype=np.int64)
        if span <= 2_000_000:
            chosen = self.rng.choice(span, size=n, replace=False)
            return np.sort(chosen + lo).astype(np.int64)
        picks: set[int] = set()
        while len(picks) < n:
            draw = self.rng.integers(lo, hi + 1, size=n - len(picks))
            picks.update(int(value) for value in draw.tolist())
        return np.asarray(sorted(picks), dtype=np.int64)

    def _sequential(self, column: Column, n: int, groups: list[Any] | None) -> list[Any]:
        start = 1 if column.start is None else column.start
        if column.scope == "within":
            if groups is None:
                raise SpecError(f"{column.name} is missing group keys")
            output: list[Any] = [None] * n
            for indexes in _group_indexes(groups).values():
                for offset, index in enumerate(indexes):
                    output[index] = self._format_sequential(column, start + offset)
            return output
        return [self._format_sequential(column, start + index) for index in range(n)]

    def _format_sequential(self, column: Column, number: int) -> Any:
        if column.type != "id":
            return number
        width = column.width or 0
        body = f"{number:0{width}d}" if width else str(number)
        return f"{column.prefix or ''}{body}"

    def _uniform(self, column: Column, n: int) -> list[Any]:
        assert column.min is not None and column.max is not None
        if column.type == "integer":
            lo = int(math.ceil(column.min))
            hi = int(math.floor(column.max))
            if lo > hi:
                raise SpecError(f"{column.name} integer range is empty")
            if lo == hi:
                return [lo] * n
            return [int(value) for value in self.rng.integers(lo, hi + 1, size=n)]
        drawn = self.rng.uniform(column.min, column.max, size=n)
        return [round_clip(float(value), column) for value in drawn]

    def _normal(self, column: Column, n: int) -> list[Any]:
        assert column.mean is not None and column.std is not None
        assert column.min is not None and column.max is not None
        drawn = self._truncated(
            lambda size: self.rng.normal(column.mean, column.std, size=size),
            n,
            column.min,
            column.max,
        )
        return [round_clip(value, column) for value in drawn]

    def _lognormal(self, column: Column, n: int) -> list[Any]:
        assert column.mu is not None and column.sigma is not None
        assert column.min is not None and column.max is not None
        drawn = self._truncated(
            lambda size: self.rng.lognormal(column.mu, column.sigma, size=size),
            n,
            column.min,
            column.max,
        )
        return [round_clip(value, column) for value in drawn]

    def _truncated(
        self,
        draw_batch: Callable[[int], np.ndarray],
        n: int,
        lo: float,
        hi: float,
    ) -> list[float]:
        kept: list[float] = []
        for _ in range(24):
            if len(kept) >= n:
                break
            need = n - len(kept)
            batch = draw_batch(max(need * 4, 32))
            for item in np.asarray(batch).tolist():
                value = float(item)
                if lo <= value <= hi:
                    kept.append(value)
                    if len(kept) == n:
                        break
        if len(kept) < n:
            extra = np.asarray(draw_batch(n - len(kept))).tolist()
            for item in extra:
                kept.append(float(min(hi, max(lo, float(item)))))
        return kept[:n]

    def _monotone(
        self,
        column: Column,
        n: int,
        groups: list[Any] | None,
        *,
        decreasing: bool,
    ) -> list[Any]:
        if column.scope == "within":
            if groups is None:
                raise SpecError(f"{column.name} is missing group keys")
            output: list[Any] = [None] * n
            for indexes in _group_indexes(groups).values():
                values = self._mono_series(column, len(indexes), decreasing=decreasing)
                for index, value in zip(indexes, values, strict=True):
                    output[index] = value
            return output
        return self._mono_series(column, n, decreasing=decreasing)

    def _mono_series(self, column: Column, n: int, *, decreasing: bool) -> list[Any]:
        if column.type == "datetime":
            start, end = self._history_seconds()
            picks = self.sample_unique_ints(start, end, n)
            if decreasing:
                picks = picks[::-1]
            return [_format_timestamp(int(value)) for value in picks]
        assert column.min is not None and column.max is not None
        precision = column.precision
        if column.type == "integer" or precision == 0:
            lo = int(math.ceil(column.min))
            hi = int(math.floor(column.max))
            picks = self.sample_unique_ints(lo, hi, n)
            if decreasing:
                picks = picks[::-1]
            return [int(value) for value in picks]
        digits = 4 if precision is None else precision
        scale = 10**digits
        lo = int(math.ceil(column.min * scale - 1e-9))
        hi = int(math.floor(column.max * scale + 1e-9))
        picks = self.sample_unique_ints(lo, hi, n)
        if decreasing:
            picks = picks[::-1]
        return [int(value) / scale for value in picks]

    def _categorical(self, column: Column, n: int) -> list[Any]:
        assert column.values is not None
        probabilities = None
        if column.weights is not None:
            total = float(sum(column.weights))
            probabilities = [weight / total for weight in column.weights]
        indexes = self.rng.choice(len(column.values), size=n, p=probabilities)
        return [column.values[int(index)] for index in np.asarray(indexes).tolist()]

    def _datetimes(self, n: int) -> list[str]:
        start, end = self._history_seconds()
        if start == end:
            return [_format_timestamp(start)] * n
        drawn = self.rng.integers(start, end + 1, size=n)
        return [_format_timestamp(int(value)) for value in drawn]

    def _correlate(self, column: Column, source: list[Any]) -> list[Any]:
        assert column.min is not None and column.max is not None and column.correlation is not None
        src = np.asarray(source, dtype=float)
        n = int(src.shape[0])
        low = float(src.min()) if n else 0.0
        high = float(src.max()) if n else 0.0
        if high == low:
            scaled = np.zeros(n)
        else:
            scaled = (src - low) / (high - low)
        strength = abs(column.correlation)
        if column.correlation < 0:
            scaled = 1.0 - scaled
        mixed = strength * scaled + (1.0 - strength) * self.rng.random(n)
        values = column.min + mixed * (column.max - column.min)
        return [round_clip(float(value), column) for value in values]

    def _history_seconds(self) -> tuple[int, int]:
        start = int(self.history_start.replace(tzinfo=timezone.utc).timestamp())
        end = int(self.history_end.replace(tzinfo=timezone.utc).timestamp())
        if end < start:
            raise SpecError("history window is empty")
        return start, end


def _group_indexes(groups: list[Any]) -> dict[Any, list[int]]:
    buckets: dict[Any, list[int]] = {}
    for index, key in enumerate(groups):
        buckets.setdefault(key, []).append(index)
    return buckets


def _format_timestamp(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
