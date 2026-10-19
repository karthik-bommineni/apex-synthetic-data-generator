"""Synthetic data generator with cross-table referential integrity."""

from apex.generate import generate
from apex.schema import Spec, SpecError, load_spec

__all__ = ["Spec", "SpecError", "generate", "load_spec"]
