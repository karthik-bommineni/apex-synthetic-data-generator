from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from apex.populate import strip_yaml_fence
from apex.schema import SpecError, load_spec, parse_spec

ROOT = Path(__file__).resolve().parents[1]
BANKING = ROOT / "specs" / "banking_populated.yaml"
MASTER = ROOT / "specs" / "master.yaml"


def _raw() -> dict:
    loaded = yaml.safe_load(BANKING.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_banking_spec_loads() -> None:
    spec = load_spec(BANKING)
    assert spec.domain == "banking"
    assert spec.request.n_customers == 5000
    assert "aml" in spec.request.functional_areas
    assert "fraud" in spec.request.functional_areas


def test_master_yaml_carries_prompt_and_skeleton() -> None:
    loaded = yaml.safe_load(MASTER.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    instructions = loaded["instructions"]
    assert "foreign_key" in instructions
    assert "interest_rate" in instructions
    assert "consistent_with" in instructions
    assert loaded["request"]["n_customers"] == 5000
    assert "guardrails" in loaded
    skeleton = loaded["skeleton"]
    assert skeleton["entities"]["customer"]["primary_key"] == ["customer_id"]
    assert "loan" in skeleton["functional_areas"]["lending"]["tables"]


def test_unknown_strategy_is_rejected() -> None:
    raw = _raw()
    raw["entities"]["customer"]["columns"][1]["strategy"] = "poisson"
    with pytest.raises(SpecError):
        parse_spec(raw)


def test_min_above_max_is_rejected() -> None:
    raw = _raw()
    income = raw["entities"]["customer"]["columns"][3]
    income["min"] = 100
    income["max"] = 10
    with pytest.raises(SpecError):
        parse_spec(raw)


def test_broken_reference_is_rejected() -> None:
    raw = _raw()
    raw["functional_areas"]["aml"]["tables"]["aml_alert"]["columns"][1]["references"] = "ghost.transaction_id"
    with pytest.raises(SpecError, match="unknown table"):
        parse_spec(raw)


def test_guardrail_blocks_unrealistic_interest_cap() -> None:
    raw = _raw()
    for column in raw["functional_areas"]["lending"]["tables"]["loan"]["columns"]:
        if column["name"] == "interest_rate":
            column["max"] = 0.40
            column["mean"] = 0.20
    with pytest.raises(SpecError, match="guardrail max"):
        parse_spec(raw)


def test_negative_balance_requires_explicit_override() -> None:
    raw = _raw()
    balance = raw["entities"]["account"]["columns"][4]
    balance["min"] = -500
    balance["mu"] = 8.4
    with pytest.raises(SpecError, match="guardrail min"):
        parse_spec(raw)

    allowed = deepcopy(raw)
    allowed["entities"]["account"]["columns"][4]["allows_negative"] = True
    parse_spec(allowed)


def test_unrequested_area_is_rejected() -> None:
    raw = _raw()
    raw["functional_areas"]["virtual_care"] = {"tables": {}}
    with pytest.raises(SpecError, match="not in the request"):
        parse_spec(raw)


def test_strip_yaml_fence() -> None:
    text = "```yaml\nspec_version: 1\n```"
    assert strip_yaml_fence(text) == "spec_version: 1"
