import csv
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from apex.cli import main
from apex.generate import generate
from apex.populate import populate
from apex.schema import SpecError, load_spec

ROOT = Path(__file__).resolve().parents[1]
BANKING = ROOT / "specs" / "banking_populated.yaml"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column(table: dict, name: str) -> dict:
    return next(column for column in table["columns"] if column["name"] == name)


def _grouped(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bank")
    spec = load_spec(BANKING)
    generate(spec, out, n_customers=24, seed=42)
    return out


def test_row_counts_and_cardinality(generated: Path) -> None:
    raw = yaml.safe_load(BANKING.read_text(encoding="utf-8"))
    customers = _read(generated / "core" / "customer.csv")
    accounts = _read(generated / "core" / "account.csv")
    loans = _read(generated / "lending" / "loan.csv")
    snapshots = _read(generated / "lending" / "loan_snapshot.csv")
    transactions = _read(generated / "payments" / "transaction.csv")
    fraud = _read(generated / "fraud" / "fraud_alert.csv")
    assert len(customers) == 24

    accounts_by_customer = _grouped(accounts, "customer_id")
    account_rule = raw["entities"]["account"]["per_parent"]
    assert set(accounts_by_customer) == {row["customer_id"] for row in customers}
    assert all(account_rule["min"] <= len(rows) <= account_rule["max"] for rows in accounts_by_customer.values())

    loans_by_customer = _grouped(loans, "customer_id")
    loan_rule = raw["functional_areas"]["lending"]["tables"]["loan"]["per_parent"]
    assert all(loan_rule["min"] <= len(rows) <= loan_rule["max"] for rows in loans_by_customer.values())
    assert set(loans_by_customer) == {row["customer_id"] for row in customers}

    snapshots_by_loan = _grouped(snapshots, "loan_id")
    snapshot_rule = raw["functional_areas"]["lending"]["tables"]["loan_snapshot"]["per_parent"]
    assert set(snapshots_by_loan) == {row["loan_id"] for row in loans}
    assert all(
        snapshot_rule["min"] <= len(rows) <= snapshot_rule["max"] for rows in snapshots_by_loan.values()
    )

    tx_by_account = _grouped(transactions, "account_id")
    tx_rule = raw["entities"]["transaction"]["per_parent"]
    assert set(tx_by_account) == {row["account_id"] for row in accounts}
    assert all(tx_rule["min"] <= len(rows) <= tx_rule["max"] for rows in tx_by_account.values())

    fraud_counts: dict[str, int] = defaultdict(int)
    for row in fraud:
        fraud_counts[row["account_id"]] += 1
    fraud_rule = raw["functional_areas"]["fraud"]["tables"]["fraud_alert"]["per_parent"]
    assert all(
        fraud_rule["min"] <= fraud_counts[row["account_id"]] <= fraud_rule["max"] for row in accounts
    )


def test_referential_integrity_across_areas(generated: Path) -> None:
    customers = _read(generated / "core" / "customer.csv")
    accounts = _read(generated / "core" / "account.csv")
    loans = _read(generated / "lending" / "loan.csv")
    snapshots = _read(generated / "lending" / "loan_snapshot.csv")
    transactions = _read(generated / "payments" / "transaction.csv")
    alerts = _read(generated / "aml" / "aml_alert.csv")
    fraud = _read(generated / "fraud" / "fraud_alert.csv")

    customer_ids = {row["customer_id"] for row in customers}
    account_ids = {row["account_id"] for row in accounts}
    account_owner = {row["account_id"]: row["customer_id"] for row in accounts}
    customer_fico = {row["customer_id"]: row["fico"] for row in customers}
    transaction_ids = {row["transaction_id"] for row in transactions}
    transactions_by_id = {row["transaction_id"]: row for row in transactions}

    assert {row["customer_id"] for row in accounts} <= customer_ids
    assert {row["customer_id"] for row in loans} <= customer_ids
    assert {row["customer_id"] for row in snapshots} <= customer_ids
    assert {row["account_id"] for row in transactions} <= account_ids
    assert {row["customer_id"] for row in transactions} <= customer_ids
    assert {row["customer_id"] for row in alerts} <= customer_ids
    assert {row["customer_id"] for row in fraud} <= customer_ids
    assert {row["transaction_id"] for row in alerts} <= transaction_ids
    assert len({row["transaction_id"] for row in alerts}) == len(alerts)

    for row in transactions:
        assert row["customer_id"] == account_owner[row["account_id"]]
    for row in loans:
        assert row["fico"] == customer_fico[row["customer_id"]]
    for row in fraud:
        assert row["customer_id"] == account_owner[row["account_id"]]
    for row in alerts:
        parent = transactions_by_id[row["transaction_id"]]
        assert row["customer_id"] == parent["customer_id"]
        assert row["account_id"] == parent["account_id"]
        assert row["amount"] == parent["amount"]
    for row in snapshots:
        assert row["customer_id"] in customer_ids


def test_ranges_and_monotone_columns(generated: Path) -> None:
    raw = yaml.safe_load(BANKING.read_text(encoding="utf-8"))
    loans = _read(generated / "lending" / "loan.csv")
    rate = _column(raw["functional_areas"]["lending"]["tables"]["loan"], "interest_rate")
    for row in loans:
        value = Decimal(row["interest_rate"])
        assert Decimal(str(rate["min"])) <= value <= Decimal(str(rate["max"]))
        assert value <= Decimal("0.30")

    income = _column(raw["entities"]["customer"], "annual_income")
    for row in _read(generated / "core" / "customer.csv"):
        value = Decimal(row["annual_income"])
        assert Decimal(str(income["min"])) <= value <= Decimal(str(income["max"]))
        assert Decimal("1000") <= Decimal(row["credit_limit"]) <= Decimal("75000")

    scores = [Decimal(row["risk_score"]) for row in _read(generated / "fraud" / "fraud_alert.csv")]
    scores += [Decimal(row["risk_score"]) for row in _read(generated / "aml" / "aml_alert.csv")]
    assert scores
    assert all(Decimal("0") <= score <= Decimal("1") for score in scores)

    for account_id, rows in _grouped(_read(generated / "payments" / "transaction.csv"), "account_id").items():
        posted = [row["posted_at"] for row in rows]
        assert posted == sorted(posted)
        assert len(posted) == len(set(posted)), account_id

    for rows in _grouped(_read(generated / "lending" / "loan_snapshot.csv"), "loan_id").values():
        months = [int(row["month_index"]) for row in rows]
        balances = [Decimal(row["remaining_balance"]) for row in rows]
        assert months == list(range(1, len(rows) + 1))
        assert balances == sorted(balances, reverse=True)
        assert len(balances) == len(set(balances))
        assert all(Decimal("0") <= balance <= Decimal("250000") for balance in balances)

    for rows in _grouped(_read(generated / "fraud" / "fraud_alert.csv"), "account_id").values():
        ranks = [int(row["suspicion_rank"]) for row in rows]
        assert ranks == sorted(ranks, reverse=True)
        assert len(ranks) == len(set(ranks))
        assert all(1 <= rank <= 50 for rank in ranks)


def test_same_seed_reproduces_files(tmp_path: Path) -> None:
    spec = load_spec(BANKING)
    first = tmp_path / "a"
    second = tmp_path / "b"
    generate(spec, first, n_customers=12, seed=99)
    generate(spec, second, n_customers=12, seed=99)
    for path in first.rglob("*.csv"):
        other = second / path.relative_to(first)
        assert path.read_bytes() == other.read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_cli_validate_and_generate(tmp_path: Path) -> None:
    assert main(["validate", str(BANKING)]) == 0
    out = tmp_path / "cli"
    assert main(["generate", str(BANKING), "-o", str(out), "--customers", "8", "--seed", "3"]) == 0
    manifest = (out / "manifest.json").read_text(encoding="utf-8")
    assert '"n_customers": 8' in manifest
    assert (out / "core" / "customer.csv").exists()
    assert (out / "aml" / "aml_alert.csv").exists()


def test_populate_requires_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SpecError, match="OPENAI_API_KEY"):
        populate(ROOT / "specs" / "master.yaml", tmp_path / "populated.yaml")
