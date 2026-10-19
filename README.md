# Apex synthetic data generator

Generate CSV tables that share keys across functional areas. A master YAML file carries both the prompt and the skeleton, so any chat model can fill it in. Python samples the rows, which keeps foreign keys, ranges, and seeds deterministic.

Banking is the included example: core customers and accounts, lending, payments, AML, and fraud. The same spec shape works for other domains. Virtual care, for example, is another functional area with its own tables and the same shared-key rules.

## Setup

```text
pip install -e ".[dev]"
```

## Populate the spec

Edit the `request` block at the top of [specs/master.yaml](specs/master.yaml) (customer count, functional areas, history window, domain). Paste the whole file into ChatGPT, Claude, Gemini, Mistral, or another model. Save the YAML it returns.

Or call an OpenAI-compatible endpoint:

```text
set OPENAI_API_KEY=...
set OPENAI_BASE_URL=https://api.openai.com/v1
set APEX_MODEL=gpt-4o-mini
python -m apex populate -i specs/master.yaml -o specs/populated.yaml
```

[specs/banking_populated.yaml](specs/banking_populated.yaml) is a ready-to-generate banking spec, so you can skip the model.

## Generate

```text
python -m apex validate specs/banking_populated.yaml
python -m apex generate specs/banking_populated.yaml -o out/
python -m apex generate specs/banking_populated.yaml -o out/ --customers 500 --seed 7
```

Tables land in `out/<functional_area>/<table>.csv`, plus `out/manifest.json`.

Shared keys (`customer_id`, `account_id`, `transaction_id`) are drawn from the parent table. Copied attributes, such as a loan's FICO or an alert's amount, match the parent row. Numeric columns stay inside the ranges in the spec. Retail `interest_rate` cannot exceed 30 percent, and the banking example stays near 3.5 to 17.9 percent.

Column strategies: `sequential`, `uniform`, `normal`, `lognormal`, `strictly_increasing`, `strictly_decreasing`, `categorical`, `bernoulli`, `datetime`, `foreign_key`, `consistent_with`, `correlate_with`, `constant`.

## Tests

```text
pytest
```
