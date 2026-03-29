# PF Radar — portfolio engine (Python)

India-focused **Modern Portfolio Theory** toolkit: Yahoo Finance ingest, return/risk metrics, long-only **min-variance** and **max-Sharpe** (scipy), random **efficient frontier** cloud, **JSON** report.

## Setup

```bash
cd engine
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## CLI

```bash
python main.py demo RELIANCE.NS TCS.NS INFY.NS --plot frontier.png
```

## Tests

- **Unit / fast (default CI):**
  ```bash
  pytest -m "not integration"
  ```
- **Integration** (network — Yahoo Finance; may skip if the API is blocked):
  ```bash
  pytest -m integration
  ```

## Conventions

- **Annualization:** `TRADING_DAYS_PER_YEAR = 252` ([`models/constants.py`](models/constants.py)).
- **Risk-free default:** `DEFAULT_RISK_FREE_ANNUAL_IN` (G-Sec proxy placeholder); override in `build_full_report` or CLI.
