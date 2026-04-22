# PfRadar

PfRadar is a modern, algorithm-driven Portfolio Optimization Engine and Dashboard. It calculates the optimal distribution of assets to balance maximum return and minimum risk using modern portfolio theory, comparing your target allocations against Nifty 50 benchmarks.

## Prerequisites

Ensure you have Python 3.9+ installed and your virtual environment activated (if you're using one).

If you haven't already, install the required dependencies:
```bash
pip install -r requirements.txt
```
*(If you do not have a requirements file, ensure at a minimum you have `fastapi`, `uvicorn`, `yfinance`, `pandas`, `numpy`, and `scipy` installed).*

## Running the Web Dashboard (Recommended)

1. **Start the Live Dashboard API**
   Open a terminal in the root directory (`PfRadar/`) and run the FastAPI server:
   ```bash
   python app.py
   ```
   *This starts an ultra-fast REST backend utilizing Uvicorn and hosts the front-end dashboard concurrently.*
// run using uvicorn not python app.py
2. **Access the Dashboard**
   Once the server is running, open your web browser and navigate to:
   **http://localhost:8000**

3. **Using the Dashboard**
   - **Tickers:** Enter valid Yahoo Finance stock symbols separated by commas *(e.g. `RELIANCE.NS, TCS.NS, INFY.NS`)*.
   - **Weights:** Enter comma-separated decimal weights corresponding to those symbols *(e.g. `0.4, 0.3, 0.3`)*. Ensure that the total sum of the weights is mathematically equal to 1.0.
   - **Risk Profile:** Adjust the slider to represent your risk tolerance.
     - `0.0` = **Minimum Variance Portfolio** (Safest possible allocation algorithmically calculated)
     - `1.0` = **Maximum Sharpe Portfolio** (Targeting the absolute highest Return-to-Risk ratio)
   - Click **Analyze Live** to execute the pipeline and render the 1-year trailing charts and asset trajectory comparisons.

---

## Running the Console CLI (Fallback)

If you only want quick numerical insights directly printed to your terminal without drawing the charts:

Navigate to the `engine/` directory:
```bash
cd engine
```

Run the built-in CLI module using `main.py demo` followed by your space-separated tickers, and appending `--weights`:
```bash
python main.py demo RELIANCE.NS TCS.NS INFY.NS --weights 0.4 0.3 0.3
```

You can view all advanced console flags (like custom start dates, risk levels, and risk-free return proxies) by running:
```bash
python main.py demo --help
```
