"""
CLI entry for the portfolio engine (thin orchestration).

Run from the `engine` directory:
    python main.py --help
    python main.py demo --tickers RELIANCE.NS TCS.NS INFY.NS

Full pipeline uses modules under services/ and utils/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Running `python main.py` from engine/: ensure this directory is on sys.path
_ENGINE_ROOT = Path(__file__).resolve().parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))


def _cmd_demo(args: argparse.Namespace) -> int:
    from services.report import build_full_report

    configure_logging(args.verbose)
    log = logging.getLogger(__name__)
    tickers = [t.strip() for t in args.tickers if t.strip()]
    if len(tickers) < 2:
        log.error("Provide at least two tickers for a meaningful portfolio demo.")
        return 2
    try:
        report = build_full_report(
            tickers=tickers,
            target_weights=None,
            start=args.start,
            end=args.end,
            risk_free_annual=args.risk_free,
            random_portfolios=args.random_portfolios,
            ridge_epsilon=args.ridge,
            plot_path=args.plot or None,
            random_seed=args.seed,
        )
    except Exception as e:
        log.exception("Demo failed: %s", e)
        return 1
    print(json.dumps(report, indent=2))
    if args.plot:
        log.info("Frontier figure written to %s", Path(args.plot).resolve())
    return 0


def configure_logging(verbose: bool) -> None:
    from utils.logging_config import setup_logging

    setup_logging(debug=verbose)


def main() -> int:
    parser = argparse.ArgumentParser(description="PF Radar portfolio engine")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Fetch data, optimize, print JSON report")
    p_demo.add_argument(
        "tickers",
        nargs="+",
        help="e.g. RELIANCE.NS TCS.NS INFY.NS",
    )
    p_demo.add_argument("--start", default=None, help="YYYY-MM-DD")
    p_demo.add_argument("--end", default=None, help="YYYY-MM-DD")
    p_demo.add_argument(
        "--risk-free",
        type=float,
        default=None,
        help="Annual risk-free rate (default: India G-Sec proxy from constants)",
    )
    p_demo.add_argument("--random-portfolios", type=int, default=2500)
    p_demo.add_argument("--ridge", type=float, default=1e-10)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument(
        "--plot",
        default="",
        help="If set, PNG path to save efficient frontier figure (e.g. frontier.png)",
    )
    p_demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
