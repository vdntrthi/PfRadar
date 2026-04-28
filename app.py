import json
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "engine"))
from engine.services.report import build_full_report
from fastapi.templating import Jinja2Templates
from xhtml2pdf import pisa
from io import BytesIO
import tempfile 
from datetime import datetime   
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/report")
def get_report(tickers: str, weights: str, risk: float | None = None, period: str = "12M"):
    try:
        t_list = [t.strip() for t in tickers.split(",") if t.strip()]
        w_list = [float(w.strip()) for w in weights.split(",") if w.strip()]
        
        if len(t_list) != len(w_list):
            raise HTTPException(status_code=400, detail=" Number of tickers must match number of weights ")
        
        valid_periods = {"1M", "3M", "6M", "12M"}
        if period not in valid_periods:
            raise HTTPException(status_code=400, detail=f"Invalid period. Choose from: {valid_periods}")
            
        target_weights = dict(zip(t_list, w_list))
        
        report_data = build_full_report(
            tickers=t_list,
            target_weights=target_weights,
            risk_score=risk,
            chart_period=period,
        )
        return report_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download_report")
def download_report(
    tickers: str,
    weights: str,
    risk: float | None = None,
    period: str = "12M"
):
    try:
        t_list = [t.strip() for t in tickers.split(",") if t.strip()]
        w_list = [float(w.strip()) for w in weights.split(",") if w.strip()]

        if len(t_list) != len(w_list):
            raise HTTPException(
                status_code=400,
                detail="Number of tickers must match number of weights"
            )

        target_weights = dict(zip(t_list, w_list))

        report_data = build_full_report(
            tickers=t_list,
            target_weights=target_weights,
            risk_score=risk,
            chart_period=period,
        )
        print(report_data.keys())
        html = templates.get_template(
            "report_template.html"
        ).render(
            USER_NAME="Investor",
            DATE=datetime.today().strftime("%d %b %Y"),

            HEALTH_SCORE=82,

            RETURN_COMMENT="Optimized Growth Portfolio",
            RISK_COMMENT="Balanced Risk Profile",

            CAGR=round(
                report_data["optimal_portfolio_cagr"] * 100, 2
            ),

            CAGR_STATUS="Healthy",

            VOL=round(
                report_data["target_risk_volatility"] * 100, 2
            ),

            VOL_STATUS="Moderate",

            SHARPE=round(
                report_data["sharpe_ratio"], 2
            ),

            SHARPE_STATUS="Good",

            SUGGESTION_1="Maintain diversification across selected sectors.",
            SUGGESTION_2="Review allocation quarterly.",
            SUGGESTION_3="Reduce concentration if volatility rises.",

            ALLOCATION_TEXT=str(
                report_data["asset_allocation"]
            )
        )

        pdf_buffer = BytesIO()

        pisa.CreatePDF(
                src=html,
                dest=pdf_buffer
        )

        pdf_buffer.seek(0)

        return Response(
            content=pdf_buffer.read(),
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                "attachment; filename=Portfolio_Report.pdf"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (index.html) at root
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
