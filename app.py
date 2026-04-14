import json
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "engine"))
from services.report import build_full_report

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/report")
def get_report(tickers: str, weights: str, risk: float | None = None):
    try:
        t_list = [t.strip() for t in tickers.split(",") if t.strip()]
        w_list = [float(w.strip()) for w in weights.split(",") if w.strip()]
        
        if len(t_list) != len(w_list):
            raise HTTPException(status_code=400, detail="Number of tickers must match number of weights")
            
        target_weights = dict(zip(t_list, w_list))
        
        report_data = build_full_report(
            tickers=t_list,
            target_weights=target_weights,
            risk_score=risk
        )
        return report_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (index.html) at root
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
