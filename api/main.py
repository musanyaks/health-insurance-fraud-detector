from fastapi import FastAPI

from api.routes.score import router as score_router

app = FastAPI(
    title="Health Claims Fraud Scoring API",
    description="Composite fraud risk scoring (rules + ML + SHAP reasons). "
                "MVP: synthetic data, time-split model.",
    version="0.1.0",
)
app.include_router(score_router)


@app.get("/health")
def health():
    return {"status": "ok"}