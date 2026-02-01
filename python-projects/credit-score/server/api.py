from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import requirements
from config.logs_configs.logging_config import setup_logging
from server.schemas import CreditRiskInput, CreditRiskOutput
from inference.inference import predictor

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Starting API")


# Initialize FastAPI app
app = FastAPI(
    title="Credit Score Prediction API",
    description="API to predict the credit risk of a customer using a deep learning model.",
    version="1.0.0",
)

# Configure CORS
origins = ["*"]
# en produccion se deben limitar los origenes
# para que otras aplicaciones de la compañia pueda consumir el api
# por ejemplo:
# origins = [
#     "http://localhost",
#     "http://localhost:3000", quí corre la interface web de la carpeta client_web
#     "http://localhost:5173",
#     "http://localhost:8000",
#     "http://localhost:8080",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    """
    Redirects the root endpoint to the API documentation.
    """
    return RedirectResponse(url="/docs")


@app.post(
    "/credit_score_prediction",
    response_model=CreditRiskOutput,
    tags=["Prediction"],
    summary="Predict Credit Risk",
    description="Predicts whether the credit risk is 'good' or 'bad' based on customer data.",
)
async def predict_credit_risk(data: CreditRiskInput) -> CreditRiskOutput:
    """
    Predicts the credit risk (good/bad) and the associated probability.
    """
    try:
        logger.info(f"Received prediction request: {data}")

        # Perform inference using the predictor singleton
        result = predictor.inference(data)

        logger.info(f"Prediction result: {result}")

        return CreditRiskOutput(
            prediction=result["prediction"], probability=result["probability"]
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred during inference: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# uv run uvicorn server.api:app --reload --port 8000
