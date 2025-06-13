import os
import joblib
import pandas as pd
import json
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Data Validation Model 
# This class defines the structure and data types for the incoming API request.
# It ensures that any data sent to your endpoint is valid before it's processed.
# Field(...) is used for aliasing to match your model's column names.
class CreditFeatures(BaseModel):
    total_credits: float = Field(..., alias="Total_Credits")
    total_debits: float = Field(..., alias="Total_Debits")
    salary_total: float = Field(..., alias="Probable_Salary_Total")
    savings_total: float = Field(..., alias="Transfers_To_Savings_Total")
    fee_monthly: float = Field(..., alias="Fees_Monthly_Account_Total")
    fee_service: float = Field(..., alias="Fees_Service_Charges_Total")
    fee_penalty: float = Field(..., alias="Fees_Penalty_Interest_Total")
    fee_unpaid: float = Field(..., alias="Fees_Unpaid_Item_Charges_Total")
    fee_honouring: float = Field(..., alias="Fees_Honouring_Charges_Total")
    days_below_zero: int = Field(..., alias="Days_Balance_Below_Zero")
    min_balance: float = Field(..., alias="Min_Balance_Recorded")
    max_balance: float = Field(..., alias="Max_Balance_Recorded")
    net_income: float = Field(..., alias="Net_Income")
    salary_ratio: float = Field(..., alias="Salary_Ratio")
    savings_ratio: float = Field(..., alias="Savings_Ratio")
    expense_ratio: float = Field(..., alias="Expense_Ratio")
    balance_stability: float = Field(..., alias="Balance_Stability")
    fees_total: float = Field(..., alias="Fees_Total")

    class Config:
        populate_by_name = True # Allows using either snake_case or the alias in requests

# Application Setup 
# Create the main FastAPI application instance
app = FastAPI(
    title="Credit Score Prediction API",
    description="An API to predict credit scores based on financial metrics.",
    version="1.0.0"
)

# Load Model and Interpretation Data 
# These files are loaded once when the application starts up for efficiency.
model = joblib.load("credit_score_regressor.pkl")
with open("score_interpretation.json", "r") as f:
    score_map = json.load(f)

# Helper function to find the interpretation for a given score
def get_interpretation(score: float):
    """Finds the rating and interpretation for a numeric score from the loaded map."""
    for entry in score_map:
        if entry["min_score"] <= score <= entry["max_score"]:
            return entry["rating"], entry["interpretation"]
    return "Unknown", "No interpretation found for this score."

# API Endpoints 
@app.get("/health", tags=["Health Check"], summary="Check if the API is running")
def health_check():
    """
    A simple health check endpoint that Vertex AI or which ever hosting platform uses to verify
    the application is live and ready to serve requests.
    """
    return {"status": "healthy"}

@app.post("/predict", tags=["Prediction"], summary="Predict a credit score")
def predict_score(data: CreditFeatures):
    """
    Receives financial data, predicts a credit score, and returns the score
    along with its rating and interpretation.
    """
    # Convert the incoming Pydantic model to a dictionary, using aliases
    # to match the model's expected column names.
    features_dict = data.model_dump(by_alias=True)
    
    # Create a pandas DataFrame from the dictionary.
    # The index=[0] is important as the model expects a 2D array-like input.
    features_df = pd.DataFrame(features_dict, index=[0])

    # Predict the score using the loaded model
    score = model.predict(features_df)[0]
    
    # Get the human-readable interpretation
    rating, interpretation = get_interpretation(score)

    # Return the structured JSON response
    return {
        "predicted_score": round(score),
        "rating": rating,
        "interpretation": interpretation
    }
