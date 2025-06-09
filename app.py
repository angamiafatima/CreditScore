import gradio as gr
import joblib
import pandas as pd
import random
import json

# Load model and score interpretation mapping
model = joblib.load("credit_score_regressor.pkl")

with open("score_interpretation.json", "r") as f:
    score_map = json.load(f)

# Find interpretation from numeric score
def get_interpretation(score):
    for entry in score_map:
        if entry["min_score"] <= score <= entry["max_score"]:
            return entry["rating"], entry["interpretation"]
    return "Unknown", "No interpretation found"

# Prediction function
def predict_score(
    total_credits, total_debits, salary_total, savings_total,
    fee_monthly, fee_service, fee_penalty, fee_unpaid, fee_honouring,
    days_below_zero, min_balance, max_balance,
    net_income, salary_ratio, savings_ratio, expense_ratio, balance_stability, fees_total
):
    features = pd.DataFrame([[  
        total_credits, total_debits, salary_total, savings_total,
        fee_monthly, fee_service, fee_penalty, fee_unpaid, fee_honouring,
        days_below_zero, min_balance, max_balance,
        net_income, salary_ratio, savings_ratio, expense_ratio, balance_stability, fees_total
    ]], columns=[
        "Total_Credits", "Total_Debits", "Probable_Salary_Total", "Transfers_To_Savings_Total",
        "Fees_Monthly_Account_Total", "Fees_Service_Charges_Total", "Fees_Penalty_Interest_Total",
        "Fees_Unpaid_Item_Charges_Total", "Fees_Honouring_Charges_Total", "Days_Balance_Below_Zero",
        "Min_Balance_Recorded", "Max_Balance_Recorded", "Net_Income", "Salary_Ratio",
        "Savings_Ratio", "Expense_Ratio", "Balance_Stability", "Fees_Total"
    ])
    score = model.predict(features)[0]
    rating, interpretation = get_interpretation(score)
    return f"Score: {round(score)}\nRating: {rating}\nInterpretation: {interpretation}"

# Randomizer
def randomize_inputs():
    return [
        round(random.uniform(5000, 40000), 2),  # Total Credits
        round(random.uniform(1000, 39000), 2),  # Total Debits
        round(random.uniform(3000, 30000), 2),  # Probable Salary Total
        round(random.uniform(500, 10000), 2),   # Transfers to Savings
        round(random.uniform(5, 50), 2),        # Monthly Account Fees
        round(random.uniform(10, 300), 2),      # Service Charges
        round(random.uniform(0, 50), 2),        # Penalty Interest
        round(random.uniform(0, 200), 2),       # Unpaid Item Charges
        round(random.uniform(0, 100), 2),       # Honouring Charges
        random.randint(0, 90),                  # Days Balance Below Zero
        round(random.uniform(-2000, 500), 2),   # Min Balance
        round(random.uniform(500, 20000), 2),   # Max Balance
        round(random.uniform(-5000, 10000), 2), # Net Income
        round(random.uniform(0.2, 0.9), 2),     # Salary Ratio
        round(random.uniform(0.0, 0.5), 2),     # Savings Ratio
        round(random.uniform(0.4, 1.2), 2),     # Expense Ratio
        round(random.uniform(1000, 15000), 2),  # Balance Stability
        round(random.uniform(0, 500), 2),       # Total Fees
    ]

# Gradio input layout
inputs = [
    gr.Number(label="Total Credits"),
    gr.Number(label="Total Debits"),
    gr.Number(label="Probable Salary Total"),
    gr.Number(label="Transfers to Savings Total"),
    gr.Number(label="Monthly Account Fees"),
    gr.Number(label="Service Charges"),
    gr.Number(label="Penalty Interest"),
    gr.Number(label="Unpaid Item Charges"),
    gr.Number(label="Honouring Charges"),
    gr.Number(label="Days Balance Below Zero"),
    gr.Number(label="Min Balance Recorded"),
    gr.Number(label="Max Balance Recorded"),
    gr.Number(label="Net Income"),
    gr.Number(label="Salary Ratio"),
    gr.Number(label="Savings Ratio"),
    gr.Number(label="Expense Ratio"),
    gr.Number(label="Balance Stability"),
    gr.Number(label="Total Fees")
]

# Launch app
with gr.Blocks() as demo:
    gr.Markdown("## Credit Score Regressor")
    gr.Markdown("Enter or randomize financial metrics to predict a numeric credit score and interpretation.")

    with gr.Row():
        input_components = [gr.Number(label=inp.label) for inp in inputs]
    
    predict_btn = gr.Button("Predict Credit Score")
    random_btn = gr.Button("🎲 Randomize Inputs")
    output = gr.Textbox(label="Prediction", lines=4)

    predict_btn.click(fn=predict_score, inputs=input_components, outputs=output)
    random_btn.click(fn=randomize_inputs, inputs=[], outputs=input_components)

if __name__ == "__main__":
    demo.launch()
