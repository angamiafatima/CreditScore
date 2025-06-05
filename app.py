import gradio as gr
import joblib
import pandas as pd

# Load your trained model
model = joblib.load("credit_score_classifier.pkl")

# Define input fields
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
    pred = model.predict(features)[0]
    return ["Low", "Medium", "High"][pred]

# Gradio interface
iface = gr.Interface(
    fn=predict_score,
    inputs=[
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
    ],
    outputs="text",
    title="Credit Score Classifier",
    description="Upload or enter your financial behavior metrics to predict a credit score class (Low, Medium, High)."
)

if __name__ == "__main__":
    iface.launch()
