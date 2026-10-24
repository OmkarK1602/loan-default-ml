"""Streamlit demo - run with:  streamlit run app.py"""

import streamlit as st

from src import data_preprocessing as dp
from src.predict import load_model, predict_one

st.set_page_config(page_title="Loan Default Risk (Demo)", page_icon="🏦")

st.title("🏦 Loan Default Risk Estimator")
st.warning(
    "CLASSROOM DEMONSTRATION ONLY. The model is trained on a synthetic dataset. "
    "It must NOT be used for real credit decisions, lending, underwriting or financial advice."
)


@st.cache_resource
def get_model():
    return load_model()


try:
    model, source = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
st.caption(f"Model loaded from: {source}")

# ---- input form ----
st.subheader("Applicant details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 40)
    income = st.number_input("Annual income", 0, 10_000_000, 70000, step=1000)
    emp_years = st.number_input("Employment years", 0, 60, 10)
    loan_amount = st.number_input("Loan amount", 1, 100_000_000, 400000, step=10000)
    term = st.number_input("Loan term (months)", 1, 480, 60)
    rate = st.number_input("Interest rate (%)", 0.0, 100.0, 12.0, step=0.1)
    credit_score = st.number_input("Credit score", 300, 900, 680)
    existing = st.number_input("Existing loans", 0, 50, 1)
    delinq = st.number_input("Past delinquencies", 0, 50, 0)
    dti = st.slider("Debt-to-income ratio", 0.0, 1.0, 0.30, 0.01)
with col2:
    emp_type = st.selectbox("Employment type", ["Salaried", "Self-employed", "Contract", "Unemployed"])
    purpose = st.selectbox("Loan purpose", ["Vehicle", "Personal", "Education", "Home Improvement", "Medical", "Business"])
    home = st.selectbox("Home ownership", ["Rent", "Mortgage", "Own", "Family"])
    region = st.selectbox("Region", ["Central", "North", "West", "South", "East"])
    bank = st.selectbox("Bank relationship", ["New", "1-3 years", "3-7 years", "7+ years"])
    history = st.selectbox("Payment history", ["Excellent", "Good", "Fair", "Poor"], index=1)
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Lower = catches more defaulters but flags more good borrowers")

applicant = {
    "Age": age, "EmploymentType": emp_type, "AnnualIncome": income,
    "EmploymentYears": emp_years, "LoanAmount": loan_amount, "LoanTermMonths": term,
    "InterestRatePct": rate, "CreditScore": credit_score, "ExistingLoans": existing,
    "PastDelinquencies": delinq, "DebtToIncomeRatio": dti, "LoanPurpose": purpose,
    "HomeOwnership": home, "Region": region, "BankRelationship": bank,
    "PaymentHistory": history,
}

if st.button("Estimate default risk"):
    try:
        result = predict_one(model, applicant, threshold)
    except ValueError as e:
        st.error(f"Invalid input: {e}")
    else:
        st.metric("Estimated default probability", f"{result['probability'] * 100:.1f}%")
        st.write(f"**Risk band:** {result['risk_band']}")
        st.write(f"**Prediction at threshold {threshold}:** {result['prediction']}")
        st.info("This is a rough estimate from a weak model on made-up data. "
                "Treat it only as a learning example.")
