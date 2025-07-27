import pandas as pd
from pathlib import Path
from .config import settings

def load_csv() -> pd.DataFrame:
    path = settings.DATA_RAW_DIR / "Training Dataset.csv"
    df = pd.read_csv(path)
    return df

def row_to_text(row: pd.Series) -> str:
    # Create a readable natural language snippet
    return (
        "Loan Applicant Record:\n"
        f"Gender: {row.get('Gender', 'NA')}, Married: {row.get('Married','NA')}, "
        f"Dependents: {row.get('Dependents','NA')}, Education: {row.get('Education','NA')}, "
        f"Self_Employed: {row.get('Self_Employed','NA')}, ApplicantIncome: {row.get('ApplicantIncome','NA')}, "
        f"CoapplicantIncome: {row.get('CoapplicantIncome','NA')}, LoanAmount: {row.get('LoanAmount','NA')}, "
        f"Loan_Amount_Term: {row.get('Loan_Amount_Term','NA')}, Credit_History: {row.get('Credit_History','NA')}, "
        f"Property_Area: {row.get('Property_Area','NA')}, Loan_Status: {row.get('Loan_Status','NA')}"
    )

def build_documents():
    df = load_csv()
    docs = [row_to_text(r) for _, r in df.iterrows()]
    ids = [str(i) for i in range(len(docs))]
    return ids, docs
