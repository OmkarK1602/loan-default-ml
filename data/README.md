# Data

`loan_default_finance_dataset.csv` - 2,200 **synthetic** loan applicant records (educational use only).

- Target: `Default` (Yes / No), about 20% Yes
- `ApplicantID` is only an identifier and is never used as a feature
- Missing values exist in `AnnualIncome` (22), `CreditScore` (18), `DebtToIncomeRatio` (20) and `PaymentHistory` (15)

| Column | Meaning |
|---|---|
| Age | Applicant age in years |
| EmploymentType | Salaried / Self-employed / Contract / Unemployed |
| AnnualIncome | Approximate yearly income (illustrative currency) |
| EmploymentYears | Years in employment |
| LoanAmount | Requested loan amount |
| LoanTermMonths | Loan duration in months |
| InterestRatePct | Illustrative interest rate |
| CreditScore | Simulated credit score |
| ExistingLoans | Number of existing loans |
| PastDelinquencies | Number of past delinquencies |
| DebtToIncomeRatio | Debt-to-income ratio (0 to 1) |
| LoanPurpose | Purpose category |
| HomeOwnership | Rent / Mortgage / Own / Family |
| Region | Broad region |
| BankRelationship | Length of relationship with the bank |
| PaymentHistory | Excellent / Good / Fair / Poor |
| Default | Simulated future default outcome |

This data is fake. Do not use it, or any model built from it, for real lending decisions.
