# Loan Default Prediction System

## Project Overview
This repository contains a machine learning project to predict whether a loan applicant will default (`Loan_Status = N`) or repay (`Loan_Status = Y`) based on historical data. Built as a self-paced internship project, it covers data preprocessing, exploratory data analysis (EDA), model building, and evaluation using Python in Google Colab.

## Dataset
- **File**: `loan_data.csv` (200 entries)
- **Features**: 
  - Categorical: Gender, Married, Dependents, Education, Self_Employed, Property_Area
  - Numerical: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
- **Target**: Loan_Status (Y/N)
- **Source**: Synthetic dataset provided for the project

## Project Structure
1. **Data Preprocessing**:
   - Remove `Loan_ID`.
   - Handle missing values (mode for categorical, mean for numerical).
   - Encode categorical features (LabelEncoder for binary, OneHotEncoder for multi-class).
   - Scale numerical features (StandardScaler).
2. **Exploratory Data Analysis (EDA)**:
   - Visualize distributions (e.g., LoanAmount, ApplicantIncome).
   - Analyze Loan_Status vs. features (e.g., Credit_History bar plot).
   - Examine correlations via heatmap.
3. **Model Building**:
   - Split data (80% train, 20% test).
   - Train Logistic Regression, Decision Tree, and Random Forest.
   - Evaluate using accuracy, confusion matrix, precision, recall, F1-score.
4. **Model Comparison**:
   - Compare model performance; Random Forest selected for robustness.
5. **Deliverables**:
   - Jupyter Notebook (`Loan_Default_Prediction.ipynb`) with code, plots, and markdown explanations.
   - Optional: PDF/HTML export, 1-page summary.

## Key Findings
- **Credit_History** strongly predicts Loan_Status (good credit → near-certain approval).
- Moderate correlation between ApplicantIncome and LoanAmount.
- Dataset is imbalanced (~90% approved loans), inflating accuracy.
- All models achieved perfect accuracy (1.00), likely due to data leakage from Credit_History.

## Repository Contents
- `loan_data.csv`: Dataset with 200 entries.
- `Loan_Default_Prediction.ipynb`: Jupyter Notebook with project code and documentation.

## How to Run
1. Open Google Colab[](https://colab.google).
2. Upload `loan_data.csv` and `Loan_Default_Prediction.ipynb`.
3. Run notebook cells sequentially to preprocess, analyze, and model data.
4. Download the `.ipynb` for submission (File > Download > Download .ipynb).
5. Optional: Export to PDF using Jupyter Notebook locally (requires LaTeX) or an online converter (e.g., nbviewer, pdfcrowd).

## Requirements
- **Environment**: Google Colab (no local installation needed)
- **Libraries**: pandas, numpy, scikit-learn, seaborn, matplotlib (pre-installed in Colab)

## Submission Deliverables
- **Required**: `Loan_Default_Prediction.ipynb` with commented code, EDA plots, and model results.
- **Optional**: PDF/HTML export of the notebook.
- **Optional**: 1-page summary (see notebook’s final markdown cell for template).
- **This Repository**: Share the GitHub link with your instructor.

## Notes
- Perfect accuracy (1.00) across models suggests data leakage, as `Credit_History` strongly predicts `Loan_Status`. Real-world datasets would likely show lower accuracy due to noise.
- For issues or questions, refer to the notebook’s markdown cells or contact the instructor.

## Author
- Created for Innovatecloud Solutions Machine Learning Trainee Internship by Raja Mehdi Ali Khan on July 14, 2025.
