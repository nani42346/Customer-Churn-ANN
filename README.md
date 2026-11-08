# Customer Churn Prediction Using an Artificial Neural Network (ANN)

Predicting which customers are likely to churn, using a TensorFlow / Keras ANN, and turning the findings into retention recommendations.

## Contents

| File | Description |
|---|---|
| `koteswara_rao_customer_churn_task_ann.ipynb` | Full notebook: 10 phases from data understanding to business recommendations |
| `cleaned_customerchurn_dataset.csv` | Dataset: 180 customers, 13 columns |
| `Customer_Churn_ANN_Presentation.pptx` | 15-slide presentation of the project |

## Approach

1. **Data understanding:** 180 customers, no missing values or duplicates, 52% churn
2. **Preprocessing:** drop `Customer_ID`, encode `Churn` as 1/0
3. **EDA:** churn rate by contract type, satisfaction, plan type and data usage
4. **Splitting:** stratified 70 / 15 / 15 (126 / 27 / 27)
5. **Feature transformation:** one-hot encoding + standard scaling, fitted on training data only (22 features)
6. **ANN architecture:** 22 → Dense 32 (ReLU) → Dense 16 (ReLU) → 1 (Sigmoid)
7. **Training:** Adam, binary crossentropy, 50 epochs
8. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix
9. **Improvement:** dropout, early stopping, and 12 configurations compared with stratified 5-fold cross-validation
10. **Business recommendations:** permutation feature importance and churn-reduction strategies

## Key Results

- The baseline ANN overfits: 92.9% training accuracy vs 48.1% validation accuracy.
- The best tuned ANN (32, 16 hidden units, dropout 0.3, learning rate 0.01) reaches a **cross-validated ROC-AUC of about 0.67**.
- Test-set scores (27 customers) are close to chance, which shows how much a dataset this small limits the model.
- **Main churn drivers:** contract type (monthly 62.5% churn vs 2-year 20.8%), satisfaction score, data usage and plan type.

## Getting Started

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
jupyter notebook koteswara_rao_customer_churn_task_ann.ipynb
```

## Author

D. Koteswara Rao
