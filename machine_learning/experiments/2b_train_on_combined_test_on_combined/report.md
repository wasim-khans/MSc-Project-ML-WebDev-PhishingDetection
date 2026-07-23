# Experiment 2b: Combined-trained models tested against the combined held-out dataset

Each model is trained on the combined training dataset and tested on the combined held-out test set.

## Winner

- Model: `XGBoost`
- Trained on: `combined_dataset`
- Phishing F1: `0.9780`

## Summary Table

| tested_on     | model               |   rows_tested |   accuracy |   phishing_precision |   phishing_recall |   phishing_f1 |
|:--------------|:--------------------|--------------:|-----------:|---------------------:|------------------:|--------------:|
| combined_test | XGBoost             |         72667 |   0.974789 |             0.985015 |          0.971052 |      0.977983 |
| combined_test | Random Forest       |         72667 |   0.974638 |             0.982743 |          0.973104 |      0.9779   |
| combined_test | Decision Tree       |         72667 |   0.968624 |             0.97885  |          0.966469 |      0.972621 |
| combined_test | Logistic Regression |         72667 |   0.902679 |             0.953421 |          0.87392  |      0.911941 |
| combined_test | Linear SVM          |         72667 |   0.896156 |             0.95536  |          0.860102 |      0.905232 |
