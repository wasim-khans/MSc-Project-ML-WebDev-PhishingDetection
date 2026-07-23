# Experiment 4: Google trailing-slash brittleness check

Checks whether a small harmless URL text change can flip the model's prediction for an obvious legitimate website.

## Summary Table

| training_scenario      | model               | google_no_slash_prediction   | google_slash_prediction   | prediction_changed   |
|:-----------------------|:--------------------|:-----------------------------|:--------------------------|:---------------------|
| legitphish             | Logistic Regression | legitimate                   | legitimate                | False                |
| legitphish             | Linear SVM          | legitimate                   | legitimate                | False                |
| legitphish             | Decision Tree       | legitimate                   | legitimate                | False                |
| legitphish             | XGBoost             | legitimate                   | legitimate                | False                |
| combined_dataset       | Logistic Regression | legitimate                   | legitimate                | False                |
| combined_dataset       | Linear SVM          | legitimate                   | legitimate                | False                |
| legitphish             | Random Forest       | legitimate                   | legitimate                | False                |
| phishstorm             | Logistic Regression | legitimate                   | legitimate                | False                |
| phishstorm             | Linear SVM          | legitimate                   | legitimate                | False                |
| phishstorm             | XGBoost             | legitimate                   | legitimate                | False                |
| phishstorm             | Decision Tree       | phishing                     | phishing                  | False                |
| phishstorm             | Random Forest       | phishing                     | phishing                  | False                |
| main_baseline_phiusiil | Decision Tree       | legitimate                   | phishing                  | True                 |
| main_baseline_phiusiil | Logistic Regression | legitimate                   | phishing                  | True                 |
| main_baseline_phiusiil | Random Forest       | legitimate                   | phishing                  | True                 |
| main_baseline_phiusiil | XGBoost             | legitimate                   | phishing                  | True                 |
| phiusiil_main          | Decision Tree       | legitimate                   | phishing                  | True                 |
| phiusiil_main          | Logistic Regression | legitimate                   | phishing                  | True                 |
| phiusiil_main          | Random Forest       | legitimate                   | phishing                  | True                 |
| phiusiil_main          | XGBoost             | legitimate                   | phishing                  | True                 |
| combined_dataset       | Decision Tree       | legitimate                   | phishing                  | True                 |
| combined_dataset       | Random Forest       | legitimate                   | phishing                  | True                 |
| main_baseline_phiusiil | Linear SVM          | legitimate                   | phishing                  | True                 |
| phiusiil_main          | Linear SVM          | legitimate                   | phishing                  | True                 |
| combined_dataset       | XGBoost             | legitimate                   | phishing                  | True                 |
