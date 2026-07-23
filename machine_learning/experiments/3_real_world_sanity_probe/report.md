# Experiment 3: 40-URL real-world sanity probe

Saved models are tested on 20 normal websites and 20 held-out phishing URLs to inspect false positives and missed phishing.

## Winner

- Model: `Logistic Regression`
- Trained on: `legitphish`
- Phishing F1: `1.0000`

## Summary Table

|   rank | training_scenario      | model               |   accuracy |   phishing_precision |   phishing_recall |   phishing_f1 |   total_correct |   legit_false_positives |   phishing_false_negatives | google_no_slash_prediction   | google_slash_prediction   |
|-------:|:-----------------------|:--------------------|-----------:|---------------------:|------------------:|--------------:|----------------:|------------------------:|---------------------------:|:-----------------------------|:--------------------------|
|      1 | legitphish             | Logistic Regression |      1     |             1        |              1    |      1        |              40 |                       0 |                          0 | legitimate                   | legitimate                |
|      2 | legitphish             | Linear SVM          |      0.975 |             1        |              0.95 |      0.974359 |              39 |                       0 |                          1 | legitimate                   | legitimate                |
|      3 | legitphish             | Decision Tree       |      0.95  |             0.909091 |              1    |      0.952381 |              38 |                       2 |                          0 | legitimate                   | legitimate                |
|      4 | legitphish             | XGBoost             |      0.95  |             0.909091 |              1    |      0.952381 |              38 |                       2 |                          0 | legitimate                   | legitimate                |
|      5 | combined_dataset       | Logistic Regression |      0.925 |             1        |              0.85 |      0.918919 |              37 |                       0 |                          3 | legitimate                   | legitimate                |
|      6 | combined_dataset       | Linear SVM          |      0.9   |             1        |              0.8  |      0.888889 |              36 |                       0 |                          4 | legitimate                   | legitimate                |
|      7 | legitphish             | Random Forest       |      0.9   |             0.833333 |              1    |      0.909091 |              36 |                       4 |                          0 | legitimate                   | legitimate                |
|      8 | phishstorm             | Logistic Regression |      0.875 |             1        |              0.75 |      0.857143 |              35 |                       0 |                          5 | legitimate                   | legitimate                |
|      9 | phishstorm             | Linear SVM          |      0.825 |             1        |              0.65 |      0.787879 |              33 |                       0 |                          7 | legitimate                   | legitimate                |
|     10 | phishstorm             | XGBoost             |      0.675 |             0.652174 |              0.75 |      0.697674 |              27 |                       8 |                          5 | legitimate                   | legitimate                |
|     11 | phishstorm             | Decision Tree       |      0.575 |             0.548387 |              0.85 |      0.666667 |              23 |                      14 |                          3 | phishing                     | phishing                  |
|     12 | phishstorm             | Random Forest       |      0.55  |             0.53125  |              0.85 |      0.653846 |              22 |                      15 |                          3 | phishing                     | phishing                  |
|     13 | main_baseline_phiusiil | Decision Tree       |      0.525 |             0.512821 |              1    |      0.677966 |              21 |                      19 |                          0 | legitimate                   | phishing                  |
|     14 | main_baseline_phiusiil | Logistic Regression |      0.525 |             0.512821 |              1    |      0.677966 |              21 |                      19 |                          0 | legitimate                   | phishing                  |
|     15 | main_baseline_phiusiil | Random Forest       |      0.525 |             0.512821 |              1    |      0.677966 |              21 |                      19 |                          0 | legitimate                   | phishing                  |
|     16 | main_baseline_phiusiil | XGBoost             |      0.525 |             0.512821 |              1    |      0.677966 |              21 |                      19 |                          0 | legitimate                   | phishing                  |
|     17 | phiusiil_main          | Decision Tree       |      0.525 |             0.512821 |              1    |      0.677966 |              21 |                      19 |                          0 | legitimate                   | phishing                  |
|     18 | phiusiil_main          | Logistic Regression |      0.525 |             0.512821 |              1    |      0.677966 |              21 |                      19 |                          0 | legitimate                   | phishing                  |
|     19 | phiusiil_main          | Random Forest       |      0.525 |             0.512821 |              1    |      0.677966 |              21 |                      19 |                          0 | legitimate                   | phishing                  |
|     20 | phiusiil_main          | XGBoost             |      0.525 |             0.512821 |              1    |      0.677966 |              21 |                      19 |                          0 | legitimate                   | phishing                  |
|     21 | combined_dataset       | Decision Tree       |      0.5   |             0.5      |              0.9  |      0.642857 |              20 |                      18 |                          2 | legitimate                   | phishing                  |
|     22 | combined_dataset       | Random Forest       |      0.5   |             0.5      |              0.9  |      0.642857 |              20 |                      18 |                          2 | legitimate                   | phishing                  |
|     23 | main_baseline_phiusiil | Linear SVM          |      0.5   |             0.5      |              0.95 |      0.655172 |              20 |                      19 |                          1 | legitimate                   | phishing                  |
|     24 | phiusiil_main          | Linear SVM          |      0.5   |             0.5      |              0.95 |      0.655172 |              20 |                      19 |                          1 | legitimate                   | phishing                  |
|     25 | combined_dataset       | XGBoost             |      0.475 |             0.485714 |              0.85 |      0.618182 |              19 |                      18 |                          3 | legitimate                   | phishing                  |
