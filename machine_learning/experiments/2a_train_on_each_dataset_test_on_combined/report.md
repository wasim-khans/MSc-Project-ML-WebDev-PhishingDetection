# Experiment 2a: Single-source models tested against the combined held-out dataset

Each model is trained on one source dataset and tested on the mixed combined test set.

## Winner

- Model: `Linear SVM`
- Trained on: `phiusiil_main`
- Phishing F1: `0.9070`

## Summary Table

| tested_on     | model               |   rows_tested |   accuracy |   phishing_precision |   phishing_recall |   phishing_f1 |
|:--------------|:--------------------|--------------:|-----------:|---------------------:|------------------:|--------------:|
| combined_test | Linear SVM          |         78982 |   0.895305 |             0.865675 |          0.952416 |      0.906976 |
| combined_test | XGBoost             |         78982 |   0.874908 |             0.812856 |          0.995842 |      0.895092 |
| combined_test | Random Forest       |         78982 |   0.87468  |             0.812525 |          0.995936 |      0.89493  |
| combined_test | Decision Tree       |         78982 |   0.874642 |             0.812502 |          0.995889 |      0.894897 |
| combined_test | Logistic Regression |         78982 |   0.872882 |             0.812313 |          0.991991 |      0.893205 |
| combined_test | Random Forest       |         78876 |   0.863875 |             0.810095 |          0.973218 |      0.884196 |
| combined_test | Decision Tree       |         78876 |   0.848344 |             0.805495 |          0.943919 |      0.869231 |
| combined_test | XGBoost             |         78876 |   0.837847 |             0.802076 |          0.92445  |      0.858927 |
| combined_test | Logistic Regression |         78876 |   0.797023 |             0.788117 |          0.847809 |      0.816874 |
| combined_test | Linear SVM          |         78876 |   0.779477 |             0.781377 |          0.815067 |      0.797866 |
| combined_test | Logistic Regression |         84759 |   0.769901 |             0.797041 |          0.72025  |      0.756702 |
| combined_test | XGBoost             |         84759 |   0.713612 |             0.657701 |          0.883208 |      0.753953 |
| combined_test | Linear SVM          |         84759 |   0.770656 |             0.80997  |          0.703389 |      0.752927 |
| combined_test | Random Forest       |         84759 |   0.59635  |             0.557158 |          0.913914 |      0.692277 |
| combined_test | Decision Tree       |         84759 |   0.594002 |             0.559721 |          0.856563 |      0.677034 |
