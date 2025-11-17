# CSC311-Introduction-to-Machine-Learning-2025Fall-Project
CSC311 Introduction to Machine Learning Project 2025 Fall

**Logistic_Regression:**

Iter1_knn: From Template Code
    Training accuracy: 0.727
    Test accuracy:     0.611

Iter2_logreg: Apply logistic regression on baseline
    Training accuracy (LogReg): 0.628
    Test accuracy (LogReg):     0.710

Iter3_logreg: previous + Split by student_id to avoid data leakage
    Training accuracy (LogReg, student-wise split): 0.673
    Test accuracy (LogReg, student-wise split):     0.619

Iter4_logreg: previous + Use 3 for missing ratings (avoid loss of data)
    Training accuracy (LogReg, student-wise split, imputed): 0.639
    Test accuracy (LogReg, student-wise split, imputed):     0.663

Iter5_logreg: previous + Consider all features (all ratings & bag of words) (max_feature for # of text = 2000)
    Training accuracy (full features): 0.984
    Test accuracy (full features):     0.635

Iter6_logreg: previous + try to find best hyperparameter C (best is C=0.1) (max_feature for # of text = 3000)
    Training accuracy: 0.958
    Test accuracy:     0.683