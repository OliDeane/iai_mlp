from sklearn import metrics 

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2_score = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)

    return mae, r2_score