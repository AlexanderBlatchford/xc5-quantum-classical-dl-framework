from sklearn.svm import SVC

def train_svm(X_train, y_train, X_test, y_test):
    model = SVC()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds