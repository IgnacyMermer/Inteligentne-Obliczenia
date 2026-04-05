from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_preparation import generate_sets

X_train, X_test, y_train, y_test = generate_sets()

knn_model = KNeighborsClassifier()
svm_model = SVC(kernel='rbf', random_state=42)

print("Trenowanie modelu KNN")
knn_model.fit(X_train, y_train)

print("Trenowanie modelu SVM")
svm_model.fit(X_train, y_train)

# wykonywanie predykcji na zbiorze testowym
knn_preds = knn_model.predict(X_test)
svm_preds = svm_model.predict(X_test)

def evaluate_model(model_name, y_true, y_pred):
    print(f"Wyniki dla modelu: {model_name}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision:{precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:   {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("Macierz błędów:")
    print(confusion_matrix(y_true, y_pred))

evaluate_model("KNN", y_test, knn_preds)
evaluate_model("SVM", y_test, svm_preds)