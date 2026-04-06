from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from data_preparation import generate_sets
from data_analysis_1 import evaluate_model

X_train, X_test, y_train, y_test = generate_sets()

scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

knn_std = KNeighborsClassifier()
svm_std = SVC(kernel='rbf', random_state=42)

print("Trenowanie modeli na danych po zastosowaniu standaryzacji")
knn_std.fit(X_train_std, y_train)
svm_std.fit(X_train_std, y_train)

evaluate_model("KNN (StandardScaler)", y_test, knn_std.predict(X_test_std))
evaluate_model("SVM (StandardScaler)", y_test, svm_std.predict(X_test_std))

# wytrenowanie modeli dla danych po MinMaxScaler
knn_minmax = KNeighborsClassifier()
svm_minmax = SVC(kernel='rbf', random_state=42)

print("Trenowanie modeli na danych przeskalowanych (MinMaxScaler)")
knn_minmax.fit(X_train_minmax, y_train)
svm_minmax.fit(X_train_minmax, y_train)

evaluate_model("KNN (MinMaxScaler)", y_test, knn_minmax.predict(X_test_minmax))
evaluate_model("SVM (MinMaxScaler)", y_test, svm_minmax.predict(X_test_minmax))