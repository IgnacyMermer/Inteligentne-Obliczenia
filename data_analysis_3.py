import warnings
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from data_preparation import generate_sets
from data_analysis_1 import evaluate_model

warnings.filterwarnings('ignore')

strategies = ['uniform', 'quantile', 'kmeans']

X_train, X_test, y_train, y_test = generate_sets()

for strategy in strategies:
    print(f"Strategia dyskretyzacji: {strategy.upper()}")
    
    # inicjalizacja dyskretyzatora (5 koszyków)
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy=strategy)
    
    # Skaler jest uczony na zbiorze treningowym!
    X_train_binned = discretizer.fit_transform(X_train)
    X_test_binned = discretizer.transform(X_test)
    
    knn_binned = KNeighborsClassifier()
    svm_binned = SVC(kernel='rbf', random_state=42)
    
    knn_binned.fit(X_train_binned, y_train)
    svm_binned.fit(X_train_binned, y_train)
    
    evaluate_model(f"KNN (Binning - {strategy})", y_test, knn_binned.predict(X_test_binned))
    evaluate_model(f"SVM (Binning - {strategy})", y_test, svm_binned.predict(X_test_binned))