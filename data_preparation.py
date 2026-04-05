import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

def generate_sets():
    df = pd.read_csv('kampania.csv', sep='\t')

    print("Dane przed przygotowaniem:", df.shape)

    # uzupelnienie przychodu
    df['Income'] = df['Income'].fillna(df['Income'].median())

    # usuniecie stalych kolumn: Z_CostContact i Z_Revenue maja u wszystkich te same wartosci
    if 'Z_CostContact' in df.columns and 'Z_Revenue' in df.columns:
        df = df.drop(columns=['Z_CostContact', 'Z_Revenue'])

    # obczlienie wieku klienta
    current_year = 2026
    df['Age'] = current_year - df['Year_Birth']

    # usuniecie wartosci odstajacych (np. osob urodzonych w 1893 roku)
    df = df[df['Age'] < 100]

    # przeksztalcenie daty dolaczenia (Dt_Customer) na liczbowa: Liczba dni jako klient i zamiana tekstu na typ datetime
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
    # obliczenie roznicy w dniach (dni beda wzgledem najnowszej daty w bazie)
    latest_date = df['Dt_Customer'].max()
    df['Days_as_Customer'] = (latest_date - df['Dt_Customer']).dt.days

    # usuniecie pierwotnych kolumn
    df = df.drop(columns=['ID', 'Year_Birth', 'Dt_Customer'])

    # zamiana kolumn 'Education' i 'Marital_Status' na format bool'owy
    categorical_features = ['Education', 'Marital_Status']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Oddzielenie cech (X) od zmiennej celu (y)
    X = df.drop(columns=['Response'])
    y = df['Response']

    # Podzial na zbior treningowy (80%) i testowy (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Dane po przygotowaniu:", df.shape)
    print("Wymiary X_train:", X_train.shape)
    print("Wymiary X_test:", X_test.shape)

    return X_train, X_test, y_train, y_test

generate_sets()
