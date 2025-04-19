from ucimlrepo import fetch_ucirepo
import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train():
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    model_path =  'app\\models\\random_forest_model.pkl'

    joblib.dump(rfc, model_path) 

if __name__ == '__main__':
    train()