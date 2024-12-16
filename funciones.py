# funciones.py

import argparse
import subprocess
import time
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def argumentos():
    """Función para obtener los argumentos de entrada del usuario"""
    parser = argparse.ArgumentParser(description='Aplicación con argumentos de entrada.')
    parser.add_argument('--nombre_job', type=str, help='Nombre del trabajo en MLflow.')
    parser.add_argument('--n_estimators_list', nargs='+', type=int, help='Lista de valores para n_estimators.')
    return parser.parse_args()

def load_dataset():
    """Carga el dataset preprocesado desde un archivo pickle."""
    df = pd.read_pickle("df_sentpreprocess.pkl")
    return df

def data_treatment(df):
    """Preprocesa los datos y los divide en entrenamiento y prueba."""
    cv = TfidfVectorizer(
        max_df=0.95,
        min_df=3,
        max_features=2500,
        strip_accents='ascii',
        ngram_range=(1, 2)
    )
    X_bow = cv.fit_transform(df['fullProcessedReview']).toarray()
    y = df['sentiment_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_bow, y, test_size=0.2, random_state=123, stratify=y
    )

    return X_train, X_test, y_train, y_test

# Modelos de RandomForest

def model_rf(X_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=2):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=123
    )
    preprocessor = Pipeline(steps=[('scaler', StandardScaler())])
    model = Pipeline(steps=[('preprocessor', preprocessor), ('RandomForestClassifier', clf)])
    model.fit(X_train, y_train)
    return model

def model_rf_2(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=150, min_samples_leaf=3, class_weight="balanced", random_state=123
    )
    model = Pipeline(steps=[('preprocessor', StandardScaler()), ('RandomForestClassifier', clf)])
    model.fit(X_train, y_train)
    return model

def model_rf_3(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_split=5, min_samples_leaf=3, class_weight="balanced", random_state=123
    )
    model = Pipeline(steps=[('preprocessor', StandardScaler()), ('RandomForestClassifier', clf)])
    model.fit(X_train, y_train)
    return model

def model_rf_4(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=6, min_samples_leaf=4, max_features='sqrt', class_weight="balanced", random_state=123
    )
    model = Pipeline(steps=[('preprocessor', StandardScaler()), ('RandomForestClassifier', clf)])
    model.fit(X_train, y_train)
    return model

def model_rf_5(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=250, max_depth=20, min_samples_split=2, min_samples_leaf=2, max_features='sqrt', class_weight="balanced", random_state=123
    )
    model = Pipeline(steps=[('preprocessor', StandardScaler()), ('RandomForestClassifier', clf)])
    model.fit(X_train, y_train)
    return model

def model_rf_grid(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=2, class_weight='balanced', random_state=123
    )
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3],
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def model_rf_cv(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=2, class_weight='balanced', random_state=123
    )
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    return cv_scores.mean()

# Modelo 8: GradientBoostingClassifier
def model_gb(X_train, y_train):
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=10, random_state=123
    )
    clf.fit(X_train, y_train)
    return clf

def mlflow_tracking(nombre_job, X_train, X_test, y_train, y_test, model, n_estimators=None):
    """Función para entrenar los modelos y registrar resultados en MLflow."""
    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui', '--port', '5000'])
    print(mlflow_ui_process)
    time.sleep(5)

    mlflow.set_experiment(nombre_job)

    with mlflow.start_run() as run:
        if n_estimators is not None:
            mlflow.log_param('n_estimators', n_estimators)

        accuracy_train = model.score(X_train, y_train)
        accuracy_test = model.score(X_test, y_test)
        mlflow.log_metric('accuracy_train', accuracy_train)
        mlflow.log_metric('accuracy_test', accuracy_test)

        mlflow.sklearn.log_model(model, 'random_forest_model')

    print("Modelo y métricas registrados en MLflow.")
