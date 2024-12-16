# main.py

from funciones import argumentos, load_dataset, data_treatment, model_rf, model_rf_2, model_rf_3, model_rf_4, model_rf_5, model_rf_grid, model_rf_cv, model_gb, mlflow_tracking

def main():
    # Obtener los argumentos del trabajo
    print("Ejecutando el main...")
    args_values = argumentos()

    # Cargar el dataset
    df = load_dataset()

    # Preprocesar los datos y dividirlos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = data_treatment(df)

    # Entrenar y registrar los modelos con MLflow

    # Modelo 1
    model_1 = model_rf(X_train, y_train, n_estimators=args_values.n_estimators_list[0])
    mlflow_tracking(args_values.nombre_job, X_train, X_test, y_train, y_test, model_1, n_estimators=args_values.n_estimators_list[0])

    # Modelo 2
    model_2 = model_rf_2(X_train, y_train)
    mlflow_tracking(args_values.nombre_job, X_train, X_test, y_train, y_test, model_2)

    # Modelo 3
    model_3 = model_rf_3(X_train, y_train)
    mlflow_tracking(args_values.nombre_job, X_train, X_test, y_train, y_test, model_3)

    # Modelo 4
    model_4 = model_rf_4(X_train, y_train)
    mlflow_tracking(args_values.nombre_job, X_train, X_test, y_train, y_test, model_4)

    # Modelo 5
    model_5 = model_rf_5(X_train, y_train)
    mlflow_tracking(args_values.nombre_job, X_train, X_test, y_train, y_test, model_5)

    # Modelo 6 (con GridSearchCV)
    model_6 = model_rf_grid(X_train, y_train)
    mlflow_tracking(args_values.nombre_job, X_train, X_test, y_train, y_test, model_6)

    # Modelo 7 (con validación cruzada)
    cv_score_7 = model_rf_cv(X_train, y_train)
    print(f"Accuracy promedio de validación cruzada (Modelo 7): {cv_score_7}")

    # Modelo 8 (GradientBoostingClassifier)
    model_8 = model_gb(X_train, y_train)
    mlflow_tracking(args_values.nombre_job, X_train, X_test, y_train, y_test, model_8)

if __name__ == "__main__":
    main()
