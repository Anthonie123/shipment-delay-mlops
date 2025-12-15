import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import pickle
from pathlib import Path

DATA_PATH = "data/processed.csv"

def main():
    # Load data yang sudah diproses (sudah tanpa ID & sudah one-hot)
    df = pd.read_csv(DATA_PATH)

    # Pisahkan X dan y
    X = df.drop("Reached.on.Time_Y.N", axis=1)
    y = df["Reached.on.Time_Y.N"]

    # Split train-test sederhana
    split_idx = int(0.7 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # beberapa kombinasi hyperparameter untuk dicoba
    n_estimators_list = [50, 100, 200]
    max_depth_list = [3, 5, None]

    best_f1 = -1.0
    best_model = None
    best_params = {}

    mlflow.set_experiment("mlops-demo")

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            with mlflow.start_run():
                mlflow.log_param("model_type", "RandomForest")
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)

                # simpan juga ke MLflow sebagai artifact (opsional)
                mlflow.sklearn.log_model(model, "model")

                print(
                    f"[RUN] n_estimators={n_estimators}, max_depth={max_depth} "
                    f"-> Accuracy: {acc:.4f}, F1: {f1:.4f}"
                )

                # update best model
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                    }

    # setelah semua kombinasi dicoba, simpan model terbaik ke file .pkl
    if best_model is not None:
        print(
            f"\nBest model: n_estimators={best_params['n_estimators']}, "
            f"max_depth={best_params['max_depth']}, F1={best_f1:.4f}"
        )

        models_dir = Path("model/model_rf.pkl")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "model_rf.pkl"
        with model_path.open("wb") as f:
            pickle.dump(best_model, f)

if __name__ == "__main__":
    main()
