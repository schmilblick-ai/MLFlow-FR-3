import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
from scipy.stats import randint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_prep_data(data_path: str):
    """Load and prepare data for training."""
    # Implémentez cette fonction
    # Import Database
    data = pd.read_csv("data/fake_data.csv")
    X = data.drop(columns=["date", "demand"])
    X = X.astype('float')
    y = data["demand"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Configuration de base
    EXPERIMENT_NAME = "Apple_Models1"
    N_TRIALS = 5
    # Configurez MLflow
    # Implémentez cette partie

    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Handle experiment creation/deletion
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    #mechanism of allocation is due

    if experiment and experiment.lifecycle_stage == 'deleted':
        # If experiment exists but is deleted, create a new one with timestamp; mandatory because still exists in the .trash folder
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        EXPERIMENT_NAME = f"{EXPERIMENT_NAME}_{timestamp}"
        client.create_experiment(EXPERIMENT_NAME)
    elif experiment is None:
        # If experiment doesn't exist, create it
        client.create_experiment(EXPERIMENT_NAME)

    mlflow.set_experiment(EXPERIMENT_NAME)

    #mlflow.autolog()
    mlflow.sklearn.autolog(log_models=True)
  
    # Chargez les données
    X_train, X_val, y_train, y_val = load_and_prep_data("path_to_your_data.csv")

    # Définissez l'espace de recherche des hyperparamètres
    param_dist = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],   # bonus : évite l'overfitting
        "max_features": ["sqrt", "log2", 0.5],  # bonus : diversité des arbres
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4),
    }

    # Créez et exécutez RandomizedSearchCV
    # Implémentez cette partie
    # En régression → minimiser la variance des deux sous-nœuds (MSE)
    # En classification → minimiser le Gini ou maximiser le gain d'information
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
            param_distributions=param_dist,
            n_iter=N_TRIALS,          # nombre de combinaisons testées
            cv=5,               # 5-fold cross-validation
            #scoring="neg_mean_squared_error",
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=42
        )

    search.fit(X_train, y_train)
    # Récupérez les informations sur le meilleur modèle
    # Implémentez cette partie
    best_params = search.best_params_
    print(best_params )

    # Find the best run from MLflow
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],
        filter_string="",
        max_results=50
    )

    # Identify the parent and its best run parameters
    parent_run = None
    for run in runs:
        if 'best_n_estimators' in run.data.params:  # Parent run has the best_ parameters
            parent_run = run
            break

    best_run = None

    if parent_run:
        # Extract best parameters from parent run
        best_params_from_parent = {
            'n_estimators': parent_run.data.params['best_n_estimators'],
            'max_depth': parent_run.data.params['best_max_depth'],
            'min_samples_split': parent_run.data.params['best_min_samples_split'],
            'min_samples_leaf': parent_run.data.params['best_min_samples_leaf']
        }

        # Find the child run with these parameters
        best_run = None
        for run in runs:
            if ('n_estimators' in run.data.params and
                run.data.params['n_estimators'] == best_params_from_parent['n_estimators'] and
                run.data.params['max_depth'] == best_params_from_parent['max_depth'] and
                run.data.params['min_samples_split'] == best_params_from_parent['min_samples_split'] and
                run.data.params['min_samples_leaf'] == best_params_from_parent['min_samples_leaf']):
                best_run = run
                break

    best_run_name = best_run.data.tags.get('mlflow.runName', 'Not found') if best_run else 'Not found'

    # Create a summary of results with better formatting
    summary = f"""Random Forest Trials Summary:
---------------------------
🏆 Best Experiment Name: {EXPERIMENT_NAME}
🎯 Best Run Name: {best_run_name}

Best Model Parameters:
🌲 Number of Trees: {search.best_params_['n_estimators']}
📏 Max Tree Depth: {search.best_params_['max_depth']}
📎 Min Samples Split: {search.best_params_['min_samples_split']}
🍂 Min Samples Leaf: {search.best_params_['min_samples_leaf']}
📊 Best CV Score: {search.best_score_:.4f}
"""

    # Log summary to the parent run
    with mlflow.start_run(run_id=parent_run.info.run_id):

        # Log summary as an artifact
        with open("summary.txt", "w") as f:
            f.write(summary)
        mlflow.log_artifact("summary.txt")

if __name__ == "__main__":
    main()