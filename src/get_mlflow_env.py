import mlflow
import os
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def get_run_env_file(experiment_name="Apple_Models", 
                     run_name="first_run", 
                     output_dir="./src", 
                     port=8080):
    """
    Copy python_env.yaml, conda.yaml and requirements.txt directly from MLflow artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"http://127.0.0.1:{port}")
    client = mlflow.MlflowClient(tracking_uri="http://127.0.0.1:8080")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise Exception(f"Experiment {experiment_name} not found")

    logging.info(f"Found experiment: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )

    if len(runs) == 0:
        raise Exception(f"Run {run_name} not found in experiment {experiment_name}")

    # We take randomy the first one that looks being the latest ??
    run_id = runs.iloc[0].run_id
    logging.info(f"Found run: {run_name} (ID: {run_id})")

    # Get artifact URI
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri

    #NON CE N'EST PLUS VALIDE EN 3.12 ON NE RETROUVE PLUS LA STRUCTURE PHYSIQUE DIRECTEMENT
    print(artifact_uri)
    if artifact_uri.startswith("file://"):
        # Si on enlève les 7 premiers digits, c'est plus une uri, c'est un dossier posix
        artifact_fld = artifact_uri[7:]

    # Get the rf_apples directory path - WE WANT THE MODEL PATH RELATED TO THE RUN
    myrun = mlflow.get_run(run_id)
    
	# Récupérer les LoggedModels d'un run
    logged_models = client.search_logged_models(experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id  = '{run_id}'"
    )
    for model in logged_models:
        print(model.artifact_location)

    #rf_apples_path = os.path.join(artifact_uri, "rf_apples")
    rf_apples_path = model.artifact_location  #autre hack, si on veut le dossier on ajoutera [7:] 

    print(rf_apples_path)

    # Copy python_env.yaml
    python_env_src = os.path.join(rf_apples_path, "python_env.yaml")
    #ATTENTION, os.path.exists ne va pas pour les URI commencant par file://
    print(f" from {python_env_src[7:]} to {output_path}/python_env.yaml")

    if os.path.exists(python_env_src[7:]):
        shutil.copy2(python_env_src[7:], output_path / "python_env.yaml")
        logging.info(f"Copied python_env.yaml to {output_dir}")

    # Copy conda.yaml if it exists
    conda_src = os.path.join(rf_apples_path, "conda.yaml")

    if os.path.exists(conda_src[7:]):
        shutil.copy2(conda_src[7:], output_path / "conda.yaml")
        logging.info(f"Copied conda.yaml to {output_dir}")

    # Copy requirements.txt
    requirements_src = os.path.join(rf_apples_path, "requirements.txt")
    print(requirements_src)
    print(os.path.exists(requirements_src))
    zzz="/home/ubuntu/MLflow_Course/mlruns/747973836628435741/models/m-7dc0381361494690ac63395635d0424e/artifacts/requirements.txt"
    print(zzz)
    print(os.path.exists(zzz))
    if not os.path.exists(requirements_src[7:]):
        raise Exception("requirements.txt not found in artifacts")
    shutil.copy2(requirements_src[7:], output_path / "requirements.txt")
    logging.info(f"Copied requirements.txt to {output_dir}")

    # Check if at least one environment file was found
    if not os.path.exists(python_env_src[7:]) and not os.path.exists(conda_src[7:]):
        raise Exception("Neither python_env.yaml nor conda.yaml found in artifacts")

if __name__ == "__main__":
    import argparse
    # get_mlflow_env --experiment "Apple_Models" --run "first_run" --output="./src" --port 8080
    parser = argparse.ArgumentParser(description="Retrieve MLflow run environment files")
    parser.add_argument("--experiment", default="Apple_Models", help="Name of the MLflow experiment")
    parser.add_argument("--run"       , default="first_run"   , help="Name of the run")
    parser.add_argument("--output"    , default="./src"       , help="Output directory for the files")
    parser.add_argument("--port"      , type=int, default=8080, help="MLflow server port")

    args = parser.parse_args()

    get_run_env_file(
        experiment_name=args.experiment,
        run_name=args.run,
        output_dir=args.output,
        port=args.port
    )