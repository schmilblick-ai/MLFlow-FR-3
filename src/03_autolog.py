import pandas as pd 
from sklearn import svm, datasets 
from sklearn.model_selection import GridSearchCV
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:8080")
experiment_id = mlflow.create_experiment("iris_Models")

mlflow.autolog()

with mlflow.start_run(experiment_id=experiment_id) as run:
    iris = datasets.load_iris()
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]} 
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(iris.data, iris.target)
    mlflow.log_params(parameters)
    #mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        sk_model=clf, input_example=iris.data, artifact_path="clf_iris"
    )