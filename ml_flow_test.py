import mlflow
from mlflow.tracking import MlflowClient
EXPERIMENT_NAME = "[SG] [SG] [alejandro] awesome regression + v99"

# Indicate mlflow to log to remote server
mlflow.set_tracking_uri("https://mlflow.lewagon.co/")

client = MlflowClient()
try:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

yourname = 'Alejandro'

if yourname is None:
    print("please define your name, il will be used as a parameter to log")

for model in ["linear"]:
    run = client.create_run(experiment_id)
    client.log_metric(run.info.run_id, "rmse", 6.544532358049879)
    client.log_param(run.info.run_id, "model", model)
    client.log_param(run.info.run_id, "student_name", yourname)
