import os
import logging
import mlflow
import mlflow.environment_variables
import dagshub
from main import ENV

class MLFlow:
    def __init__(self, username: str, token:str, exp: str, 
                 server: str, proj_name: str, env: ENV):
        self.token = token
        self.exp = exp
        self.server = server
        self.username = username
        self.env = env
        self.proj_name = proj_name

    def start(self):
        logging.info(
            f"MLFlow server is starting on {self.server.upper()} "
            f"with experiment {self.exp}..."
        )
        # Set mlflow env. variables
        for (var, val) in zip(self.env.vars, self.env.vals):
            os.environ[var] = str(val)
        if self.server == "local":
            self.start_local()
        else:
            self.start_dagshub()

    def start_local(self):
        mlflow.set_experiment(self.exp)
        mlflow.start_run()
        run = mlflow.active_run()
        if run is not None:
            logging.info(f"MLFlow run ID: {run.info.run_id}, status: {run.info.status}")

    def start_dagshub(self):
        dagshub.init(self.proj_name, self.username)
        mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = self.token
        mlflow.set_experiment(self.exp)
        mlflow.start_run()
        run = mlflow.active_run()
        logging.info(f"MLFlow run ID: {run.info.run_id}, status: {run.info.status}")
