from utils import dotdict
from utils import seed_everything

import yaml

from experiment import Experiment
import mlflow 
import mlflow.pytorch 

if __name__ == "__main__":

    args = None
    with open("train_config.yaml") as f:
        tmp_args = yaml.load(f, Loader=yaml.FullLoader)
        args = dotdict(tmp_args)

    if args is None:
        print("could not found config.yaml file")
        exit()

    print("-" * 20, "experiment arguments list", "-" * 20)
    for k, v in args.items():
        print(k, ":", v)

    margs = None
    with open(f"./model_config/{args.model_config}.yaml") as f:
        tmp_args = yaml.load(f, Loader=yaml.FullLoader)
        margs = dotdict(tmp_args)
    margs_artifacts = dict(margs)
    
    print("-" * 20, "model arguments list", "-" * 20)
    for k, v in margs.items():
        print(k, ":", v)
    print("-" * 50)

    seed_everything(args.seed)

    experiment = Experiment(args, margs)

    # -- mlflow
    remote_server_uri = "http://101.101.211.226:30005"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow_experiment = mlflow.get_experiment_by_name(args.experiment_name)
    client = mlflow.tracking.MlflowClient()
    
    run = client.create_run(mlflow_experiment.experiment_id)
    
    with mlflow.start_run(run_id=run.info.run_id): 
        mlflow.set_tag("mlflow.user", args.user) 
        mlflow.set_tag("mlflow.runName", args.run_name) 
        
        mlflow.log_dict(dict(args), args.run_name+"_args.yaml") 
        mlflow.log_dict(dict(margs_artifacts), args.run_name+"_margs.yaml")  
        
        experiment.run()
