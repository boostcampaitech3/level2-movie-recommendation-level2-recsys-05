from utils import dotdict
from utils import seed_everything

import yaml

from experiment import Experiment
from inference import Inference
import mlflow 
import mlflow.pytorch 

import nni
from nni.utils import merge_parameter

if __name__ == "__main__":
    tuner_params = None
    tuner_params = nni.get_next_parameter()

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
        
    margs = merge_parameter(margs, tuner_params)
    margs_artifacts = dict(margs)
    
    print("-" * 20, "model arguments list", "-" * 20)
    for k, v in margs.items():
        print(k, ":", v)
    print("-" * 50)

    seed_everything(args.seed)

    if args.mode == "experiment":
        experiment = Experiment(args, margs)
        
        # -- mlflow
        remote_server_uri = "http://101.101.211.226:30005"
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(args.experiment_name)
        mlflow_experiment = mlflow.get_experiment_by_name(args.experiment_name)
        client = mlflow.tracking.MlflowClient()
        
        run = client.create_run(mlflow_experiment.experiment_id)
        
        with mlflow.start_run(run_id=run.info.run_id): 
            mlflow.set_tag("mlflow.user", args.user) 
            mlflow.set_tag("mlflow.runName", args.run_name) 
            
            mlflow.log_params(dict(args))
            mlflow.log_params(dict(margs_artifacts))
            
            mlflow.log_dict(dict(args), args.run_name+"_args.yaml") 
            mlflow.log_dict(dict(margs_artifacts), args.run_name+"_margs.yaml")  
            
            experiment.run()
            
    elif args.mode == "inference":
        inference = Inference(args, margs)
        inference.run()