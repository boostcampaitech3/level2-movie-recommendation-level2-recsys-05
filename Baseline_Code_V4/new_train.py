from utils import dotdict
from utils import seed_everything

import yaml

from experiment import Experiment

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

    print("-" * 20, "model arguments list", "-" * 20)
    for k, v in margs.items():
        print(k, ":", v)
    print("-" * 50)

    seed_everything(args.seed)

    experiment = Experiment(args, margs)

    experiment.run()
