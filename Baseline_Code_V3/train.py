from models.MultiDAE import MultiDAE
from models.MultiVAE import MultiVAE
from models.AutoRec import AutoRec
from models.RecVAE import RecVAE
from models.EASE import EASE

# from models.EASE import EASE
from preprocessing.MakeMatrixDataSet import MakeMatrixDataSet, AEDataSet
from trainers.Matrix import evaluate, train as matrix_train
from trainers.Matrix import evaluate as matrix_evaluate
from trainers.Ensemble import total_evaluate

import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
import argparse

from box import Box

import mlflow
import mlflow.pytorch

import nni
from nni.utils import merge_parameter


def matrix_main(config, model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    make_matrix_data_set = MakeMatrixDataSet(config=config)
    user_valid = make_matrix_data_set.get_train_valid_data()[1]

    ae_dataset = AEDataSet(
        num_user=make_matrix_data_set.num_user,
    )

    data_loader = DataLoader(
        ae_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
    )

    if model_type == "MultiVAE":
        model = MultiVAE(
            p_dims=config.p_dims + [make_matrix_data_set.num_item],
            anneal_cap=config.anneal_cap,
            total_anneal_steps=config.total_anneal_steps,
            dropout_rate=config.dropout_rate,
        ).to(device)

    elif model_type == "MultiDAE":
        model = MultiDAE(
            p_dims=config.p_dims + [make_matrix_data_set.num_item],
            dropout_rate=config.dropout_rate,
        ).to(device)

    elif model_type == "AutoRec":
        model = AutoRec(
            input_dim=make_matrix_data_set.num_item, hidden_dim=config.hidden_dim
        ).to(device)

    elif model_type == "RecVAE":
        model = RecVAE(
            input_dim=make_matrix_data_set.num_item,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            beta=config.beta,
            gamma=config.gamma,
            dropout_rate=config.dropout_rate,
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_hit = 0
    for epoch in range(1, config.num_epochs + 1):
        train_loss = matrix_train(
            model=model,
            optimizer=optimizer,
            data_loader=data_loader,
            make_data_set=make_matrix_data_set,
        )

        ndcg, hit = matrix_evaluate(
            model=model,
            data_loader=data_loader,
            user_valid=user_valid,
            make_data_set=make_matrix_data_set,
        )

        if best_hit < hit:
            best_hit = hit
            torch.save(
                model.state_dict(), os.path.join(config.model_path, config.model_name)
            )
            mlflow.pytorch.log_model(model, 'bestModel')

        print(
            f"Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}"
        )
        
        mlflow.log_metric("Train_loss", train_loss, epoch)
        mlflow.log_metric("NDCG-10", ndcg, epoch)
        mlflow.log_metric("HIT-10", hit, epoch)
    
        nni.report_final_result(hit)   

def ensemble_main(config): #model_type
    device = "cuda" if torch.cuda.is_available() else "cpu"

    make_matrix_data_set = MakeMatrixDataSet(config = config)
    user_train, user_valid = make_matrix_data_set.get_train_valid_data()
    X = make_matrix_data_set.make_sparse_matrix()
    
    model1 = EASE(X = X, reg = 750)
    model1.fit()
    
    model2 = EASE(X = X.T, reg = 4400)
    model2.fit()
    
    model3 = RecVAE(
    input_dim = make_matrix_data_set.num_item,).to(device)

    model3.load_state_dict(torch.load(os.path.join(config.model_path, 'RecVAE_v3.pt')))
    
    model4 = AutoRec(
    input_dim = make_matrix_data_set.num_item, 
    hidden_dim= 64).to(device)

    model4.load_state_dict(torch.load(os.path.join(config.model_path, 'AutoRec_v1.pt')))
    
    model5 = MultiDAE(
    p_dims = [100, 200, 400] + [make_matrix_data_set.num_item], 
    dropout_rate = 0.5).to(device)

    model5.load_state_dict(torch.load(os.path.join(config.model_path, 'Multi-DAE_v1.pt')))
    
    model6 = MultiVAE(
    p_dims = [100, 200, 400] + [make_matrix_data_set.num_item], 
    dropout_rate = 0.5).to(device)

    model6.load_state_dict(torch.load(os.path.join(config.model_path, 'Multi-VAE_v1.pt')))
    
    df = total_evaluate(
        model1 = model1,
        model2 = model2,
        RecVAE = model3,
        AutoRec = model4,
        MultiDAE = model5,
        MultiVAE = model6,
        X = X.todense(),
        user_train = user_train,
        user_valid = user_valid,
        candidate_cnt = 10,
        )

    # 유저들 마다 rec1 or rec2 or rec3 or ranking 등 맞는 방법에 따라사 추천을 해주는 것도 좋은 방법이 될 수 있음

    new_df = pd.DataFrame(df)

    def get_total_name(x):
        val_list = [x['rec1'], x['rec2'], x['rec3'], x['rec4'], x['rec5'] , x['rec6']]
        max_val = max(val_list)
        val_idx = val_list.index(max_val)
        if val_idx == 0 : return 'rec1'
        elif val_idx == 1 : return 'rec2'
        elif val_idx == 2 : return 'rec3'
        elif val_idx == 3 : return 'rec4'
        elif val_idx == 4 : return 'rec5'
        elif val_idx == 5 : return 'rec6'

    new_df['total_val'] = new_df.apply(lambda x: max(x['rec1'], x['rec2'], x['rec3'], x['rec4'], x['rec5'] , x['rec6']), axis = 1)
    new_df['total_name'] = new_df.apply(lambda x: get_total_name(x), axis = 1)


    candidate_cnt = 30

    ndcg, hit = evaluate(
                model1 = model1, 
                model2 = model2, 
                RecVAE = model3,
                AutoRec = model4,
                MultiDAE = model5,
                MultiVAE = model6,
                X = X.todense(),
                user_train = user_train, 
                user_valid = user_valid, 
                candidate_cnt = candidate_cnt)

    print(f'candidate_cnt: {candidate_cnt}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')

    for candidate_cnt in [5 * i for i in range(2, 21)]:
        
        ndcg, hit = evaluate(
                    model1 = model1,
                    model2 = model2, 
                    RecVAE = model3,
                    AutoRec = model4,
                    MultiDAE = model5,
                    MultiVAE = model6,
                    X = X.todense(),
                    user_train = user_train, 
                    user_valid = user_valid, 
                    candidate_cnt = candidate_cnt)

        print(f'candidate_cnt: {candidate_cnt}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')
        
        # mlflow.log_metric("Train_loss", train_loss, epoch)
        mlflow.log_metric("NDCG-10", ndcg, candidate_cnt)
        mlflow.log_metric("HIT-10", hit, candidate_cnt)
    
        nni.report_final_result(hit)   


        
if __name__ == "__main__":

    config = {
        # ensemble
        'data_path' : "/opt/ml/input/data/train" , # 데이터 경로
        'model_path' : "./model",

        'submission_path' : "./submission",
        'submission_name' : 'Ensembel_v5_submission.csv',

        'candidate_item_num' : 50,
        'valid_samples' : 10, # 검증에 사용할 sample 수
        'seed' : 22,
        # 공통
        # "data_path": "/opt/ml/input/data/train",  # 데이터 경로
        # "submission_path": "../submission",
        # "submission_name": "multi-VAE_submission.csv",
        # "model_path": "../model",  # 모델 저장 경로
        # "model_name": "test.pt",
        # "seed": 22,
        "lr": 0.001,
        "batch_size": 256,
        "num_epochs": 1,
        "num_workers": 2,
        "valid_samples": 10,
        "weight_decay": 0.00,
        # # #### VAE
        "p_dims": [100, 200, 400],
        "dropout_rate": 0.5,
        "anneal_cap": 0.2,
        "total_anneal_steps": 200000,
        # #### DAE
        "p_dims": [100, 200, 400],
        "dropout_rate": 0.5,
        # #### AutoRec
        "hidden_dim": 64,
        # #### RecVAE
        "hidden_dim": 300,
        "latent_dim": 100,
        "dropout_rate": 0.7,
        "gamma": 0.0005,
        "beta": None,
    }

    config = Box(config)

    if not os.path.isdir(config.model_path):
        os.mkdir(config.model_path)

    if not os.path.isdir(config.submission_path):
        os.mkdir(config.submission_path)

    parser = argparse.ArgumentParser()
    
    remote_server_uri = "http://101.101.211.226:30005"
    mlflow.set_tracking_uri(remote_server_uri)
    
    parser.add_argument("--model_type", default="MultiVAE", type=str)
    
    # -- mlflow args
    parser.add_argument(
        "--experiment",
        type=str,
        default="Test",
        help="set experiment name (default: Test)",
    )
    parser.add_argument(
        "--run_name", type=str, default="test_run", help="set experiment runname"
    )
    parser.add_argument(
        "--user", type=str, default="unknown", help="set experiment username"
    )
    
    args = parser.parse_args()
    tuner_params = nni.get_next_parameter()

    # -- mlflow experiment
    experiment_name_dict = {
        "Test": "Test_experiment",  # default
        "Matrix": "Matrix_model_experimet",
        "Graph": "Graph_model_experiment",
        "Sequence": "Sequence_model_experiment" 
    }

    experiment_name = experiment_name_dict[args.experiment]
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    client = mlflow.tracking.MlflowClient()

    run = client.create_run(experiment.experiment_id)

    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.set_tag("mlflow.user", args.user)
        mlflow.set_tag("mlflow.runName", args.run_name)
        mlflow.log_params(args.__dict__)
        mlflow.log_params(config)
    
        if args.model_type not in ["MultiVAE", "MultiDAE", "AutoRec", "RecVAE", "Ensemble"]:
            print("Not in model_type")
        elif args.model_type == 'Ensemble':
            ensemble_main(config=config)
        else:
            matrix_main(config=config, model_type=args.model_type)