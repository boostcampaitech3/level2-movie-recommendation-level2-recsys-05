from models.BERTRec import BERT4Rec
from models.SASRec import SASRec
from preprocessing.MakeSequenceDataSet import MakeSequenceDataSet, SASRecDataSet, BERTRecDataSet
from trainers.BERT4Rec import train as BERT4Rec_train
from trainers.BERT4Rec import evaluate as BERT4Rec_evaluate
from trainers.SASRec import train as SASRec_train
from trainers.SASRec import evaluate as SASRec_evaluate

from models.MultiDAE import MultiDAE
from models.MultiVAE import MultiVAE
from preprocessing.MakeMatrixDataSet import MakeMatrixDataSet, AEDataSet
from trainers.MultiDAE import train as MultiDAE_train
from trainers.MultiDAE import evaluate as MultiDAE_evaluate
from trainers.MultiVAE import train as MultiVAE_train
from trainers.MultiVAE import evaluate as MultiVAE_evaluate
from trainers.MultiDAE import loss_function_dae
from trainers.MultiVAE import loss_function_vae

from models.NGCF import NGCF
from models.LightGCN import LightGCN
from preprocessing.MakeGraphDataSet import MakeGraphDataSet
from trainers.NGCF import train as NGCF_train
from trainers.NGCF import evaluate as NGCF_evaluate
from trainers.LightGCN import train as LightGCN_train
from trainers.LightGCN import evaluate as LightGCN_evaluate

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

from box import Box

def MultiVAE_main():
    config = {
        'data_path' : "/opt/ml/input/data/train" , # 데이터 경로
        
        'submission_path' : "../submission",
        'submission_name' : 'test.csv', 

        'model_path' : "../model", # 모델 저장 경로
        'model_name' : 'test.pt',

        'p_dims': [250, 500, 1000], 
        'dropout_rate' : 0.5,
        'weight_decay' : 0.00,
        'valid_samples' : 10, # 검증에 사용할 sample 수
        'seed' : 22,
        'anneal_cap' : 0.2,
        'total_anneal_steps' : 200000,

        'lr' : 0.001,
        'batch_size' : 128,
        'num_epochs' : 2,
        'num_workers' : 2,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Box(config)

    if not os.path.isdir(config.model_path):
        os.mkdir(config.model_path)

    if not os.path.isdir(config.submission_path):
        os.mkdir(config.submission_path)


    make_matrix_data_set = MakeMatrixDataSet(config = config)
    user_train, user_valid = make_matrix_data_set.get_train_valid_data()

    ae_dataset = AEDataSet(
        num_user = make_matrix_data_set.num_user,
        )

    data_loader = DataLoader(
        ae_dataset,
        batch_size = config.batch_size, 
        shuffle = True, 
        pin_memory = True,
        num_workers = config.num_workers,
        )

    model = MultiVAE(
        p_dims = config.p_dims + [make_matrix_data_set.num_item], 
        dropout_rate = config.dropout_rate).to(device)

    criterion = loss_function_vae
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay = config.weight_decay)

    best_hit = 0
    for epoch in range(1, config.num_epochs + 1):
        train_loss = MultiVAE_train(
            model = model, 
            criterion = criterion, 
            optimizer = optimizer, 
            data_loader = data_loader,
            make_matrix_data_set = make_matrix_data_set,
            config = config,
            )
        
        ndcg, hit = MultiVAE_evaluate(
            model = model, 
            data_loader = data_loader,
            user_train = user_train,
            user_valid = user_valid,
            make_matrix_data_set = make_matrix_data_set,
            )

        if best_hit < hit:
            best_hit = hit
            torch.save(model.state_dict(), os.path.join(config.model_path, config.model_name))

        print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')


def MultiDAE_main():
    config = {
    'data_path' : "/opt/ml/input/data/train" , # 데이터 경로
    
    'submission_path' : "../submission",
    'submission_name' : 'multi-DAE_v2_submission.csv', 

    'model_path' : "../model", # 모델 저장 경로
    'model_name' : 'Multi-DAE_v2.pt',

    'p_dims': [250, 500, 1000], 
    'dropout_rate' : 0.5,
    'weight_decay' : 0.00,
    'valid_samples' : 10, # 검증에 사용할 sample 수
    'seed' : 22,

    'lr' : 0.001,
    'batch_size' : 128,
    'num_epochs' : 50,
    'num_workers' : 2,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Box(config)

    if not os.path.isdir(config.model_path):
        os.mkdir(config.model_path)

    if not os.path.isdir(config.submission_path):
        os.mkdir(config.submission_path)

    make_matrix_data_set = MakeMatrixDataSet(config = config)
    user_train, user_valid = make_matrix_data_set.get_train_valid_data()

    ae_dataset = AEDataSet(
    num_user = make_matrix_data_set.num_user,
    )

    data_loader = DataLoader(
    ae_dataset,
    batch_size = config.batch_size, 
    shuffle = True, 
    pin_memory = True,
    num_workers = config.num_workers,
    )

    model = MultiDAE(
    p_dims = config.p_dims + [make_matrix_data_set.num_item], 
    dropout_rate = config.dropout_rate).to(device)

    criterion = loss_function_dae
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay = config.weight_decay)

    best_hit = 0
    for epoch in range(1, config.num_epochs + 1):
        train_loss = MultiDAE_train(
            model = model, 
            criterion = criterion, 
            optimizer = optimizer, 
            data_loader = data_loader,
            make_matrix_data_set = make_matrix_data_set
            )
        
        ndcg, hit = MultiDAE_evaluate(
            model = model,
            data_loader = data_loader,
            user_train = user_train,
            user_valid = user_valid,
            make_matrix_data_set = make_matrix_data_set,
            )

        if best_hit < hit:
            best_hit = hit
            torch.save(model.state_dict(), os.path.join(config.model_path, config.model_name))

        print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')


def SASRec_main():
    config = {

    'data_path' : "/opt/ml/input/data/train" , # 데이터 경로
    'model_path' : "../model", # 모델 저장 경로
    'model_name' : 'SASRec_v1.pt',

    'max_len' : 50,
    'hidden_units' : 50, # Embedding size
    'num_heads' : 1, # Multi-head layer 의 수 (병렬 처리)
    'num_layers': 2, # block의 개수 (encoder layer의 개수)
    'dropout_rate' : 0.5, # dropout 비율
    'lr' : 0.001,
    'batch_size' : 128,
    'num_epochs' : 200,
    'num_workers' : 2,

    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Box(config)

    if not os.path.isdir(config.model_path):
        os.mkdir(config.model_path)

    if not os.path.isdir(config.submission_path):
        os.mkdir(config.submission_path)

    make_sequence_dataset = MakeSequenceDataSet(config = config)
    user_train, user_valid = make_sequence_dataset.get_train_valid_data()

    sasrec_dataset = SASRecDataSet(
        user_train = user_train, 
        max_len = config.max_len, 
        num_user = make_sequence_dataset.num_user, 
        num_item = make_sequence_dataset.num_item, 
        )

    data_loader = DataLoader(
        sasrec_dataset, 
        batch_size = config.batch_size, 
        shuffle = True, 
        pin_memory = True,
        num_workers = config.num_workers,
        )

    model = SASRec(
        num_user = make_sequence_dataset.num_user, 
        num_item = make_sequence_dataset.num_item, 
        hidden_units = config.hidden_units, 
        num_heads = config.num_heads, 
        num_layers = config.num_layers, 
        max_len = config.max_len, 
        dropout_rate = config.dropout_rate, 
        device = device,
        ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_ndcg = 0
    for epoch in range(1, config.num_epochs + 1):
        train_loss = SASRec_train(
            model = model, 
            criterion = criterion, 
            optimizer = optimizer, 
            data_loader = data_loader)
        
        ndcg, hit = SASRec_evaluate(
            model = model, 
            user_train = user_train, 
            user_valid = user_valid, 
            max_len = config.max_len,
            sasrec_dataset = sasrec_dataset, 
            make_sequence_dataset = make_sequence_dataset,
            )

        if best_ndcg < ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), os.path.join(config.model_path, config.model_name))

        print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')


def BERTRec_main():
    config = {

        'data_path' : "/opt/ml/input/data/train" , # 데이터 경로
        'model_path' : "../model", # 모델 저장 경로
        'model_name' : 'BRETRec_v1.pt',

        'max_len' : 50,
        'hidden_units' : 50, # Embedding size
        'num_heads' : 1, # Multi-head layer 의 수 (병렬 처리)
        'num_layers': 2, # block의 개수 (encoder layer의 개수)
        'dropout_rate' : 0.5, # dropout 비율

        'lr' : 0.001,
        'batch_size' : 128,
        'num_epochs' : 200,
        'num_workers' : 2,
        'mask_prob' : 0.15, # for cloze task
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Box(config)

    if not os.path.isdir(config.model_path):
        os.mkdir(config.model_path)

    if not os.path.isdir(config.submission_path):
        os.mkdir(config.submission_path)

    make_sequence_dataset = MakeSequenceDataSet(config = config)
    user_train, user_valid = make_sequence_dataset.get_train_valid_data()

    bertrec_dataset = BERTRecDataSet(
        user_train = user_train, 
        max_len = config.max_len, 
        num_user = make_sequence_dataset.num_user, 
        num_item = make_sequence_dataset.num_item,
        mask_prob = config.mask_prob,
        )

    data_loader = DataLoader(
        bertrec_dataset, 
        batch_size = config.batch_size, 
        shuffle = True, 
        pin_memory = True,
        num_workers = config.num_workers,
        )

    model = BERT4Rec(
        num_user = make_sequence_dataset.num_user, 
        num_item = make_sequence_dataset.num_item, 
        hidden_units = config.hidden_units, 
        num_heads = config.num_heads, 
        num_layers = config.num_layers, 
        max_len = config.max_len, 
        dropout_rate = config.dropout_rate, 
        device = device,
        ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_ndcg = 0
    for epoch in range(1, config.num_epochs + 1):
        train_loss = BERT4Rec_train(
            model = model, 
            criterion = criterion, 
            optimizer = optimizer, 
            data_loader = data_loader)
        
        ndcg, hit = BERT4Rec_evaluate(
            model = model, 
            user_train = user_train, 
            user_valid = user_valid, 
            max_len = config.max_len,
            dataset = bertrec_dataset, 
            make_sequence_dataset = make_sequence_dataset,
            )

        if best_ndcg < ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), os.path.join(config.model_path, config.model_name))

        print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')


def NGCF_main():
    config = {
        'data_path' : "/opt/ml/input/data/train" , # 데이터 경로
        
        'submission_path' : "../submission",
        'submission_name' : 'NGCF_submission.csv', 

        'model_path' : "../model", # 모델 저장 경로
        'model_name' : 'NGCF_v1.pt',

        'num_epochs' : 50,
        "reg" : 1e-5,
        'lr' : 0.0001,
        "emb_dim" : 512,
        "layers" : [512, 512],
        'batch_size' : 1024,
        "node_dropout" : 0.2,
        "mess_dropout" : 0.2,

        'valid_samples' : 10, # 검증에 사용할 sample 수
        'seed' : 22,
        'n_batch' : 30,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Box(config)

    if not os.path.isdir(config.model_path):
        os.mkdir(config.model_path)

    if not os.path.isdir(config.submission_path):
        os.mkdir(config.submission_path)

    make_graph_data_set = MakeGraphDataSet(config = config)
    ngcf_adj_matrix = make_graph_data_set.get_ngcf_adj_matrix_data()
    R_train, R_valid, R_total = make_graph_data_set.get_R_data()

    model = NGCF(
        n_users = make_graph_data_set.num_user,
        n_items = make_graph_data_set.num_item,
        emb_dim = config.emb_dim,
        layers = config.layers,
        reg = config.reg,
        node_dropout = config.node_dropout,
        mess_dropout = config.mess_dropout,
        adj_mtx = ngcf_adj_matrix,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_hit = 0
    for epoch in range(1, config.num_epochs + 1):
        train_loss = NGCF_train(
            model = model, 
            make_graph_data_set = make_graph_data_set, 
            optimizer = optimizer,
            n_batch = config.n_batch,
            )
        with torch.no_grad():
            ndcg, hit = NGCF_evaluate(
                u_emb = model.u_final_embeddings.detach(), 
                i_emb = model.i_final_embeddings.detach(), 
                Rtr = R_train, 
                Rte = R_valid, 
                k = 10,
                )

        if best_hit < hit:
            best_hit = hit
            torch.save(model.state_dict(), os.path.join(config.model_path, config.model_name))

        print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')


def LightGCN_main():
    config = {
        'data_path' : "/opt/ml/input/data/train" , # 데이터 경로
        
        'submission_path' : "../submission",
        'submission_name' : 'LightGCN_submission.csv', 

        'model_path' : "../model", # 모델 저장 경로
        'model_name' : 'LightGCN_v1.pt',

        'num_epochs' : 50,
        "reg" : 1e-5,
        'lr' : 0.0001,
        "emb_dim" : 512,
        "n_layers" : 3,
        'batch_size' : 1024,
        "node_dropout" : 0.2,

        'valid_samples' : 10, # 검증에 사용할 sample 수
        'seed' : 22,
        'n_batch' : 30,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Box(config)

    if not os.path.isdir(config.model_path):
        os.mkdir(config.model_path)

    if not os.path.isdir(config.submission_path):
        os.mkdir(config.submission_path)

    make_graph_data_set = MakeGraphDataSet(config = config)
    ngcf_adj_matrix = make_graph_data_set.get_ngcf_adj_matrix_data()
    R_train, R_valid, R_total = make_graph_data_set.get_R_data()

    model = LightGCN(
        n_users = make_graph_data_set.num_user,
        n_items = make_graph_data_set.num_item,
        emb_dim = config.emb_dim,
        n_layers = config.n_layers,
        reg = config.reg,
        node_dropout = config.node_dropout,
        adj_mtx = ngcf_adj_matrix,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_hit = 0
    for epoch in range(1, config.num_epochs + 1):
        train_loss = LightGCN_train(
            model = model, 
            make_graph_data_set = make_graph_data_set, 
            optimizer = optimizer,
            n_batch = config.n_batch,
            )
        with torch.no_grad():
            ndcg, hit = LightGCN_evaluate(
                u_emb = model.u_final_embeddings.detach(), 
                i_emb = model.i_final_embeddings.detach(), 
                Rtr = R_train, 
                Rte = R_valid, 
                k = 10,
                )

        if best_hit < hit:
            best_hit = hit
            torch.save(model.state_dict(), os.path.join(config.model_path, config.model_name))

        print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="MultiVAE", type=str)
    args = parser.parse_args()
    if args.model_type == "MultiVAE": MultiVAE_main()
    elif args.model_type == "MultiDAE": MultiDAE_main()
    elif args.model_type == "SASRec": SASRec_main()
    elif args.model_type == "BERTRec": BERTRec_main()
    elif args.model_type == "NGCF": NGCF_main()
    elif args.model_type == "LightGCN": LightGCN_main()
    else: print('Not in model_type')