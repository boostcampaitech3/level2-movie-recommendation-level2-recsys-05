import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from models import S3RecModel
from trainers import FinetuneTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)

    
def train():
    set_seed(args.seed)         # 시드 설정
    check_path(args.output_dir) # output 경로 지정(없다면 생성)

    # -- settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id   # 환경변수 "CUDA_VISIBLE_DEVICES"에 gpu_id 저장 
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda  # gpu 있을 경우 사용    


    # -- save log & model  
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")   # log file 모델이름-데이터이름.txt로 저장

    checkpoint = args_str + ".pt"                                          # 모델이름-데이터이름.pt로 모델 저장
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)       # 모델 저장할 경로


    # -- get data & preprocessing
    args.data_file = args.data_dir + "train_ratings.csv"  # 데이터 경로 + train_rating 데이터 -> arg.data_file
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json" # item2attribute 파일 경로

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(args.data_file)
    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)  # item2attributes를 item2attribute, size로 나눔

    args.item_size = max_item + 2  
    args.mask_id = max_item + 1    
    args.attribute_size = attribute_size + 1  

    args.item2attribute = item2attribute     # ml_item2attributes.json 파일의 key(item)
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix  # 학습할 매트릭스를 valid_rating_matrix로 지정
    

    # -- dataset
    train_dataset = SASRecDataset(args, user_seq, data_type="train")      
    train_sampler = RandomSampler(train_dataset)                           

    eval_dataset = SASRecDataset(args, user_seq, data_type="valid")       
    eval_sampler = SequentialSampler(eval_dataset)                      
    
    test_dataset = SASRecDataset(args, user_seq, data_type="test")       
    test_sampler = SequentialSampler(test_dataset)                 

        
    
    # -- data loader
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size   
    )

    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.batch_size
    )


    # -- model
    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(
        model, train_dataloader, eval_dataloader, test_dataloader, None, args
    )

    print(args.using_pretrain)
    
    if args.using_pretrain:
        pretrained_path = os.path.join(args.output_dir, "Pretrain.pt")
        try:
            trainer.load(pretrained_path)
            print(f"Load Checkpoint From {pretrained_path}!")

        except FileNotFoundError:
            print(f"{pretrained_path} Not Found! The Model is same as SASRec")
    else:
        print("Not using pretrained model. The Model is same as SASRec")

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    
    for epoch in range(args.epochs):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)

        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    trainer.args.train_matrix = test_rating_matrix
    print("---------------Change to test_rating_matrix!-------------------")
    
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0)
    print(result_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../input/data/train/", type=str)  
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument("--using_pretrain", action="store_true")

    args = parser.parse_args()
    print(args)
    
    train()