# Experiment
## 1. 검증 전략 
- 유저 별로 10개의 영화를 random sampling하여, 주어진 Task와 비슷한 validation set을 구축
- 실제로 validation set에서 성능이 오르면 리더 보드에서도 동일하게 성능이 오르는 것을 확인
- 이를 활용해 모델 성능 평가와 피드백을 빠른 속도로 진행하여, 다양한 모델을 실험

## 2. Model

### (1) AutoEncoder 기반 (유저-아이템 상호작용을 복원할 수 있는 parameterized function을 만듬)
- AutoRec (non-)
- Multi-DAE
- Multi-VAE
- Mutli-CDAE
- RecVAE
- EASE
- Multi-EASE
- ADMM-SLIM
- EASER

### (2) Transformer 기반 (유저의 영화 평가 이력 sequence를 바탕으로 다음 영화를 예)
- SASRec
- BERT4Rec
- Multi-BERT4Rec

### (3) MLP 기반 (유저-아이템 상호작용을 표현할 수 있는 MLP Layer를 만듬)
- NCF
- DeepFM
- Item2Vec

### (4) GNN 기반 (유저 간의 상호작용을 Layer로 표현하여 영화와 유저를 임베딩하여, 유사한 영화를 예측)
- NCF
- NGCF

## 3. Ensemble