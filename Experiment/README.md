# Experiment
## 1. 검증 전략 
- 유저 별로 10개의 영화를 random sampling하여, 주어진 Task와 비슷한 validation set을 구축
- 실제로 validation set에서 성능이 오르면 리더 보드에서도 동일하게 성능이 오르는 것을 확인
- 이를 활용해 모델 성능 평가와 피드백을 빠른 속도로 진행하여, 다양한 모델을 실험

## 2. Model

### (1) AutoEncoder 기반 (유저-아이템 상호작용을 복원할 수 있는 parameterized function을 만듬)
- AutoRec (non-linear)
- Multi-DAE (non-linear)
- Multi-VAE (non-linear)
- Mutli-CDAE (non-linear)
- RecVAE (non-linear)
- EASE (linear)
- Multi-EASE (linear)
- ADMM-SLIM (linear)
- EASER (linear)

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
- 모델 별 candidate 아이템의 순위를 바탕으로 1 / log2(rank + 1)을 계산하여 모델 별 아이템 score 값을 구함
- 모델 별 아이템 score 값을 sum하여 candidate 집단에서 re-ranking
- 모델 별로 가중치를 두어 re-ranking
- 실험한 모델 중 가장 효과적인 조합을 찾기 위해 Model Best Combination Serch를 진행