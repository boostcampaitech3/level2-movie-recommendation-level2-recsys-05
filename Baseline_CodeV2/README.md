# Movie Recommendation Baseline Code V2
---
## 🔍 업데이트 노트
### v.2.2.0
[[#4](https://github.com/boostcampaitech3/level2-movie-recommendation-level2-recsys-05/pull/4)] MLFlow로 원격 로깅이 가능해졌습니다.

- MLFlow 설치  
   ```
   pip install mlflow
   ```

- 원격 로깅 주소   
http://101.101.211.226:30005

- MLFlow 로깅 사용법   
노션에 정리해두었으니 아래 글을 참고해주세요.   
https://abyssinian-decade-924.notion.site/MLFlow-53b428564de143c8a91e462bb14484a3


### **v.2.1.0**
[[#3](https://github.com/boostcampaitech3/level2-movie-recommendation-level2-recsys-05/pull/3)]
argument를 yaml파일로 관리할 수 있습니다.


### **v.2.0.0**
[[#1](https://github.com/boostcampaitech3/level2-movie-recommendation-level2-recsys-05/pull/1)] 코드 가독성을 높이기 위해 코드를 리팩토링했습니다. 


<br></br>

--- 
## Getting Started
### Installation

```
pip install -r requirements.txt
```

### How to run

1. Pretraining
   ```
   python run_pretrain.py
   ```
2. Fine Tuning (Main Training)
   1. with pretrained weight
      ```
      python run_train.py --using_pretrain
      ```
   2. without pretrained weight
      ```
      python run_train.py
      ```
3. Inference
   ```
   python inference.py
   ```
