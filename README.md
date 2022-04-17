<p align="center"><img src="https://user-images.githubusercontent.com/65529313/163712073-7d2dcd09-4c1f-4bab-935f-42de292300bb.png" width="800" height="100"/></p>

# 프로젝트 목표
- 사용자의 영화 평가 이력을 바탕으로 사용자가 선호할 10개의 영화를 예측

# 활용 장비
- Ubuntu 18.04.5 LTS
- GPU Tesla V100-PCIE-32GB

# 프로젝트 팀 구성 및 역할
- **김건우**: 모델 탐색 및 튜닝, 일정 관리
- **김동우**: 모델 탐색 및 튜닝, Ensemble
- **박기정:** Project Template 설계 및 리드, 모델 튜닝, Ensemble
- **심유정:** MLFlow 및 NNI 적용, Project Template 작업, 모델 튜닝
- **이성범:** 모델 탐색, 모델 선정 및 분석, Ensemble, Project Template 제작을 위한 모델 모듈화

# 개요
<p align="center"><img src="https://user-images.githubusercontent.com/65529313/163713560-2eabc68f-1aaa-4bf8-ad14-0e6078a817ab.png" width="1000" height="400"/></p>

# 검증 전략
- 유저 별로 10개의 영화를 random sampling하여, 주어진 Task와 비슷한 validation set을 구축
-  validation set에서 성능이 오르면 리더 보드에서도 동일하게 성능이 오르는 것을 확인하여 빠른 속도로 다양한 실험을 진행

# 모델 결과
<p align="center"><img src="https://user-images.githubusercontent.com/65529313/163712308-8c09cdd5-7cde-4bb8-8e3d-cddd329bde53.png" width="1000" height="400"/></p>

# Project Template
<p align="center"><img src="https://user-images.githubusercontent.com/65529313/163712323-df153c2e-1502-4441-b3d3-ab187372d593.png" width="1000" height="600"/></p>

# 프로젝트 결과
<p align="center"><img src="https://user-images.githubusercontent.com/65529313/163712409-28c29a8d-b13d-4328-a617-6818f232c84e.png" width="1000" height="400"/></p>
