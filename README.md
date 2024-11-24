# XRay-Sinus-Diagnosis
부비동 X-Ray를 입력으로 받아들여 딥러닝 알고리즘을 통해 부비동염 유무를 판별하는 시스템

## Overview
This repository contains the code I personally contributed to during the development of the Sinusitis Diagnosis Assistant project.  
본 리포지토리는 '부비동염 진단 보조 시스템' 프로젝트에서 제가 기여한 코드만 포함하고 있습니다.

The project focuses on utilizing X-Ray images to train EfficientNet, Inception, and ResNet models for detecting sinusitis and integrates predictions through a majority voting algorithm for final diagnosis.  
이 프로젝트는 부비동염 X-Ray 데이터를 활용하여 EfficientNet, Inception, ResNet 모델을 학습시키고, 다수결 알고리즘을 통해 최종 진단 결과를 도출하는 것을 목표로 합니다.

## My Contributions
1. Preprocessing sinusitis X-ray data for training.  
   부비동염 X-Ray 데이터를 학습에 적합하도록 전처리
2. Training EfficientNet and Inception models for sinusitis diagnosis.  
   부비동염 진단을 위한 EfficientNet 및 Inception 모델 학습
3. Implementing a majority voting algorithm to combine predictions from multiple models.  
   여러 모델의 예측 결과를 통합하는 다수결 알고리즘 구현
