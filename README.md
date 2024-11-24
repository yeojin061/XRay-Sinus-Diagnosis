# XRay-Sinus-Diagnosis
부비동 X-Ray를 입력으로 받아들여 딥러닝 알고리즘을 통해 부비동염 유무를 판별하는 시스템

## Overview
This repository contains the code I personally contributed to during the development of the XRay Sinus Diagnosis project.
본 리포지토리는 '부비동염 진단 보조 시스템' 프로젝트에서 제가 기여한 코드만 포함하고 있습니다.

The project aims to improve the accuracy of sinusitis diagnosis using X-Ray imaging by training advanced deep learning models (EfficientNet, Inception, ResNet) and applying a majority voting algorithm for final predictions.
이 프로젝트는 고급 딥러닝 모델(EfficientNet, Inception, ResNet)을 학습하고, 다수결 알고리즘을 적용하여 X-ray를 기반으로 부비동염 진단의 정확도를 향상시키는 것을 목표로 합니다.

## My Contributions
1. Data preprocessing for X-Ray images and preparation for model training.
   X-Ray 이미지 데이터 전처리 및 모델 학습 준비
2. Training deep learning models (EfficientNet, Inception, ResNet) for sinusitis diagnosis.
   부비동염 진단을 위한 딥러닝 모델(EfficientNet, Inception, ResNet) 학습
3. Implementing a majority voting algorithm to combine predictions from multiple models.
   여러 모델의 예측을 통합하기 위한 다수결 알고리즘 구현

## Notes
This repository includes only my contributions. Other aspects of the project, such as UI development, are not included.
이 리포지토리에는 제 기여만 포함되어 있으며, UI 개발 등 다른 부분은 포함되지 않았습니다.
Due to data dependency issues, the code is not executable as is.
데이터 의존성 문제로 인해 현재 상태로는 실행이 불가능합니다.

# Project Structure

## Folder Structure
- data_preprocessing.py: Code for loading and preprocessing sinus X-ray data.
부비동 X-ray 데이터를 로드하고 전처리하는 코드를 포함합니다.
- model_definition.py: Code for defining the EfficientNet, Inception, and ResNet models.
EfficientNet, Inception, ResNet 모델을 정의하는 코드를 포함합니다.
- model_training.py: Code for training and evaluating the models.
모델을 학습하고 평가하는 코드를 포함합니다.
- ensemble/predict_with_models.py: Code for predicting using multiple models and applying the majority voting algorithm.
여러 모델을 사용해 예측하고 다수결 알고리즘을 적용하는 코드를 포함합니다.

## How to Use
1. Preprocess X-ray data using data_preprocessing.py.
   해당 코드를 사용하여 X-ray 데이터를 전처리합니다.
2. Define the models using model_definition.py.
   해당 코드를 사용하여 모델을 정의합니다.
3. Train the models using model_training.py.
   해당 코드를 사용하여 모델을 학습합니다.
4. Combine predictions from multiple models using ensemble/predict_with_models.py.
   여러 모델의 예측을 통합하기 위해 다수결 알고리즘을 사용합니다.


