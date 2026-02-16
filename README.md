# Jigsaw Agile Community Rules Classification
- This repository contains a hybrid deep learning and classical machine learning solution for detecting community rule violations in text data.
- The architecture combines the semantic understanding of Deep Learning (DistilBERT) with the robust tabular performance of Gradient Boosting Machines (LGBM, XGBoost, CatBoost) through a weighted ensemble.

## Dataset
- Data is provided by the competition organizers.  
- It contains user-generated content paired with specific community guidelines and contextual examples.

*Structure:* Each row includes a body, a rule, 2 positive and 2 negative examples.

## Notebooks
#### 01_EDA.ipynb  
- Exploratory Data Analysis of text lengths, rule distributions, and label balance.

#### 02_DistilBERT_Training.ipynb (Optional)  
- Code for training the custom 5-Channel DistilBERT model from scratch.
- In this approach, each input (body, positive example 1 & 2, and negative example 1 & 2) is individually combined with the corresponding rule and fed into the model's five parallel channels to learn contrastive nuances.

#### 03_Preprocess.ipynb  
- Feature engineering, text cleaning, and generating embeddings/CSR matrices.

#### 04_Hyperparameter_Search.ipynb  
- Hyperparameter optimization process using Optuna.

#### 05_Final_Training_and_Inference.ipynb  
- Training the final ensemble and generating weighted hybrid predictions.

## Models & Weights
Deep Learning Model: *5-Channel DistilBERT*  

Classical ML: *A Voting Classifier integrating LightGBM, XGBoost, and CatBoost, each optimized via Optuna.*  

Ensemble: *Weighted blend (75% DL / 25% GBDT)*  

Pretrained Weights: *The fine-tuned weights are hosted on Hugging Face and are automatically integrated into the inference pipeline*: BernaTS/distilbert-5channel-jigsaw.

## How to Run
1. Install dependencies  
2. Run the notebooks in the following order:  
   (You can skip 02_DistilBERT_Training.ipynb as the pipeline is configured to pull pretrained weights from Hugging Face.)
- 01_EDA.ipynb
- 03_Preprocess.ipynb
- 04_Hyperparameter_Search.ipynb
- 05_Final_Training_and_Inference.ipynb 
