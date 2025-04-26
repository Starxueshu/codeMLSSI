Artificial Intelligence-Based Prediction Model for Surgical Site Infection in Metastatic Spinal Disease
This repository contains the source code for developing and validating machine learning models to predict surgical site infection (SSI) in patients with metastatic spinal disease, as described in our multicenter study.

Overview
The code implements a comprehensive machine learning pipeline for:

Data preprocessing and feature selection

Class imbalance handling using SMOTETomek

Model development and hyperparameter tuning

Performance evaluation and visualization

Clinical utility assessment via decision curve analysis

Requirements
The code requires the following Python packages:

Python 3.7+

NumPy

pandas

scikit-learn

scikit-learn-intelex

imbalanced-learn

XGBoost

SHAP

TensorFlow (v2.0.0)

Keras

Matplotlib

SciPy

Numba

Installation
Clone this repository:

bash
git clone [repository_url]
cd [repository_name]
Install the required packages:

bash
pip install -r requirements.txt
Note: For optimal performance, we recommend using Intel's scikit-learn extension:

bash
conda install scikit-learn-intelex
Data Preparation
The code expects input data in CSV format with the following features:

Demographic and Clinical Features:

Age, Gender, Smoking status, BMI

Comorbidities: Number of comorbidities, Coronary disease, Diabetes, Hypertension

Tumor Characteristics:

Tumor type, Primary tumor

Metastasis: Extravertebral bone metastasis, Visceral metastases

ECOG performance status

Surgical Factors:

Preoperative treatments: Chemotherapy, Targeted therapy, Endocrinology therapy, Embolization

Surgical details: Time, Process, Site, Segments

Blood transfusion

Laboratory Values:

WBC, Glucose, Albumin, HGB, PLT

Outcome:

SSI (Surgical Site Infection) - binary outcome variable

Model Development
The code implements and compares seven machine learning algorithms:

Logistic Regression (with L2 regularization)

XGBoost (eXtreme Gradient Boosting)

Decision Tree

k-Nearest Neighbors (KNN)

Neural Network (Multilayer Perceptron)

Gradient Boosting Machine (GBM)

Support Vector Machine (SVM)

Each model undergoes:

Hyperparameter tuning via GridSearchCV or RandomizedSearchCV

5-fold cross-validation

Performance evaluation on a held-out test set (20% of data)

Key Features
Data Preprocessing Pipeline:

Custom transformers for categorical and continuous variables

Standardization of numerical features

SMOTETomek for handling class imbalance

Model Evaluation:

Multiple metrics: Accuracy, Precision, Recall, F1, AUC-ROC

Brier score and log loss for probability calibration

Bootstrap confidence intervals for AUC

Decision Curve Analysis (DCA) for clinical utility

Interpretability:

SHAP values for model explanation (XGBoost and GBM)

Feature importance analysis

Visualization:

ROC curves with confidence intervals

Learning curves

Bootstrap stability plots

Usage
Prepare your dataset in CSV format following the structure described above

Update the file paths in the code to point to your data

Run the Jupyter notebook or Python script sequentially

The code will:

Split data into training (80%) and test (20%) sets

Train and tune all models

Generate performance metrics and visualizations

Save trained models as pickle files

Outputs
The code generates:

Trained model files (.pkl)

Performance metrics in Excel format

ROC curves

Decision curve analyses

Bootstrap stability plots

SHAP explanation plots (for tree-based models)

Citation
If you use this code in your research, please cite our paper:

[Artificial Intelligence-Based Prediction Model for Surgical Site Infection in Metastatic Spinal Disease: A Multicenter Development and Validation Study]

License
[Specify your license here, e.g., MIT License]

Contact
For questions or issues, please contact [Mingxing Lei [Star]] at [leimingxing2@sina.com].

This README provides a comprehensive overview of your code while maintaining scientific rigor. You may want to customize the contact information, citation, and license sections as appropriate for your publication.# codeMLSSI
