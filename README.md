T20 Cricket Match Winner Prediction

This project is a machine learning solution developed for the American Express Campus Challenge (March 2024 - July 2024) to predict the winner of T20 cricket matches. The model achieved a 64% accuracy and ranked among the Top 30 out of 4,000+ participants.

Project Overview

The core objective of this project is to forecast the outcomes of T20 cricket matches by leveraging advanced feature engineering and a robust machine learning pipeline. The solution involves in-depth data preprocessing, feature selection, and the implementation of powerful boosting algorithms to build a highly predictive model.

Key Achievements

Top 30 Rank: Placed among the top 30 performers in a competitive field of over 4,000 participants.

High Accuracy: Achieved a prediction accuracy of 64% on the test dataset.

Advanced Feature Engineering: Extracted and engineered over 35 insightful features from the raw data to capture the nuances of T20 cricket matches.

Optimized Model: Developed a comprehensive ML pipeline that includes data cleaning, feature selection, and model tuning to ensure optimal performance.

Methodology

The project follows a structured machine learning workflow:

Feature Engineering: Generated over 35 features, capturing team form, player statistics, venue details, and historical match data.

Feature Selection:

Principal Component Analysis (PCA): Used for dimensionality reduction to select the most informative features while preserving variance.

Variance Inflation Factor (VIF): Employed to identify and mitigate multicollinearity among features, ensuring model stability.

Modeling:

Utilized a suite of powerful boosting algorithms, including CatBoost and LightGBM.

Performed extensive hyperparameter tuning and applied regularization techniques to prevent overfitting and enhance the generalizability of the models.

A Stacking Classifier was implemented, using the predictions of base models (XGBoost, LightGBM, CatBoost, GradientBoosting) as input for a final logistic regression model to improve overall accuracy.

Getting Started

Follow these instructions to set up and run the project locally.

Prerequisites

Python 3.x

Jupyter Notebook or any Python IDE

Installation

Clone the repository to your local machine:

git clone [https://github.com/your-username/t20-cricket-prediction.git](https://github.com/your-username/t20-cricket-prediction.git)


Navigate to the project directory:

cd t20-cricket-prediction


Install the required dependencies using the requirements.txt file:

pip install -r requirements.txt


Usage

Launch Jupyter Notebook:

jupyter notebook


Open the AmEx_T20.ipynb notebook.

Execute the cells in the notebook sequentially to train the model and generate predictions.

The notebook contains the complete pipeline, from data loading and preprocessing to model training and evaluation. The final predictions will be saved to an output CSV file.
