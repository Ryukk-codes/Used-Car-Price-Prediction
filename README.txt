USED CAR PRICE PREDICTION SYSTEM
================================

Overview
--------
Peer-to-peer used vehicle markets suffer from pricing opacity, subjective appraisals, and extreme listing variance. This project ingests large-scale Craigslist vehicle data, filters out anomalies and spam listings, trains an ensemble regression model (RandomForestRegressor), and serves real-time price valuations through an interactive web interface.

- UN-SDG Alignment: Supports Goal 12 (Responsible Consumption and Production) by improving transparency and trust in secondary automotive markets to extend vehicle lifecycles, and Goal 9 (Industry, Innovation, and Infrastructure) by applying machine learning to informal pricing sectors.

Project Structure
-----------------
- app.py                     : Streamlit web application for real-time price prediction
- train_model.py             : Data preprocessing, training, evaluation, and serialization script
- requirements.txt           : Python package dependencies
- craigslist_car_model.pkl   : Serialized Random Forest model (generated after training)
- encoders.pkl               : Serialized LabelEncoders for categorical features (generated after training)
- README.txt                 : Project documentation

Dataset Notice (Manual Download Required)
-----------------------------------------
IMPORTANT: The dataset file (vehicles.csv) is NOT included in this repository due to its large file size (~1.4 GB / 426,000+ records), which exceeds standard repository upload limits. You must download it separately before running the training script.

1. Download the Craigslist Cars and Trucks Data by Austin Reese from Kaggle:
   https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

2. Extract the archive and place vehicles.csv directly in the root folder of this project alongside train_model.py and app.py.

Features Used:
- Target Variable     : price (USD)
- Predictive Features : year, manufacturer, condition, fuel, odometer, transmission

Setup & Installation
--------------------
1. Clone or download this repository and navigate to the project directory.

2. Install the required dependencies:
   python -m pip install -r requirements.txt

Usage
-----
Step 1: Train and Serialize the Model
Ensure vehicles.csv is inside the project directory, then run:
   python train_model.py

This script cleans the raw data (filtering prices between $500 and $80,000, years between 1995 and 2024, and odometer readings up to 300,000 miles), encodes categorical attributes, trains a Random Forest Regressor on an 80/20 split, prints the R2 score and Mean Absolute Error (MAE), and saves craigslist_car_model.pkl and encoders.pkl.

Step 2: Launch the Web Application
Once the .pkl files are generated, start the Streamlit interface:
   python -m streamlit run app.py

Open the local URL displayed in your terminal (typically http://localhost:8501) to input vehicle specifications and generate real-time market price predictions.