Know Your Diabetes App

1. OVERVIEW
This project is a generative style diabetes risk app that uses a probabilistic Deep Learning model and a logic layer to help users understand whether they may fall into one of three categories: no diabetes, prediabetes, or diabetes. I also added a fourth deterministic check for Type 1 diabetes using symptom based rules.
The goal was to build an app that does not just output a label, but actually reasons with uncertainty, confidence, and probability distributions.

2. DATASETS
I used real world data from the NHANES (National Health and Nutrition Examination Survey) dataset. NHANES includes demographic information and clinical lab values such as A1C, glucose, insulin, BMI, cholesterol, blood pressure, kidney indicators, physical activity, smoking history, and weight.
I originally explored BRFSS as well, but between a dataset that focused on risk factors or laboratorial outputs, I decided to go with NHANES Dataset. 

3. DATA CLEANING AND PREPROCESSING
NHANES is real world data, so it required a lot of cleanup. The following were a list of cleanup I performed on the dataset:
•	Merged about 15 NHANES CSV files using SEQN
•	Renamed technical column names to readable ones
•	Consistently checked for and removed invalid row data
•	Removed invalid diabetes codes (7 = refused, 9 = don’t know)
•	Dropped SEQN after merging
•	Checked distributions using describe(), histograms, and quartiles
•	Replaced placeholder values like 999 with NaN
•	Capped impossible values (e.g., weight > 700 lbs)
•	Split data before imputation to avoid leakage
•	Imputed:
         o	Median for numeric columns (skewed data)
         o	Mode for categorical columns
•	Converted NHANES category codes into strings (safer for encoding)
•	One hot encoded categorical features
•	Scaled numeric features using RobustScaler
•	Saved encoders/scalers for inference
•	Converted everything into tensors for TensorFlow

4. MODEL TRAINING
I trained a Multilayer Perceptron (MLP) using TensorFlow/Keras.
This is sufficient for a small tabular Dataset like NHANES.
Key characteristics:
•	Dense layers with reLu
•	Dropout layers
•	Softmax output for probability distributions
•	Categorical cross entropy loss because this is a multi class classification.
•	Adam optimizer
•	Class weights to handle imbalance
•	Early stopping (10 epochs)
The model outputs three classes:
•	Diabetes
•	No diabetes
•	Prediabetes

5. UNCERTAINTY ESTIMATION. MONTE CARLO DROPOUT
To make the app probabilistic, I used Monte Carlo Dropout at inference:
•	Forced dropout ON
•	Ran 50 forward passes
•	Collected:
       •	Mean prediction
       •	Variance (uncertainty)
       •	Confidence interval
Low variance across passes means the model is confident. My results showed extremely low variance, which means the model’s predictions are stable.
 
6. EVALUATION
Training Results
•	Loss: ~28%
•	Accuracy: ~92%
•	Training and validation curves match which means there is no overfitting.
Metrics
•	F1 Score: 90% - Strong balance of precision and recall.
•	ROC AUC: 84% - Good ranking ability across classes.
•	Confusion Matrix - Excellent on the majority class, decent on the mid-frequency class, weaker on the tiny minority class (I learned this is to be expected because NHANES has very few borderline cases).
Label Note: 
NHANES uses codes (1, 2, 3), but LabelEncoder remaps them to (0, 1, 2). 

7. LOGIC AND REASONING LAYER 
Beyond the neural network, I built a separate logic layer that adds reasoning:
•	Symptom ranking using a heap based priority system
•	Deterministic Type 1 diabetes check using rule based criteria
•	Additional deterministic check for definite diabetes diagnosis if glucose is > 200 and a1c > 8.
•	Additional deterministic check for definite diabetes if all major symptoms and high glucose and a1c detected.
•	Markov chain to estimate progression probabilities
•	Integration of model uncertainty into the final message
•	Handling of edge cases (low glucose, conflicting symptoms, missing data)
•	Human readable and relatable explanations
This layer is separate from the model and can be updated without retraining the model.

8. BACKEND 
The backend is a clean logic module with:
•	No UI
•	No input() or logger.debug()
•	A single analyze() entry point
•	Deterministic behavior
•	Clear separation between:
     •	Feature encoding
     •	Model inference
     •	Logic reasoning

9. FRONTEND
I built a simple Streamlit frontend so users can interact with the model and logic layer. The UI collects inputs, sends them to the backend, displays the probability distribution, uncertainty, and the final reasoning message. The frontend is lightweight and only handles user interaction. All logic and inference happen in the backend.
   
10. CONCLUSION
This project combines:
•	Real world clinical data
•	Deep learning
•	Probabilistic reasoning
•	Uncertainty estimation
•	A custom logic layer
•	Markov chains
•	Deterministic rule based checks

11. LOADING MODEL:
The neural network is saved as diabetesgen.keras using tf.keras.models.save_model().
The one‑hot encoder is saved as ohe.pkl using joblib.dump().
The scaler is saved as scaler.pkl using joblib.dump().

loaded in backend using:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "diabetesgen.keras"),
    compile=False
)

ohe = joblib.load(os.path.join(BASE_DIR, "ohe.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

12. PROJECT ARCHITECTURE
KnowYourDiabetesApp/
├── backend/
│   ├── backend.py
│   ├── logic.py
│   ├── model.pkl
│   ├── ohe.pkl
│   ├── scaler.pkl
│   └── diabetesgen.keras
│   └── requirements.txt
├── frontend/
│   └── frontend.py
│   └── requirements.txt
└── notebooks/
│   └── diabetesgen.ipynb
└── docs
│   └── NHANES Data.pdf
│   └── CDC Diabetes Facts.pdf
│   └── Screenshots.pdf
└── README.md
└── .gitignore

12. INSTALLATION
 - git clone https://github.com/fayedoneaway/Know_Your_Diabetes_App
 - cd Know_Your_Diabetes_App
 - python -m venv venv
 - venv\Scripts\activate (windows)
 - source venv/bin/activate (mac or linux)
 - pip install -r backend/requirements.txt
 - pip install -r frontend/requirements.txt

13. RUN 
 a. backend
 - uvicorn backend.backend:app --reload
   http://127.0.0.1:8000/docs

 b. frontend
 - streamlit run frontend/frontend.py

14. REQUIREMENTS 
 a. backend
 - fastapi
 - uvicorn
 - tensorflow
 - keras
 - numpy
 - pandas
 - matplotlib
 - joblib
 - heapq
 - typing Optional, Dict, Any
 - logging

 b. frontend
 - streamlit
 - requests
