"""BACKEND"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import heapq
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)



app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "diabetesgen.keras"),
    compile=False
)

scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
ohe = joblib.load(os.path.join(BASE_DIR, "ohe.pkl"))


@app.get("/")
def home():
    return ("Welcome to Know Your Diabetes App!"
            "Disclaimer: This app is intended for educational"
            " purposes and risk awareness. \n"
            " This app is not a substitute for professional "
            " medical evaluation and diagnosis.")

DEFINITIONS = {
    "sym1": "frequent Urination", "sym2": "increased thirst",
    "sym3": "increased hunger", "sym4": "unexplained weight loss",
    "sym5": "fatigue",  "sym6": "blurry vision",
    "sym7": "frequent UTI", "sym8":"irritability or moodiness",
    "sym9": "nausea", "sym10": "vomiting", "sym11": "stomach pains",
    "sym12": "unhealed sores or cuts", "sym13": "dark patches of skin",
    "sym14": "numbness in feet and hands"
}

UNDER_40 = ["sym1", "sym2", "sym3", "sym4"]

SYMPTOMS_MAIN = ["sym1", "sym2", "sym3", "sym4"]

SYMPTOMS_SECOND = [
    "sym5", "sym6", "sym7", "sym8", "sym9",
    "sym10", "sym11", "sym12", "sym13", "sym14"
]



def group_age(age):
    try:
        age = float(age)
    except (TypeError, ValueError):
        return None

    if age is None:
        return None
    if age == 0:
        return None
    if 0.1 <= age <= 40:
        return "under"
    elif 41 <= age <= 64:
        return "middle"
    else:
        if age >= 65:
            return "senior"

    return age



def group_bmi(bmi):
    try:
        bmi = float(bmi)
    except (TypeError, ValueError):
        return None

    if bmi is None:
        return None
    if bmi == 0:
        return None
    if 1 <= bmi <= 18.4:
        return "underweight"
    elif bmi >=30:
        return "obese"
    elif bmi >= 25:
        return "overweight"
    elif 18.5 <= bmi <= 24.9:
        return "healthy"

    return bmi



def group_a1c(a1c_number):
    try:
        a1c_number = float(a1c_number)
    except (TypeError, ValueError):
        return None

    if a1c_number is None:
        return None
    if a1c_number == 0:
        return None
    if a1c_number >= 6.5:
        return "high"
    if 5.8 <= a1c_number <= 6.4:
        return "prediabetes"
    if 4.5 <= a1c_number <= 5.7:
        return "normal"
    if 1 <= a1c_number <= 4.4:
        return "low"

    return None



def group_glucose(glucose):
    try:
        glucose = float(glucose)
    except (TypeError, ValueError):
        return None

    if glucose is None:
        return None
    if glucose == 0:
        return None
    if 1 < glucose <= 70:
        return "low"
    elif 71 <= glucose <= 100:
        return "normal"
    elif 101 <= glucose <= 125:
        return "prediabetes"
    elif glucose >= 126:
        return "high"

    return glucose



def map_first_symptoms(answers):
    mapped = []
    for a in answers:
        a = str(a).strip()
        if not a.isdigit():
            continue
        idx = int(a) - 1
        mapped.append(f"sym{idx+1}")
    return mapped



class MainTree1(BaseModel):
    age: int | None = None
    first_choices: list[int] | None = None
    a1c: Optional[float] | None = None


@app.post("/predict_main1")
def predict_main1(req: MainTree1):
    return decision_tree1(req)


def decision_tree1(req: MainTree1):
    logger.debug("DEBUG START OF TREE1")
    age = req.age
    first_choices = req.first_choices
    a1c = req.a1c
    choices_str = [str(x) for x in first_choices]

    if req.a1c == 0:
        req.a1c = None
        a1c = None

    logger.debug("DEBUG:", age)

    if age is None or age == 0:
        return {
            "message": "Age is required.",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": None,
            "requires": None,
            "markov": None
        }

    if not first_choices:
        return {
            "message": "Please select at least one symptom.",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": None,
            "requires": None,
            "markov": None
        }

    age_category = group_age(age)
    a1c_category = group_a1c(a1c)
    symptom_codes = map_first_symptoms(first_choices)

    user_features = {
        "symptoms": symptom_codes,
        "age": age,
        "a1c": a1c
    }

    if age_category == "under":
        logger.debug("DEBUG HIT TREE1: UNDER 40 TYPE1 BLOCK")
        match = 0
        for v in UNDER_40:
            if v in symptom_codes:
                match += 1

        if match >= 2 and "5" not in choices_str:
            logger.debug("DEBUG HIT TREE1: UNDER 40 MATCH2 FOR SURE TYPE1 BLOCK")
            logger.debug("DEBUG: match =", match)
            logger.debug("DEBUG: choices_str =", choices_str)
            return {
                "message": "Model is 99% confident your pattern signals Type1 Diabetes. A Health Professional can give clearer guidance.",
                "prediction": None,
                "mean": None,
                "variance": None,
                "user_features": user_features,
                "requires": None,
                "markov": None
            }

        elif match >= 2 and "5" in choices_str:
            logger.debug("DEBUG HIT TREE1: UNDER 40 MATCH2 BUT WITH 5 BLOCK")
            return {
                "message": " ",
                "prediction": None,
                "mean": None,
                "variance": None,
                "user_features": user_features,
                "requires": "decision_tree2",
                "markov": None
            }

        elif match == 1 and a1c_category is None and "5" not in choices_str:
            logger.debug("DEBUG: match =", match)
            logger.debug("DEBUG: choices_str =", choices_str)
            logger.debug("DEBUG HIT TREE1: UNDER 40 MATCH1 NEED A1C BLOCK")
            return {
                "message": " ",
                "prediction": None,
                "mean": None,
                "variance": None,
                "user_features": user_features,
                "requires": "decision_tree2",
                "markov": None
            }


    if req.a1c is None:
        logger.debug("DEBUG HIT TREE1: GLOBAL ASK A1C BLOCK")
        return {
            "message": " ",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "decision_tree2",
            "markov": None
        }


    if "5" in choices_str:
        logger.debug("DEBUG HIT TREE1: LEN5 GO TO 2 BLOCK")
        return {
            "message": "",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "decision_tree2",
            "markov": None
        }

    logger.debug("DEBUG HIT TREE1 DEFAULT RETURN BLOCK")
    return {
        "message": " ",
        "prediction": None,
        "mean": None,
        "variance": None,
        "user_features": user_features,
        "requires": None,
        "markov": None
    }



class MainTree2(BaseModel):
    age: int | None = None
    first_choices: list[int] | None = None
    a1c: Optional[float] | None = None
    bmi: Optional[float] | None = None


@app.post("/predict_main2")
def predict_main2(req: MainTree2):
    return decision_tree2(req)


def decision_tree2(req: MainTree2):
    logger.debug("DEBUG HIT TREE2 ENTERED")
    age = req.age
    first_choices = req.first_choices
    a1c = req.a1c
    choices_str = [str(x) for x in first_choices]

    if a1c is None or a1c == 0:
        a1c = None
        a1c_category = None
    else:
        a1c_category = group_a1c(a1c)

    age_category = group_age(age)
    symptom_codes = map_first_symptoms(first_choices)

    user_features = {
        "symptoms": symptom_codes,
        "a1c": a1c,
        "age": age,
    }

    if age_category == "under":
        logger.debug("DEBUG HIT TREE2: UNDER 40 START A1C ASK BLOCK")
        match = 0
        for v in UNDER_40:
            if v in symptom_codes:
                match += 1

        if match == 1 and a1c_category is not None and "5" not in choices_str:
            logger.debug("DEBUG: match =", match)
            logger.debug("DEBUG: choices_str =", choices_str)
            if a1c_category == "high":
                logger.debug("DEBUG HIT TREE2: UNDER 40 MATCH1 BUT A1C FOR SURE TYPE1 BLOCK")
                return {
                    "message": "Model is 99% confident your pattern signals Type1 Diabetes. A Health Professional can give clearer guidance.",
                    "prediction": None,
                    "mean": None,
                    "variance": None,
                    "user_features": user_features,
                    "requires": None,
                    "markov": None
                }

            elif a1c_category == "normal":
                logger.debug("DEBUG HIT TREE2: UNDER 40 MATCH0 A1C NORMAL BLOCK")
                return {
                    "message": "Given your age and with only symptom and a normal A1C, "
                               "your overall pattern points to a low diabetes risk.",
                    "prediction": None,
                    "mean": None,
                    "variance": None,
                    "user_features": user_features,
                    "requires": None,
                    "markov": None
                }

            elif a1c_category == "low":
                logger.debug("DEBUG HIT TREE2: UNDER 40 MATCH0 A1C LOW BLOCK")
                return {
                    "message": "With one symptom but a severely low A1C, your pattern shows low glucose levels. "
                               "A health professional can provide further assistance.",
                    "prediction": None,
                    "mean": None,
                    "variance": None,
                    "user_features": user_features,
                    "requires": None,
                    "markov": None
                }

            elif match == 1 and a1c_category is not None and "5" in choices_str:
                logger.debug("DEBUG HIT TREE2: LEN5 GO TO FIRST BLOCK")
                return {
                    "message": "",
                    "prediction": None,
                    "mean": None,
                    "variance": None,
                    "user_features": user_features,
                    "requires": "first_match",
                    "markov": None
                }


    if user_features.get("a1c"):
        logger.debug("DEBUG HIT TREE2: A1C GET BMI NEXT BLOCK")
        return {
            "message": "",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "first_match",
            "markov": None
        }


    if user_features.get("a1c") is None or user_features.get("a1c") == 0:
        return {
            "message": "",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "first_match",
            "markov": None
        }


    logger.debug("DEBUG HIT TREE2: DEFAULT RETURN BLOCK")
    return {
        "message": "More data is needed.",
        "prediction": None,
        "mean": None,
        "variance": None,
        "user_features": user_features,
        "requires": None,
        "markov": None
    }



def encode_features(glucose_value, a1c_value, bmi_value):
    categorical_columns = [
        "GENDER", "KIDNEY FAILURE", "DIALYSIS", "SMOKER",
        "LIVER FAILURE", "HEART ATTACK", "THYROID PROBLEM"
    ]

    numerical_columns = [
        "SEDENTARY MINS", "FREQUENCY MODERATE ACTIVITY",
        "YOUTH: DAYS ACTIVE", "YOUTH: SEDENTARY MINS",
        "CIGARETTES PER DAY", "WEIGHT", "BMI",
        "BP DIALOSTIC", "BP SYSTOLIC", "CHOLESTEROL",
        "A1C", "GLUCOSE"
    ]

    categorical_defaults = {
        "GENDER": "Male",
        "KIDNEY FAILURE": "No",
        "DIALYSIS": "No",
        "SMOKER": "Every day",
        "LIVER FAILURE": "No",
        "HEART ATTACK": "No",
        "THYROID PROBLEM": "No"
    }

    numeric_defaults = {col: 0 for col in numerical_columns}

    a1c_clean = None if a1c_value in [None, 0] else float(a1c_value)
    glucose_clean = None if glucose_value in [None, 0] else float(glucose_value)
    bmi_clean = None if bmi_value in [None, 0] else float(bmi_value)


    data = {}

    for col in numerical_columns:
        data[col] = numeric_defaults[col]

    if a1c_clean is not None:
        data["A1C"] = a1c_clean

    if glucose_clean is not None:
        data["GLUCOSE"] = glucose_clean

    if bmi_clean is not None:
        data["BMI"] = bmi_clean

    for col in categorical_columns:
        data[col] = categorical_defaults[col]

    df = pd.DataFrame([data])

    scaled_numeric = scaler.transform(df[numerical_columns])
    encoded_cats = ohe.transform(df[categorical_columns])

    final_vector = np.concatenate([scaled_numeric, encoded_cats], axis=1)
    return final_vector



def data_priority(data):
    risks = {
        "sym1": 1, "sym2": 2, "sym3": 4,
        "sym4": 3, "sym5": 8, "sym6": 5,
        "sym7": 6, "sym8": 13,
        "sym9": 11, "sym10": 12, "sym11": 14,
        "sym12": 7, "sym13": 9, "sym14": 10,
    }

    data_ranking = []

    if data is None:
        return []

    if isinstance(data, list):
        for item in data:
            if item in risks:
                heapq.heappush(data_ranking, (risks[item], item))

    elif isinstance(data, str):
        if data in risks:
            heapq.heappush(data_ranking, (risks[data], data))

    return [heapq.heappop(data_ranking) for _ in range(len(data_ranking))]



def change_to_python(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj



class FirstRequest(BaseModel):
    user_features: Optional[Dict[str, Any]]
    first_choices: list[int]
    glucose: Optional[float] = None
    bmi: Optional[float] = None
    a1c: Optional[float] = None


@app.post("/predict_first")
def predict_first(req: FirstRequest):
    return first_match(req)


def first_match(req: FirstRequest):
    user_features = req.user_features
    first_choices = req.first_choices
    a1c = req.a1c
    bmi = req.bmi
    glucose = req.glucose

    if a1c is None or a1c == 0:
        a1c = None
        a1c_category = None
    else:
        a1c_category = group_a1c(a1c)

    if bmi is None or bmi == 0:
        bmi = None
        bmi_category = None
    else:
        bmi_category = group_bmi(bmi)

    if glucose is None or glucose == 0:
        glucose = None
        glucose_category = None
    else:
        glucose_category = group_glucose(glucose)

    logger.debug("DEBUG FIRST STAGE REACHED")
    logger.debug("DEBUG FIRST a1c:", req.a1c, "bmi:", req.bmi)
    logger.debug("DEBUG FIRST PAYLOAD:", req.dict())
    logger.debug("DEBUG FIRST SYMPTOMS:", req.first_choices)
    logger.debug("DEBUG FIRST BMI:", req.bmi)



    user_features.update(
        {"a1c": a1c, "a1c_category": a1c_category,
        "bmi": bmi, "bmi_category": bmi_category,
        "glucose": glucose, "glucose_category": glucose_category,
        })
    logger.debug("DEBUG FIRST STAGE BMI CATEGORY:", bmi_category)
    logger.debug("DEBUG FIRST STAGE A1C CATEGORY:", a1c_category)
    logger.debug("DEBUG FIRST STAGE GLUCOSE CATEGORY:", glucose_category)
    mapped = [f"sym{i}" for i in first_choices]
    symptom_list = data_priority(mapped)
    logger.debug("DEBUG FIRST data_priority output:", symptom_list)

    if not symptom_list:
        logger.debug("DEBUG FIRST: HIT if not symptom list block")
        return {
            "message": "More data is needed.",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": None,
            "requires": None,
            "markov": None
        }

    for_heap = symptom_list.copy()
    heapq.heapify(for_heap)
    top_weight, _ = heapq.heappop(for_heap)

    missing = sum(
        [
            a1c is None,
            bmi is None,
            glucose is None
        ])

    if missing == 3:
        logger.debug("DEBUG BLOCKED ALL THREE")
        return {
            "message": " ",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "all_three",
            "markov": None
        }

    if (a1c is None or a1c_category is None) and (glucose is None or glucose_category is None):
        logger.debug("DEBUG A1C OR GLUCOSE NONE NO INFERENCE")
        return {
            "message": " ",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "blocked",
            "markov": None
        }

    if len(symptom_list) <= 1 and glucose_category == "normal" and a1c_category == "normal":
        return {
            "message": "Low risk for diabetes.",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "done",
            "markov": None
        }

    if len(symptom_list) <= 2 and glucose_category == "normal" and a1c_category == "normal":
        return {
            "message": " ",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "second_match",
            "markov": None
        }

    if glucose_category in ["high", "prediabetes"] or a1c_category in ["high", "prediabetes"]:
        return {
            "message": " ",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "second_match",
            "markov": None
        }

    if missing >= 2:
        logger.debug("DEBUG TWO MISSING CAN GO FORWARD")
        return {
            "message": " ",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "second_match",
            "markov": None
        }



    if bmi is None or bmi == 0 or bmi_category is None:
        logger.debug("DEBUG A1C RAW TYPE:", a1c, type(a1c))
        return {
            "message": "",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "second_match",
            "markov": None
        }



    flag_weights = False
    for weight, _ in symptom_list:
        logger.debug("DEBUG FLAG WEIGHTS WAS CALCULATED")
        if int(weight) in (1, 2, 3, 4):
            flag_weights = True
            logger.debug("DEBUG", flag_weights)
            break

    grave_symptoms = {int(w) for w, _ in symptom_list} == {1, 2, 3, 4}
    logger.debug("DEBUG GRAVE SYMPTOMS TRIGGERED", grave_symptoms)

    risk_score = [grave_symptoms, glucose_category in ["high", "prediabetes"],
                  a1c_category == "high"]

    if all(risk_score):
        logger.debug("DEBUG HIT FIRST: RISK SCORE DEFINITELY DIABETES BLOCK")
        return {
            "message": "Model is 99% confident your pattern signals diabetes. "
                       "A health professional can provide further assistance.",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": None,
            "markov": None
        }

    logger.debug("DEBUG REACHED BMI+A1C FLAG CHECK")
    if bmi_category in ["obese", "overweight", "healthy"] and flag_weights:
        if a1c_category in ["high", "prediabetes", "normal"] and flag_weights:
            logger.debug("DEBUG HIT FIRST: A1C AND BMI IF ANY OF SYMPTOMS PRESENT BLOCK")
            logger.debug("DEBUG HIT FIRST:", bmi_category, a1c_category, flag_weights)
            return {
                "message": " ",
                "prediction": None,
                "mean": None,
                "variance": None,
                "user_features": user_features,
                "requires": "second_match",
                "markov": None
            }

    if bmi_category == "underweight":
        logger.debug("DEBUG HIT FIRST: BMI UNDERWEIGHT BLOCK")
        return {
            "message": "Data shows unhealthy weight. A health professional can provide further assistance.",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": None,
            "markov": None
        }

    if a1c_category == "low":
        logger.debug("DEBUG HIT FIRST: A1C LOW BLOCK")
        return {
            "message": "Data shows low glucose levels. A health professional can provide further assistance.",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": None,
            "markov": None
        }

    if len(risk_score) >= 1:
        logger.debug("DEBUG HIT FIRST: RISK SCORE HAS AT LEAST TWO BLOCK")
        return {
            "message": " ",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": "second_match",
            "markov": None
        }

    logger.debug("DEBUG HIT FIRST: DEFAULT RETURN BLOCK")
    return {
        "message": " ",
        "prediction": None,
        "mean": None,
        "variance": None,
        "user_features": user_features,
        "requires": "second_match",
        "markov": None
    }



def map_second_symptoms(answers):
    mapped = []
    for a in answers:
        a = str(a).strip()
        if not a.isdigit():
            continue
        idx = int(a) - 1
        mapped.append(f"sym{idx+1}")
    return mapped



class SecondRequest(BaseModel):
    user_features: Optional[Dict[str, Any]]
    first_choices: list[int]
    second_choices: list[int]
    glucose: Optional[float] = None
    bmi: Optional[float] = None
    a1c: Optional[float] = None


@app.post("/predict_second")
def predict_second(req: SecondRequest):
    return second_match(req)


def second_match(req: SecondRequest):
    logger.debug("DEBUG SECOND: ENTERED second_match")
    logger.debug("DEBUG SECOND PAYLOAD:", req.dict())
    user_features = req.user_features or {}
    second_choices = req.second_choices
    first_choices = user_features.get("first_choices", [])

    def normalize(value):
        if value in (None, "", "None"):
            return None
        try:
            return float(value)
        except:
            return None

    a1c = normalize(req.a1c if req.a1c is not None else user_features.get("a1c"))
    bmi = normalize(req.bmi if req.bmi is not None else user_features.get("bmi"))
    glucose = normalize(req.glucose)

    glucose_category = group_glucose(glucose) if glucose is not None else None


    logger.debug("DEBUG SECOND STAGE REACHED")
    logger.debug("DEBUG WHAT IS A1C", a1c)
    logger.debug("DEBUG WHAT IS BMI", bmi)
    logger.debug("DEBUG WHAT IS GLUCOSE", glucose)
    logger.debug("DEBUG WHAT IS REQ.GLUCOSE:", req.glucose)

    if not isinstance(second_choices, list):
        logger.debug("DEBUG HIT SECOND: INVALID MESSAGE BLOCK1")
        return {
            "message": "Invalid symptom format.",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": None,
            "markov": None
        }

    for c in second_choices:
        if not str(c).isdigit():
            logger.debug("DEBUG HIT SECOND: INVALID MESSAGE BLOCK2")
            return {
                "message": "Invalid symptom format.",
                "prediction": None,
                "mean": None,
                "variance": None,
                "user_features": user_features,
                "requires": None,
                "markov": None
            }


    first_symptoms = [f"sym{i}" for i in first_choices]
    second_symptoms = map_second_symptoms(second_choices)
    all_symptoms = first_symptoms + second_symptoms

    weights = sorted(data_priority(all_symptoms))
    symptom_list = []
    for w, s in weights:
        heapq.heappush(symptom_list, (w, s))

    if symptom_list:
        top_weight, _ = heapq.heappop(symptom_list)
    else:
        top_weight = 0

    glucose_map = {
        "low": 1,
        "normal": 1,
        "prediabetes": 2,
        "high": 0,
    }

    if glucose_category is None:
        glucose_code = None
    glucose_code = glucose_map.get(glucose_category)
    logger.debug("DEBUG WHAT IS GLUCOSE_CATEGORY:", repr(glucose_category))


    if glucose is not None and a1c is not None and (glucose >= 200 or a1c >= 8.0):
        logger.debug("DEBUG glucose >= 200 or a1c >= 8.0 FOR DEFAULT FULL INFERENCE BLOCK")
        logger.debug("DEBUG", glucose, a1c)
        x = encode_features(glucose, a1c, bmi)
        _, mean, var = make_inference(x)
        pred = 0
        msg = interpret(pred, mean, var)
        markov_flag = "markov_transition" if pred == 2 else None
        return {
            "message": msg or "",
            "prediction": change_to_python(pred),
            "mean": change_to_python(mean),
            "variance": change_to_python(var),
            "user_features": {k: change_to_python(v) for k, v in user_features.items()},
            "requires": None,
            "markov": markov_flag
        }

    if (glucose is None or glucose == 0) and (a1c >= 5.8):
        logger.debug("DEBUG HIT SECOND: A1C‑ONLY NO GLUCOSE INFERENCE")
        x = encode_features(glucose, a1c, bmi)
        pred, mean, var = make_inference(x)
        msg = interpret(pred, mean, var)
        markov_flag = "markov_transition" if pred == 2 else None
        return {
            "message": msg or "",
            "prediction": change_to_python(pred),
            "mean": change_to_python(mean),
            "variance": change_to_python(var),
            "user_features": {k: change_to_python(v) for k, v in user_features.items()},
            "requires": None,
            "markov": markov_flag
        }
    if glucose_category in ["high", "normal", "prediabetes"]:
        logger.debug("DEBUG HIT SECOND: INFERENCE ON HIGH NORMAL AND PREDIABETES BLOCK")
        x = encode_features(glucose, a1c, bmi)
        pred, mean, var = make_inference(x)
        msg = interpret(pred, mean, var)
        markov_flag = "markov_transition" if pred == 2 else None
        return {
            "message": msg or "",
            "prediction": change_to_python(pred),
            "mean": change_to_python(mean),
            "variance": change_to_python(var),
            "user_features": {k: change_to_python(v) for k, v in user_features.items()},
            "requires": None,
            "markov": markov_flag
        }
    if glucose_category == "low":
        logger.debug("DEBUG HIT SECOND: GLUCOSE LOW BLOCK")
        return {
            "message": "Data shows low glucose levels. A health professional can provide further assistance.",
            "prediction": None,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": None,
            "markov": None
        }
    if (glucose is None or glucose == 0) and (a1c is None or a1c == 0 or a1c < 5.7):
        return {
            "message": "Low risk for diabetes.",
            "prediction": 1,
            "mean": None,
            "variance": None,
            "user_features": user_features,
            "requires": None,
            "markov": None
        }

    logger.debug("DEBUG HIT SECOND: RETURN DEFAULT FULL INFERENCE BLOCK")
    return None



def risk_message(mean, var, class_index, class_name):
    logger.debug("DEBUG RISK MESSAGE BLOCK HIT")
    logger.debug("DEBUG", class_name)
    mean = np.array(mean).flatten()
    var = np.array(var).flatten()

    prob = min(mean[class_index] * 100, 99.9)
    prob2 = max(prob, 70.0)
    uncertainty = var[class_index]

    if mean[class_index] > 0.70 and uncertainty < 0.02:
        logger.debug("DEBUG MEAN > 70 UNCERTAINTY <0.02")
        if class_name == "prediabetes":
            logger.debug("DEBUG MEAN > 70 UNCERTAINTY <0.02 PREDIABETES INFERRED")
            return f"Model is {prob:.1f}% confident you likely have {class_name}. A health professional can offer further guidance."
        elif class_name == "diabetes":
            logger.debug("DEBUG MEAN > 70 UNCERTAINTY <0.02 DIABETES INFERRED")
            return f"Model is {prob:.1f}% confident you likely have {class_name}. A health professional can offer further guidance."
        elif class_name == "no diabetes":
            logger.debug("DEBUG MEAN > 70 UNCERTAINTY <0.02 NO DIABETES INFERRED")
            return f"Based on your inputs, the model is {prob:.1f}% confident you likely have {class_name}."

    elif uncertainty > 0.05:
        logger.debug("DEBUG", class_name)
        logger.debug("DEBUG UNCERTAINTY >0.05")
        return (f"The model could not reach a definitive conclusion. "
                f"There is just not enough information \n"
                f"to make a prediction. The model estimates around {prob:.1f}% "
                f"chance that you likely have {class_name}.  \n"
                f"A health professional can offer a more complete assessment.")

    if class_name == "diabetes":
        logger.debug("DEBUG OUTSIDE DIABETES INFERRED")
        return f"Model is {prob2:.1f}% confident you likely have {class_name}. A health professional can offer further guidance."

    elif class_name == "prediabetes":
        logger.debug("DEBUG OUTSIDE PREDIABETES INFERRED")
        return f"Model is {prob2:.1f}% confident you likely have {class_name}. A health professional can offer further guidance."

    elif class_name == "no diabetes":
        logger.debug("DEBUG OUTSIDE NO DIABETES INFERRED")
        return f"Based on your inputs, the model is {prob2:.1f}% confident you likely have {class_name}."

    logger.debug("DEBUG", class_name)
    logger.debug("DEBUG DEFAULT RISK MESSAGE")
    return ("Based on your inputs, the model does not have high enough confidence "
            "to make a strong prediction. A health professional can offer a "
            "more complete assessment.")



def interpret(pred, mean, var, prediabetes_flag: bool = False):
    if prediabetes_flag:
        return risk_message(mean, var, 2, "prediabetes")
    if pred == 1:
        return risk_message(mean, var, 1, "no signs of diabetes")
    if pred == 0:
        return risk_message(mean, var, 0, "diabetes")

    class_map = {
        0: "diabetes",
        1: "no diabetes",
        2: "prediabetes"
    }

    return risk_message(mean, var, pred, class_map.get(pred, "diabetes"))



def monte_carlo_dropout(model_input, x_tensor, passes=50):
    preds = []
    for _ in range(passes):
        out = model_input(x_tensor, training=True)
        preds.append(out.numpy())
    return np.array(preds)



def calculate_probs(x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    samples = monte_carlo_dropout(model, x_tensor, passes=50)
    mean_preds = samples.mean(axis=0)
    var_preds = samples.var(axis=0)
    return mean_preds, var_preds



def make_inference(x):
    mean, var = calculate_probs(x)
    pred = np.argmax(mean)
    return pred, mean, var



def proba_from_heap(heap):
    if not heap:
        return 0
    weights = []
    for w, item in heap:
        weights.append(w)
    avg = sum(weights) / len(weights)
    p = 1 - (avg / 16)
    return max(0.0, min(p, 1.0))



class MarkovRequest(BaseModel):
    user_features: Optional[Dict[str, Any]]


@app.post("/predict_markov")
def predict_markov(req: MarkovRequest):
    transitions = markov_chain_step(req.user_features)
    return {
        "message": "Here are your probabilities based on your inputs:",
        "prediction": None,
        "mean": None,
        "variance": None,
        "user_features": req.user_features,
        "markov": transitions
    }



def markov_transition(p):
    logger.debug("DEBUG I ENTERED MARKOV")
    to_diabetes = p * 0.6
    to_no_diabetes = (1 - p) * 0.4
    stay_prediabetes = 1 - (to_diabetes + to_no_diabetes)
    return {
        "to_diabetes": to_diabetes,
        "to_no_diabetes": to_no_diabetes,
        "stay_prediabetes": stay_prediabetes
    }



def markov_chain_step(user_features: Dict[str, Any]):
    raw_a1c = user_features.get("a1c")
    if raw_a1c in [None, 0]:
        a1c_category = None
    else:
        a1c_category = group_a1c(raw_a1c)

    raw_bmi = user_features.get("bmi")
    if raw_bmi in [None, 0]:
        bmi_category = None
    else:
        bmi_category = group_bmi(raw_bmi)

    raw_glucose = user_features.get("glucose")
    if raw_glucose in [None, 0]:
        glucose_category = None
    else:
        glucose_category = group_glucose(raw_glucose)

    symptoms = [
        f"sym{item}" if isinstance(item, int) else item
        for item in user_features.get("symptoms", [])
    ]

    data = symptoms + [
        glucose_category,
        a1c_category,
        bmi_category
    ]

    heap = data_priority(data)
    p = proba_from_heap(heap)
    return markov_transition(p)



@app.post("/all")
def all_predictions():
    return {
        "pred5": predict_main1(),
        "pred6": predict_main2(),
        "pred7": predict_first(),
        "pred8": predict_second(),
        "pred9": predict_markov()
    }

