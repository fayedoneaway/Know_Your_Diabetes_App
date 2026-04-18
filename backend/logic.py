import tensorflow as tf
import numpy as np
import heapq
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "diabetesgen.keras"),
    compile=False
)

scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
ohe = joblib.load(os.path.join(BASE_DIR, "ohe.pkl"))


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
    var_preds  = samples.var(axis=0)

    return mean_preds, var_preds



def make_inference(x):
    mean, var = calculate_probs(x)
    pred = np.argmax(mean)
    return pred, mean, var



DEFINITIONS = {
    "sym1": "frequent Urination", "sym2": "increased thirst",
    "sym3": "increased hunger", "sym4": "unexplained weight loss",
    "sym5": "fatigue",  "sym6": "blurry vision",
    "sym7": "frequent UTI", "sym8":"irritability or moodiness",
    "sym9": "nausea", "sym10": "vomiting", "sym11": "stomach pains",
    "sym12": "unhealed sores or cuts", "sym13": "dark patches of skin",
    "sym14": "numbness in feet and hands"
}



GRAPH_OVERALL = {
    "under_40": ["sym1", "sym2", "sym3", "sym4"],
    "overweight_bmi": ["obese_bmi", "diabetes_a1c", "middle_crowd"],
    "diabetes_a1c": ["diabetes_glucose", "sym1", "prediabetes_a1c"],
    "sym4": ["overweight_bmi", "obese_bmi", "middle_crowd", "male"]
}



GRAPH_SYMPTOM = {
    "sym1": ["sym2",  "sym3",  "sym4"],
    "sym4": ["sym1", "sym2", "sym3"],
    "sym3": ["sym8", "sym12", "sym14"],
    "sym14": ["sym5", "sym10", "sym11"],
    "sym2": ["sym1", "sym6", "sym9", "sym13"],
    "sym13": ["sym4", "sym14", "sym12"],
    "sym12": ["sym8", "sym9", "sym10", "sym6"],
    "sym6": ["sym3", "sym7", "sym11"],
    "sym7": ["sym1", "sym8", "sym12"],
    "sym5": ["sym11", "sym9", "sym10", "sym12"],
    "sym9": ["sym10", "sym1", "sym2"],
    "sym10": ["sym3", "sym12", "sym5", "sym7"],
    "sym11": ["sym13", "sym5"],
    "sym8": ["sym12", "sym13"]
    }


def encode_features(glucose_code, symptom_count):
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
        "GENDER": 1,
        "KIDNEY FAILURE": 2,
        "DIALYSIS": 2,
        "SMOKER": 3,
        "LIVER FAILURE": 2,
        "HEART ATTACK": 2,
        "THYROID PROBLEM": 2
    }

    numeric_defaults = {col: 0 for col in numerical_columns}

    categorical_defaults = {
        "GENDER": "Male",
        "KIDNEY FAILURE": "No",
        "DIALYSIS": "No",
        "SMOKER": "Every day",
        "LIVER FAILURE": "No",
        "HEART ATTACK": "No",
        "THYROID PROBLEM": "No"
    }

    data = {}

    for col in numerical_columns:
        data[col] = numeric_defaults[col]

    data["GLUCOSE"] = glucose_code
    data["A1C"] = symptom_count

    for col in categorical_columns:
        data[col] = categorical_defaults[col]

    df = pd.DataFrame([data])

    scaled_numeric = scaler.transform(df[numerical_columns])
    encoded_cats = ohe.transform(df[categorical_columns])

    final_vector = np.concatenate([scaled_numeric, encoded_cats], axis=1)

    return final_vector



def catch_numbers(answer):
    try:
        return float(answer)
    except ValueError as e:
        print("Numbers only please.")
        return None



def capture_info(data, info):
    DEFINITIONS[f"{data}:{info}"] = data



def get_gender(female, male):
    return female, male



def get_age(age):
    age = catch_numbers(age)
    if age <= 40:
        return "under_40"
    elif 41 <= age <= 64:
        return "middle_crowd"
    else:
        if age >= 65:
            return "senior_crew"

    return age



def get_bmi():
    answer = input("Enter your BMI: (Enter 0 if unknown) ").strip()
    bmi = catch_numbers(answer)
    if 0 <= bmi <= 18.4:
        return "underweight"
    elif bmi >=30:
        return "obese_bmi"
    elif bmi >= 25:
        return "overweight_bmi"
    elif 18.5 <= bmi <= 24.9:
        return "healthy_bmi"

    return bmi



def get_a1c():
    answer = input("Whats is your A1C: (Enter 0 if unknown) ").strip()
    a1c = catch_numbers(answer)
    if 0 <= a1c <= 4.4:
        return "low_glucose"
    elif a1c >= 6.5:
        return "diabetes_a1c"
    elif 4.5 <= a1c <= 5.7:
        return "normal_a1c"
    elif 5.8 <= a1c <= 6.4:
        return "prediabetes_a1c"

    return a1c



def get_glucose():
    answer = input("Whats is your glucose level: (Enter 0 if unknown) ").strip()
    glucose = catch_numbers(answer)
    if glucose == 0:
        return "unknown_glucose"
    if 0 < glucose <= 70:
        return "low_glucose"
    elif 71 <= glucose <= 100:
        return "normal_glucose"
    elif 101 <= glucose <= 125:
        return "prediabetes_glucose"
    elif glucose >= 126:
        return "diabetes_glucose"

    return glucose



def build_features(age, a1c, glucose, bmi):
    x = []
    x.append(1 if age == "under_40" else 0)
    x.append(1 if age == "middle_crowd" else 0)
    x.append(1 if age == "senior_crew" else 0)

    x.append(1 if glucose == "low_glucose" else 0)
    x.append(1 if glucose == "normal_glucose" else 0)
    x.append(1 if glucose == "prediabetes_glucose" else 0)
    x.append(1 if glucose == "diabetes_glucose" else 0)

    # bmi category (one-hot)
    x.append(1 if bmi == "underweight" else 0)
    x.append(1 if bmi == "healthy_bmi" else 0)
    x.append(1 if bmi == "overweight_bmi" else 0)
    x.append(1 if bmi == "obese_bmi" else 0)

    x.append(1 if a1c == "low_glucose" else 0)
    x.append(1 if a1c == "diabetes_a1c" else 0)
    x.append(1 if a1c == "normal_a1c" else 0)
    x.append(1 if a1c == "prediabetes_a1c" else 0)

    return x



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



SYMPTOMS_FIRST = ["sym1", "sym2", "sym3", "sym4"]

SYMPTOMS_SECOND = ["sym5", "sym6", "sym7", "sym8", "sym9",
                   "sym10", "sym11", "sym12", "sym13", "sym14"]



def get_first_symptoms():
    symptoms_main = [
        "Frequent Urination",
        "Increased Thirst",
        "Increased Hunger",
        "Unexplained Weight Loss",
        "Other symptoms",
        "No symptoms"
    ]

    for i, s in enumerate(symptoms_main, 1):
        print(f"{i}. {s}")

    answers = input(
        "Enter the number matching your symptom(s). "
        "Separate answers by spaces then hit enter: "
    ).strip().lower()

    choices = []
    for c in answers.split():
        if c.isdigit():
            num = int(c)
            if 1 <= num <= len(symptoms_main):
                choices.append(c)

    if not choices:
        return None, "Invalid input."

    if "6" in choices and len(choices) == 1:
        return [], "no_symptoms"

    elif "6" in choices and len(choices) > 1:
        return [], "invalid"

    elif "5" in choices and len(choices) == 1:
        return ["5"], "other_symptoms"

    elif "5" in choices:
        return ["5"], "other_symptoms"

    real_choices = []
    for c in choices:
        if c in {"1", "2", "3", "4"}:
            real_choices.append(c)

    return real_choices, None



def map_first_symptoms(answers):
    mapped = []
    for a in answers:

        if not a.isdigit():
            continue

        idx = int(a) - 1
        mapped.append(SYMPTOMS_FIRST[idx])
    return mapped



def first_match(symptoms, visited):
    user_features = {
        "symptoms": symptoms,
        "a1c": "unknown",
        "bmi": "unknown"
    }

    if visited == "invalid":
        return ("Invalid selection. You cannot choose 'No symptoms' with other symptoms.", None, None, None,
                user_features)

    all_symptoms = symptoms + list(visited)
    symptom_list = data_priority(all_symptoms)
    top_weight, top_symptom = heapq.heappop(symptom_list)

    if not symptom_list:
        return "More data is needed.", None, None, None, user_features

    bmi = get_bmi()
    bmi_key = bmi.replace("_bmi", "")
    bmi_list = data_priority([bmi_key])
    user_features["bmi"] = bmi

    a1c = get_a1c()
    a1c_key = a1c.replace("_a1c", "")
    a1c_list = data_priority([a1c_key])
    user_features["a1c"] = a1c


    if not bmi_list:
        bmi_weight = 0
    else:
        bmi_weight = bmi_list[0][0]

    if not a1c_list:
        a1c_weight = 0  # healthy or neutral BMI
    else:
        a1c_weight = a1c_list[0][0]


    flag_any = False
    for weight, _ in symptom_list:
        if int(weight) in (1, 2, 3, 4):
            flag_any = True
            break

    first_two = heapq.nsmallest(2, symptom_list)
    flag2 = False
    for weight, _ in first_two:
        if int(weight) in (1, 2, 3, 4):
            flag2 = True
            break

    top_three = heapq.nsmallest(3, symptom_list)
    flag3 = False
    for weight, _ in top_three:
        if int(weight) in (1, 2, 3, 4):
            flag3 = True
            break

    all_four = heapq.nsmallest(4, symptom_list)
    flag4 = False
    for weight, _ in all_four:
        if int(weight) in (1, 2, 3, 4):
            flag4 = True
            break

    if bmi in ["obese", "overweight", "healthy"] and flag_any:
        return second_match(user_features["bmi"])
    elif bmi in ["obese", "overweight", "healthy"] and flag2:
        return second_match(user_features["bmi"])
    elif bmi in ["obese", "overweight", "healthy"] and flag3:
        return second_match(user_features["bmi"])
    elif bmi in ["obese", "overweight", "healthy"] and flag4:
        return second_match(user_features["bmi"])
    elif bmi == "underweight":
        return ("Data shows unhealthy weight. A health professional can provide further assistance.", None, None,
                None, user_features)

    if a1c in ["high", "prediabetes", "normal"] and flag_any:
        return second_match(user_features["a1c"])
    elif a1c in ["high", "prediabetes", "normal"] and flag2:
        return second_match(user_features["a1c"])
    elif a1c in ["high", "prediabetes", "normal"] and flag3:
        return second_match(user_features["a1c"])
    elif a1c in ["high", "prediabetes", "normal"] and flag4:
        return second_match(user_features["a1c"])
    elif a1c == "low":
        return ("Data shows low glucose levels. A health professional can provide further assistance.", None, None,
                None, user_features)


    risk_score = [(flag3, flag4), bmi == "obese", a1c == "high"]
    if all(risk_score):
        return ("Model is 99% confident your pattern signals diabetes. "
                "A health professional can provide further assistance."), None, None, None, user_features
    elif len(risk_score) <= 2:
        return "More data is needed", None, None, None, user_features
    else:
        return "Model is 99% confident your pattern does NOT signal diabetes.", None, None, None, user_features



def get_second_symptoms():
    symptoms_second = [
        "Fatigue",
        "Blurry Vision",
        "Frequent UTI",
        "Irritability or Moodiness",
        "Nausea",
        "Vomiting",
        "Stomach pains",
        "Unhealed sores or cuts",
        "Dark patches of skin",
        "Numbness in feet and hands",
        "None of these symptoms"
    ]

    print("Here are other diabetes symptoms:")
    for i, s in enumerate(symptoms_second, 1):
        print(f"{i}. {s}")

    answers = input(
        "Enter the number matching your symptom(s). "
        "Separate answers by spaces then hit enter: "
    ).strip().lower()

    choices = []
    for c in answers.split():
        digits = "".join(ch for ch in c if ch.isdigit())
        if digits:
            choices.append(digits)

    if "11" in choices:
        if len(choices) > 1:
            return [], "invalid_none_mixed"
        return [], "none_selected"

    real_choices = [c for c in choices if c in {str(i) for i in range(1, 11)}]
    return real_choices, None



def proba_from_heap(heap):
    if not heap:
        return 0
    weights = []
    for w, item in heap:
        weights.append(w)
    avg = sum(weights) / len(weights)
    p = 1 - (avg / 16)
    return max(0.0, min(p, 1.0))



def markov_transition(p):
    to_diabetes = p * 0.6
    to_no_diabetes = (1 - p) * 0.4
    stay_prediabetes = 1 - (to_diabetes + to_no_diabetes)

    return {
        "to_diabetes": to_diabetes,
        "to_no_diabetes": to_no_diabetes,
        "stay_prediabetes": stay_prediabetes
    }



def markov_chain_step(user_features):
    data = (
        user_features["symptoms"]
        + [user_features["glucose"], user_features["a1c"], user_features["bmi"]]
    )
    heap = data_priority(data)
    p = proba_from_heap(heap)
    return markov_transition(p)



def map_second_symptoms(answers):
    mapped = []
    for a in answers:
        idx = int(a) - 1
        mapped.append(SYMPTOMS_SECOND[idx])
    return mapped



def second_match(user_features):
    glucose = get_glucose().strip().lower()
    user_features["glucose"] = glucose

    choices, flag = get_second_symptoms()

    if flag == "none_selected":
        if glucose == "normal_glucose":
            return "Low risk for diabetes.", None, None, None, user_features
        elif glucose == "low_glucose":
            return (
                "Data shows low glucose levels. A health professional can provide further assistance.",
                None, None, None, user_features
            )
        else:
            return "Low risk for diabetes.", None, None, None, user_features

    if flag == "invalid_none_mixed":
        return (
            "Invalid selection. You cannot choose 'None of these symptoms' with other symptoms.",
            None, None, None, user_features
        )


    second_symptoms = map_second_symptoms(choices)
    user_features["symptoms"] = second_symptoms
    symptom_count = len(second_symptoms)

    glucose_map = {
        "low_glucose": 1,
        "normal_glucose": 1,
        "prediabetes_glucose": 2,
        "diabetes_glucose": 0
    }

    glucose_code = glucose_map[glucose]

    if glucose == "low_glucose":
        return ("Data shows low glucose levels. A health professional can provide further assistance.", None, None,
                None, user_features)
    elif glucose in ["high", "prediabetes", "normal"]:
        x = encode_features(glucose_code, symptom_count)
        pred, mean, var = make_inference(x)
        msg = interpret(pred, mean, var, pred == 2)
        return msg, pred, mean, var, user_features

    x = encode_features(glucose_code, symptom_count)
    pred, mean, var = make_inference(x)
    msg = interpret(pred, mean, var, pred == 2)
    return msg, pred, mean, var, user_features



def interpret(pred, mean, var, prediabetes_flag=False):
    def safe(msg):
        return msg if msg is not None else ""
    if pred is None or mean is None or var is None:
        return "Low risk for diabetes."
    if prediabetes_flag:
        return safe(risk_message(mean, var, 2, "prediabetes"))
    if pred == 1:
        return safe(risk_message(mean, var, 1, "no diabetes"))
    if pred == 0:
        return safe(risk_message(mean, var, 0, "diabetes"))

    class_map = {
        0: "diabetes",
        1: "no diabetes",
        2: "prediabetes"
    }

    return safe(risk_message(mean, var, pred, class_map.get(pred, "diabetes")))



def risk_message(mean, var, class_index, class_name):
    mean = np.array(mean).flatten()
    var = np.array(var).flatten()

    prob = min(mean[class_index] * 100, 99.9)
    uncertainty = var[class_index]

    if mean[class_index] > 0.70 and uncertainty < 0.02:
        return f"Model is {prob:.1f}% confident you likely have {class_name}. A health professional can offer further guidance."
    elif mean[class_index] < 0.30 and uncertainty < 0.02:
        if prob < 1.0:
            return (f"The nature of your inputs prompted the model to check for {class_name}. \n"
                f" After evaluating your pattern, the model didn’t find a clear signal. \n"
                f" The estimated likelihood was around {prob:.1f}%. "
                f"A health professional can offer a more complete assessment.")
        return None
    elif uncertainty > 0.05:
        return (f"The model could not reach a definitive conclusion. "
                f"There is just not enough information \n"
                f"to make a prediction. The model estimates around {prob:.1f}% "
                f"chance that you likely have {class_name}.  \n"
                f"A health professional can offer a more complete assessment.")
    return ("Based on your inputs, the model does not have high enough confidence "
            "to make a strong prediction. A health professional can offer a "
            "more complete assessment.")


def weighted_bfs(graph, start):
    visited = set()
    queue = []

    for s in start:
        heapq.heappush(queue, (data_priority(s), s))

    while queue:
        _, node = heapq.heappop(queue)

        if node not in visited:
            visited.add(node)

            for neighbor in graph[node]:
                heapq.heappush(queue, (data_priority(neighbor), neighbor))

    return visited



def tree(age):
    first_choices, flag = get_first_symptoms()

    if flag == "invalid":
        return (
            "Invalid selection. You cannot choose 'No symptoms' with other symptoms.",
            None, None, None, None
        )

    if flag == "no_symptoms":
        user_features = {
            "symptoms": [],
            "glucose": get_glucose().strip().lower(),
            "a1c": get_a1c(),
            "bmi": get_bmi(),
            "age_group": age
        }
        return "Low risk for diabetes.", None, None, None, user_features

    if flag == "other_symptoms":
        user_features = {
            "symptoms": [],
            "glucose": "unknown",
            "a1c": "unknown",
            "bmi": "unknown",
            "age_group": age
        }

        msg, pred, mean, var, user_features = second_match(user_features)
        return msg, pred, mean, var, user_features

    symptom_codes = map_first_symptoms(first_choices)
    visited_nodes = weighted_bfs(GRAPH_SYMPTOM, symptom_codes)

    user_features = {
        "symptoms": symptom_codes,
        "glucose": "unknown",
        "a1c": "unknown",
        "bmi": "unknown",
        "age_group": age
    }

    if age == "under_40":
        match = 0
        for v in GRAPH_OVERALL["under_40"]:
            if v in symptom_codes:
                match += 1
        if match >= 2:
            return (
                "Model is 99% confident your pattern signals Type1 Diabetes. "
                "A Health Professional can give clearer guidance."
            ), None, None, None, user_features
        elif match == 1:
            while True:
                numbers = catch_numbers(input("Enter your A1C: ").strip())
                if numbers is None:
                    print("Invalid A1C input. Please enter numbers.")
                    continue
                if numbers >= 6.5:
                    user_features["a1c"] = "high"
                    return (
                        "Model is 99% confident your pattern signals Type1 Diabetes. "
                        "A Health Professional can give clearer guidance."
                    ), None, None, None, user_features
                if 5.8 <= numbers <= 6.4:
                    user_features["a1c"] = "prediabetes"
                    return first_match(symptom_codes, visited_nodes)
                if 4.5 <= numbers <= 5.7:
                    user_features["a1c"] = "normal"
                    return "Low risk for diabetes.", None, None, None, user_features
                if 0 <= numbers <= 4.4:
                    user_features["a1c"] = "low"
                    return (
                        "Data shows low glucose levels. "
                        "A professional can provide further assistance."
                    ), None, None, None, user_features
                break

    msg, pred, mean, var, user_features = first_match(symptom_codes, visited_nodes)
    if msg is not None:
        return msg, pred, mean, var, user_features

    if age in ["middle_crowd", "senior_crew"]:
        return first_match(symptom_codes, visited_nodes)

    return "More data is needed.", None, None, None, user_features



def check_again():
    again = input("Do you want to do another diabetes symptom check? ").strip().lower()
    if again in ["y", "yes"]:
        return True
    elif again in ["n", "no"]:
        print("Thank you for using the Know Your Diabetes App!")
        return False
    else:
        print("Please enter yes or no.")



def main():
    print("Welcome to Know Your Diabetes App!")
    print("Disclaimer: This app is intended for educational"
          " purposes and risk awareness. \n"
          " This app is not a substitute for professional "
          " medical evaluation and diagnosis.")
    while True:
        while True:
            age = input("Enter your age: ").strip()
            age = catch_numbers(age)
            if age is not None:
                break
            print("Please enter a valid number.")
        age_group = get_age(age)
        print("Here is the list of Diabetes symptoms:")
        result, pred, mean, var, user_features = tree(age_group)
        print(result)
        if pred == 2 and mean[pred] > 0.01:
            further = input("Would you like to see how this risk could change over time? ")
            if further.lower() in ["y", "yes"]:
                chances = markov_chain_step(user_features)
                print("\nBased on your inputs, here’s how your risk may shift over time:")
                print(f"- Chance of progressing to diabetes is: {chances['to_diabetes']:.1%}")
                print(f"- Chance of staying in prediabetes is: {chances['stay_prediabetes']:.1%}")
                print(f"- Chance of improving is: {chances['to_no_diabetes']:.1%}")
        if not check_again():
             break


if __name__ == "__main__":
    main()
