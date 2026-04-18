import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

if "a1c" not in st.session_state:
    st.session_state.a1c = None

if st.session_state.get("reset_flag", False):
    st.session_state.age = ""
    st.session_state.a1c = 0.0
    st.session_state.bmi = 0
    st.session_state.glucose = 0.0
    st.session_state.symptoms_main = []
    st.session_state.symptoms_second = []

    for key in [
        "main1_result",
        "main2_result",
        "first_result",
        "second_result",
        "a1c_category",
        "payload",
        "choices_str",
        "symptom_codes",
        "user_features",
        "bmi_category",
        "glucose_category",
        "jump_to_second",
        "symptom_list",
        "second_symptoms",
        "markov",
    ]:
        st.session_state.pop(key, None)

    st.session_state.stage = "main1"
    st.session_state.reset_flag = False
    st.rerun()


if "stage" not in st.session_state:
    st.session_state.stage = "main1"

if "markov_pending" not in st.session_state:
    st.session_state.markov_pending = False



st.title("Know Your Diabetes!")

st.caption(
    "Disclaimer: This app is intended for educational"
    " purposes and risk awareness. This app is not a substitute for professional "
    " medical evaluation and diagnosis.")

age_str = st.text_input("AGE", placeholder="Enter Here", key="age")
age_num = None
if age_str.strip().isdigit():
    age_num = int(age_str)
else:
    age_num = None

if "first_choices" not in st.session_state:
    st.session_state.first_choices = []

if "payload" not in st.session_state:
    st.session_state.payload = {}

symptoms_main = st.multiselect(
    "Click below to select all your matching symptom(s). When Done, Click Run Check",
    [
        "1 Frequent Urination",
        "2 Increased Thirst",
        "3 Increased Hunger",
        "4 Unexplained Weight Loss",
        "5 Other Symptoms",
        "6 No Symptoms"
    ],
    key="symptoms_main"
)

first_choices = []
for s in symptoms_main:
    s = str(s)
    parts = s.split()
    if parts and parts[0].isdigit():
        first_choices.append(int(parts[0]))

if "first_choices" not in st.session_state.payload:
    st.session_state.payload["first_choices"] = first_choices

symptom = []
for i in symptoms_main:
    num = int(i.split()[0])
    symptom.append(num)


def call_main1(payload):
    try:
        response = requests.post(f"https://know-your-diabetes-app-1.onrender.com/predict_main1", json=payload)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


def call_main2(payload):
    try:
        response = requests.post(f"https://know-your-diabetes-app-1.onrender.com/predict_main2", json=payload)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


def call_first(payload):
    try:
        response = requests.post(f"https://know-your-diabetes-app-1.onrender.com/predict_first", json=payload)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


def call_second(payload):
    try:
        response = requests.post(f"https://know-your-diabetes-app-1.onrender.com/predict_second", json=payload)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


def call_markov(payload):
    try:
        response = requests.post(f"https://know-your-diabetes-app-1.onrender.com/predict_markov", json=payload)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


terminal_main1_msgs = [
    "Low risk for diabetes.",
    "Model is 99% confident your pattern signals Type1 Diabetes. A Health Professional can give clearer guidance."]

terminal_main2_msgs = [ "With one symptom but a severely low A1C, your pattern shows low glucose levels. "
                               "A health professional can provide further assistance.",
                        "Model is 99% confident your pattern signals Type1 Diabetes. A Health Professional can give "
                        "clearer guidance.",
                        "Given your age and with only symptom and a normal A1C, your overall pattern points to a "
                        "low diabetes risk."]

terminal_first_msgs = [
    "Model is 99% confident your pattern signals diabetes. A health professional can provide further assistance.",
    "Data shows unhealthy weight. A health professional can provide further assistance.",
    "Data shows low glucose levels. A health professional can provide further assistance.",
    "Low risk for diabetes."]

terminal_second_msgs = [
    "Data shows low glucose levels. A health professional can provide further assistance.",
    "Model is 99% confident your pattern signals diabetes. "
    "A health professional can provide further assistance.",
    "Low risk for diabetes."]



if st.session_state.stage == "main1":
    if not age_num:
        st.warning("Please enter a valid age.")
        st.stop()

    if symptom == [6] and age_num:
        st.success("No symptoms reported. Low risk of diabetes.")
        st.button(
            "Start Over",
            on_click=lambda: (st.session_state.update(
                reset_flag=True,
                stage="main1")))
        st.stop()

    if 5 in symptom and 6 in symptom:
        st.error("Invalid. You cannot enter 'Other Symptoms' with 'No Symptoms'.")
        st.stop()
    elif 6 in symptom and len(symptom) > 1:
        st.error("You cannot select 'No Symptoms' with other valid symptoms.")
        st.stop()
    elif "user_features" not in st.session_state.payload:
        st.session_state.payload["user_features"] = {}

    col1, col2 = st.columns([ 3, 1])
    with col1:
        nums = first_choices
        sym_keys = [f"sym{x}" for x in nums]
        st.session_state.payload["first_choices"] = nums
        st.session_state.payload["symptoms"] = sym_keys
        st.session_state.payload["age"] = age_num
    with col2:
        run_check = st.button("Run Check")

    if run_check:
        if age_num is None or age_num <= 0:
            st.error("Please enter your age.")
            st.stop()

        if not symptom:
            st.error("Please select at least one option.")
            st.stop()

        main1_result, main_error = call_main1(st.session_state.payload)
        st.session_state.main1_result = main1_result
        msg = main1_result.get("message")
        requires = main1_result.get("requires")

        if requires == "decision_tree2":
            st.session_state.stage = "main2"
            st.rerun()

        if msg and msg in terminal_main1_msgs:
            st.session_state.stage = "done"
            st.rerun()


elif st.session_state.stage == "main2":
    st.write("More data is needed")
    col1, col2 = st.columns([3, 1])
    with col1:
        a1c = st.number_input(
            "Enter A1C (When Done, Click Run Check)",
            min_value=0.0,
            max_value=20.0,
            step=0.1,
            placeholder="Enter Here (Enter 0 if unknown)"
        )
    with col2:
        run_main2 = st.button("Run Check", key="runcheck_main2")

    if run_main2:
        st.session_state.payload["a1c"] = None if a1c == 0 else a1c
        st.session_state.a1c = st.session_state.payload["a1c"]

        main2_result, main_error = call_main2(st.session_state.payload)

        if main_error:
            st.error(main_error)
            st.rerun()

        st.session_state.main2_result = main2_result
        requires = main2_result.get("requires")
        if requires == "first_match":
            st.session_state.stage = "first"
            st.rerun()

        user_features = main2_result.get("user_features", {})
        msg = main2_result.get("message", "")
        st.success(msg)

        if msg and msg in terminal_main2_msgs:
            st.session_state.main2_result = main2_result
            st.session_state.stage = "done"
            st.rerun()



elif st.session_state.stage == "first":
    st.write("More data is needed")

    col1, col2 = st.columns([3, 1])
    with col1:
        bmi = st.number_input(
            "Enter BMI (When Done, Proceed To Glucose Numbers)",
            min_value=0.0,
            max_value=130.0,
            step=0.1,
            placeholder="Enter Here (Enter 0 if unknown)"
        )
        bmi_num = None if bmi == 0 else bmi

        glucose = st.number_input(
            "Enter Glucose Numbers (When Done, Click Run Check)",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            placeholder="Enter Here (Enter 0 if unknown)"
        )
        glucose_num = None if glucose == 0 else glucose


    with col2:
        run_first = st.button("Run Check", key="runcheck_first")

    st.session_state.payload["bmi"] = bmi_num
    st.session_state.payload["glucose"] = glucose_num
    st.session_state.payload["a1c"] = st.session_state.a1c


    if st.session_state.get("missing_all_three"):
        if bmi_num not in (None, 0) or glucose_num not in (None, 0) or st.session_state.a1c not in (None, 0):
            st.session_state.missing_all_three = False

    if run_first:
        first_result, first_error = call_first(st.session_state.payload)

        if first_error:
            st.error(first_error)
            st.rerun()

        st.session_state.first_result = first_result
        user_features = first_result.get("user_features", {})
        bfs_result = first_result.get("bfs_result")
        markov = first_result.get("markov")
        requires = first_result.get("requires")
        msg = first_result.get("message", "")
        bmi_cat = first_result.get("bmi_category")
        glucose_cat = first_result.get("glucose_category")

        if requires == "blocked":
            st.error("Please provide at least one data value:")
            st.stop()
        st.success(msg)

        if requires == "all_three":
            st.error("Please provide at least one data value:")
            st.stop()
        st.success(msg)

        if markov == "markov_transition":
            st.session_state.markov_pending = True

        if requires == "second_match":
            st.session_state.stage = "second"
            st.rerun()

        if msg and msg in terminal_first_msgs:
            st.session_state.stage = "done"
            st.rerun()

    if st.session_state.get("markov_pending") and st.session_state.stage == "first":
            pass


elif st.session_state.stage == "done":
    result = (
            st.session_state.get("second_result")
            or st.session_state.get("first_result")
            or st.session_state.get("main2_result")
            or st.session_state.get("main1_result")
            or {}
    )

    msg = result.get("message", "")
    if msg:
        st.success(msg)

    st.write("Click Below To Do Another Check")
    st.button(
        "Start Over",
        on_click=lambda: (st.session_state.update(reset_flag = True,
                          stage="main1")))


elif st.session_state.stage == "second":
    st.write("More data is needed")

    st.session_state.payload["a1c"] = st.session_state.a1c
    a1c = st.session_state.a1c

    col1, col2 = st.columns([3, 1])
    with col1:
        symptoms_second = st.multiselect(
            "Click below, select all matching possible symptom(s). When Done, Proceed To Complete My Assessment.",
            [
                "1 Fatigue",
                "2 Blurry Vision",
                "3 Frequent UTI",
                "4 Irritability or Moodiness",
                "5 Nausea",
                "6 Vomiting",
                "7 Stomach pains",
                "8 Unhealed sores or cuts",
                "9 Dark patches of skin",
                "10 Numbness in feet and hands",
                "11 None of these symptoms"
            ]
        )
    with col2:
        complete_btn = st.button("Complete My Assessment")

    secondary = [int(i.split()[0]) for i in symptoms_second]

    if not secondary:
        st.error("Please select at least one option from the choices above.")
    else:
        st.session_state.payload["second_choices"] = secondary

        if 11 in secondary and len(secondary) > 1:
            st.error("You cannot select 'None of these symptoms' with other symptoms.")
            st.session_state.stage = "second"
            st.rerun()

        elif 11 in secondary and len(secondary) == 1:
            if complete_btn:
                st.session_state.payload["a1c"] = st.session_state.a1c
                st.session_state.payload["bmi"] = st.session_state.payload.get("bmi")

                second_result, second_error = call_second(st.session_state.payload)

                if second_error:
                    st.error(second_error)
                    st.rerun()

                st.session_state.second_result = second_result
                msg = second_result.get("message", "")
                prediction = second_result.get("prediction")
                markov = second_result.get("markov", False)
                st.success(msg)

                if markov:
                    st.session_state.stage = "markov_stage"
                    st.rerun()
                st.session_state.stage = "done"
                st.rerun()

        else:
            if complete_btn:
                st.session_state.payload = {
                    "user_features": st.session_state.first_result.get("user_features", {}),
                    "first_choices": st.session_state.payload["first_choices"],
                    "second_choices": secondary,
                    "glucose": st.session_state.payload["glucose"],
                    "bmi": st.session_state.payload["bmi"],
                    "a1c": st.session_state.a1c,
                }

                second_result, second_error = call_second(st.session_state.payload)

                if second_error:
                    st.error(second_error)
                    st.rerun()

                if second_result.get("message") == "Invalid symptom format.":
                    st.error("Invalid symptom format. Please check your selections and try again.")
                else:
                    st.session_state.second_result = second_result

                    markov = second_result.get("markov")
                    if markov:
                        st.session_state.markov_pending = True
                        st.session_state.stage = "markov_stage"
                        st.rerun()

                    user_features = second_result.get("user_features", {})
                    prediction = second_result.get("prediction")
                    msg = second_result.get("message", "")


                    if msg and not st.session_state.get("markov_pending"):
                        st.success(msg)
                        st.session_state.stage = "done"
                        st.rerun()

                    if markov and "prediabetes" in msg.lower():
                        st.session_state.markov_pending = True
                        st.session_state.stage = "markov_stage"
                        st.rerun()

                    if prediction in (0, 1):
                        st.session_state.stage = "done"
                        st.rerun()

                    if msg and msg in terminal_second_msgs:
                        st.session_state.stage = "done"
                        st.rerun()


elif st.session_state.stage == "markov_stage":
    second_result = st.session_state.get("second_result")
    if not second_result:
        st.warning("No previous result found. Please restart the assessment.")
        if st.button("Start Over"):
            st.session_state.reset_flag = True
            st.session_state.stage = "main1"
            st.rerun()
        st.stop()

    if st.session_state.payload.get("second_choices") is None:
        st.warning("Missing symptom data. Please restart the assessment.")
        if st.button("Start Over"):
            st.session_state.reset_flag = True
            st.session_state.stage = "main1"
            st.rerun()
        st.stop()

    st.session_state.payload["user_features"] = {
        "symptoms": st.session_state.payload.get("second_choices", []),
        "glucose": st.session_state.payload.get("glucose"),
        "a1c": st.session_state.payload.get("a1c"),
        "bmi": st.session_state.payload.get("bmi"),
    }

    st.write("Prediction:")
    st.success(second_result.get("message"))


    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("See The Model's Predictions on How Your Risks May Change Over Time")

    with col2:
        markov_run = st.button("Model Projections", key="markov_run")

    if markov_run:
        markov_result, markov_error = call_markov(st.session_state.payload)
        markov_result = markov_result.get("markov", {})

        if markov_error:
            st.error(markov_error)
            st.stop()
        else:
            required_keys = ["to_diabetes", "stay_prediabetes", "to_no_diabetes"]
            if not all(k in markov_result and isinstance(markov_result[k], (int, float)) for k in required_keys):
                st.error("Possibilities are not available at this time.")
                st.stop()

            st.success(f"The model predicts that you are in the prediabetes zone.  \n"
                    f"If you keep things the same, your numbers lean about  \n"
                    f"{markov_result['to_diabetes']:.1%} toward diabetes,  \n"
                    f"about {markov_result['stay_prediabetes']:.1%} toward staying in prediabetes,  \n"
                    f"and about a {markov_result['to_no_diabetes']:.1%} natural push in the right "
                    f"direction once you start making changes.")


    if st.button("Start Over"):
        st.session_state.reset_flag = True
        st.rerun()
