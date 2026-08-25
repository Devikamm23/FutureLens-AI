from flask import Flask, render_template, request, jsonify
import joblib
import math
import os
from datetime import datetime, timezone
import pandas as pd

app = Flask(__name__)

# Path to the trained model
MODEL_FILE = os.path.join(
    "models",
    "facility_models.pkl"
)

# Check whether model exists
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "facility_models.pkl not found. "
        "Run 'python train_model.py' first."
    )

# Load trained models
bundle = joblib.load(MODEL_FILE)

models = bundle["models"]

LIVE_FACILITIES = [
    "benches",
    "washrooms",
    "classrooms",
    "labs",
    "library_seats"
]

SCENARIO_DEFAULTS = {
    "departments": 8,
    "faculty": 250,
    "courses": 120,
    "average_class_size": 40,
    "attendance_rate": 80,
    "library_usage_percent": 50,
    "lab_usage_percent": 50,
    "events_per_month": 2,
    "staff": 100,
    "working_hours": 8,
    "online_hybrid_percent": 20,
    "hostel_residents": 500,
    "campus_expansion_percent": 0,
    "building_closures": 0,
    "seasonal_demand_percent": 0,
    "accessibility_policy_percent": 0,
    "new_labs": 0,
    "new_library_seats": 0,
    "demand_period": "normal"
}

FORECAST_HORIZONS = [
    ("current", "Current"),
    ("year_1", "1 year"),
    ("year_3", "3 years"),
    ("year_5", "5 years"),
    ("year_10", "10 years")
]


# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------

def predict_facilities(
    students,
    vehicles,
    temperature,
    working_hours
):

    input_data = pd.DataFrame([{
        "students": students,
        "vehicles": vehicles,
        "temperature": temperature,
        "working_hours": working_hours
    }])

    predictions = {}

    for name, model in models.items():

        prediction = model.predict(input_data)[0]

        predictions[name] = math.ceil(
            max(0, prediction)
        )

    return predictions


def apply_scenario_factors(predicted, scenario):

    baseline = {
        "departments": 8,
        "faculty": 250,
        "courses": 120,
        "average_class_size": 40,
        "attendance_rate": 80,
        "library_usage_percent": 50,
        "lab_usage_percent": 50,
        "events_per_month": 2,
        "staff": 100,
        "working_hours": 8,
        "online_hybrid_percent": 20,
        "hostel_residents": 500,
        "campus_expansion_percent": 0,
        "building_closures": 0,
        "seasonal_demand_percent": 0,
        "accessibility_policy_percent": 0,
        "new_labs": 0,
        "new_library_seats": 0
    }

    enrollment_factor = scenario["students"] / 5000
    online_factor = max(0.5, 1 - scenario["online_hybrid_percent"] / 200)
    attendance_factor = max(0.5, min(1.25, scenario["attendance_rate"] / 80))
    period_factor = {
        "normal": 1,
        "exams": 1.15,
        "admissions": 1.1
    }[scenario["demand_period"]]
    access_factor = 1 + scenario["accessibility_policy_percent"] / 100
    seasonal_factor = 1 + scenario["seasonal_demand_percent"] / 100
    closure_factor = max(0.5, 1 + scenario["building_closures"] / 100)
    expansion_factor = max(0.5, 1 + scenario["campus_expansion_percent"] / 100)
    teaching_factor = (
        0.55 * enrollment_factor
        + 0.2 * scenario["departments"] / baseline["departments"]
        + 0.15 * scenario["courses"] / baseline["courses"]
        + 0.1 * scenario["faculty"] / baseline["faculty"]
        + 0.05 * scenario["staff"] / baseline["staff"]
    )
    class_size_factor = max(0.6, min(1.5, 40 / scenario["average_class_size"]))
    event_factor = 1 + min(0.2, scenario["events_per_month"] * 0.02)
    common_factor = period_factor * seasonal_factor * closure_factor * expansion_factor

    factors = {
        "benches": teaching_factor * class_size_factor * attendance_factor * online_factor * common_factor,
        "classrooms": teaching_factor * class_size_factor * online_factor * common_factor,
        "labs": (
            0.7 * teaching_factor
            + 0.3 * scenario["lab_usage_percent"] / 50
        ) * common_factor,
        "library_seats": (
            0.65 * enrollment_factor
            + 0.35 * scenario["library_usage_percent"] / 50
        ) * common_factor * access_factor,
        "washrooms": (
            0.7 * teaching_factor
            + 0.3 * scenario["hostel_residents"] / 500
        ) * attendance_factor * event_factor * common_factor * access_factor
    }

    adjusted = predicted.copy()

    for name, factor in factors.items():
        adjusted[name] = math.ceil(max(0, predicted[name] * factor))

    adjusted["labs"] += math.ceil(scenario["new_labs"])
    adjusted["library_seats"] += math.ceil(scenario["new_library_seats"])

    return adjusted


def read_scenario(data):

    scenario = {
        name: float(data.get(name, default))
        for name, default in SCENARIO_DEFAULTS.items()
        if name != "demand_period"
    }
    scenario["students"] = float(data["students"])
    scenario["demand_period"] = data.get("demand_period", "normal")

    numeric_values = [
        scenario[name]
        for name in scenario
        if name != "demand_period"
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("All scenario values must be finite numbers.")
    if scenario["students"] <= 0:
        raise ValueError("Students must be greater than zero.")
    if any(value < 0 for value in numeric_values):
        raise ValueError("Scenario values cannot be negative.")
    if not 0 <= scenario["attendance_rate"] <= 100:
        raise ValueError("Attendance rate must be between 0 and 100.")
    if not 0 <= scenario["library_usage_percent"] <= 100:
        raise ValueError("Library usage must be between 0 and 100.")
    if not 0 <= scenario["lab_usage_percent"] <= 100:
        raise ValueError("Lab usage must be between 0 and 100.")
    if scenario["average_class_size"] == 0:
        raise ValueError("Average class size must be greater than zero.")
    if scenario["demand_period"] not in {"normal", "exams", "admissions"}:
        raise ValueError("Demand period must be normal, exams, or admissions.")

    return scenario


def resource_requirements(students, scenario):

    scenario = dict(scenario)
    scenario["students"] = students
    vehicles = max(0, 2000 * students / 5000)
    predicted = predict_facilities(
        students,
        vehicles,
        28,
        scenario["working_hours"]
    )
    predicted = apply_scenario_factors(predicted, scenario)

    return {
        "benches": predicted["benches"],
        "classrooms": predicted["classrooms"],
        "faculty": math.ceil(students / 20),
        "labs": predicted["labs"],
        "library_seats": predicted["library_seats"],
        "washrooms": predicted["washrooms"],
        "parking": predicted["parking"],
        "computers": math.ceil(students * 0.25)
    }


def forecast_campus(data):

    scenario = read_scenario(data)
    current_students = scenario["students"]
    projections = {
        "current": current_students,
        "year_1": float(data["students_1"]),
        "year_3": float(data["students_3"]),
        "year_5": float(data["students_5"]),
        "year_10": float(data["students_10"])
    }

    if any(
        not math.isfinite(value) or value <= 0
        for value in projections.values()
    ):
        raise ValueError("All student projections must be greater than zero.")

    forecast = []
    for key, label in FORECAST_HORIZONS:
        forecast.append({
            "key": key,
            "label": label,
            "students": math.ceil(projections[key]),
            "requirements": resource_requirements(projections[key], scenario)
        })

    growth_percent = round(
        ((projections["year_3"] - current_students) / current_students) * 100,
        1
    )
    three_year = forecast[2]["requirements"]
    current = forecast[0]["requirements"]
    priorities = []
    for name, label in [
        ("classrooms", "classrooms"),
        ("faculty", "faculty members"),
        ("parking", "parking spaces"),
        ("labs", "labs"),
        ("benches", "benches"),
        ("computers", "computer systems")
    ]:
        increase = max(0, three_year[name] - current[name])
        if increase:
            priorities.append(f"add {increase:,} {label}")

    return {
        "forecast": forecast,
        "recommendation": {
            "growth_percent_3_year": growth_percent,
            "priority": (
                "Priority: " + ", ".join(priorities[:4]) + "."
                if priorities
                else "Priority: Current capacity is sufficient for the 3-year projection."
            ),
            "planning_horizon": "3 years"
        }
    }


def analyze_live_facilities(actual, required):

    analysis = {}

    for name in LIVE_FACILITIES:
        actual_value = max(0, math.ceil(float(actual[name])))
        required_value = required[name]
        gap = actual_value - required_value
        coverage = round((actual_value / required_value) * 100, 1) if required_value else 100

        analysis[name] = {
            "actual": actual_value,
            "required": required_value,
            "gap": gap,
            "shortage": max(0, -gap),
            "surplus": max(0, gap),
            "coverage_percent": coverage,
            "status": (
                "shortage" if gap < 0
                else "near_capacity" if coverage < 110
                else "available"
            )
        }

    return analysis


def generate_live_recommendations(analysis):

    recommendations = []

    labels = {
        "benches": "benches",
        "washrooms": "washrooms",
        "classrooms": "classrooms",
        "labs": "labs",
        "library_seats": "library seats"
    }

    for name, item in analysis.items():

        if item["gap"] < 0:

            recommendations.append(
                f"Add {item['shortage']:,} {labels[name]} to meet the estimated requirement."
            )

        elif item["status"] == "near_capacity":

            recommendations.append(
                f"{labels[name].capitalize()} are close to the estimated capacity. Monitor usage closely."
            )

    if not recommendations:

        recommendations.append(
            "Live facility levels meet or exceed the estimated requirement."
        )

    return recommendations


# ---------------------------------------------------
# AI RECOMMENDATIONS
# ---------------------------------------------------

def generate_recommendations(
    students,
    vehicles,
    temperature,
    working_hours,
    predicted
):

    recommendations = []

    if students >= 7000:

        recommendations.append(
            "Student population is high. "
            "Consider adding classrooms or optimizing the timetable."
        )

    if predicted["benches"] >= 4000:

        recommendations.append(
            "Additional classroom seating may be required."
        )

    if predicted["washrooms"] >= 60:

        recommendations.append(
            "Consider expanding washroom facilities across campus."
        )

    if predicted["classrooms"] >= 40:

        recommendations.append(
            "Additional classrooms or better classroom scheduling may be needed."
        )

    if predicted["labs"] >= 20:

        recommendations.append(
            "Laboratory capacity may need expansion."
        )

    if predicted["library_seats"] >= 1000:

        recommendations.append(
            "Consider increasing library seating and study areas."
        )

    if predicted["parking"] >= 1500:

        recommendations.append(
            "Parking demand is high. "
            "Consider smart parking or additional parking space."
        )

    if predicted["canteen_seats"] >= 1000:

        recommendations.append(
            "Consider increasing canteen seating and service capacity."
        )

    if temperature >= 35:

        recommendations.append(
            "High temperature detected. "
            "Consider shaded areas and additional cooling facilities."
        )

    if working_hours >= 11:

        recommendations.append(
            "Long working hours detected. "
            "Optimize facility usage and scheduling."
        )

    if vehicles > students * 0.5:

        recommendations.append(
            "Vehicle demand is relatively high. "
            "Improve campus transport and parking management."
        )

    if not recommendations:

        recommendations.append(
            "Current infrastructure appears manageable "
            "for the selected scenario."
        )

    return recommendations


# ---------------------------------------------------
# SIMULATE FUTURE SCENARIO
# ---------------------------------------------------

@app.route(
    "/forecast",
    methods=["POST"]
)
def forecast():

    try:
        result = forecast_campus(request.get_json(silent=True) or {})
        result["success"] = True
        result["updated_at"] = datetime.now(timezone.utc).isoformat()
        return jsonify(result)

    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

@app.route(
    "/simulate",
    methods=["POST"]
)
def simulate():

    try:

        data = request.get_json(silent=True) or {}

        scenario = read_scenario(data)
        students = scenario["students"]

        vehicles = float(data.get("vehicles", 2000))

        temperature = float(data.get("temperature", 28))

        working_hours = float(data.get("working_hours", 8))

        predicted = predict_facilities(
            students,
            vehicles,
            temperature,
            working_hours
        )
        predicted = apply_scenario_factors(predicted, scenario)

        actual = {name: float(data[name]) for name in LIVE_FACILITIES}

        values = [students, vehicles, temperature, working_hours, *actual.values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("All values must be finite numbers.")
        if students <= 0:
            raise ValueError("Students must be greater than zero.")
        if any(value < 0 for value in actual.values()):
            raise ValueError("Facility counts cannot be negative.")

        live_analysis = analyze_live_facilities(actual, predicted)
        recommendations = generate_live_recommendations(live_analysis)

        return jsonify({

            "success": True,

            "predicted": predicted,

            "actual": actual,

            "analysis": live_analysis,

            "updated_at": datetime.now(timezone.utc).isoformat(),

            "recommendations":
                recommendations

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 400


# ---------------------------------------------------
# CURRENT VS FUTURE COMPARISON
# ---------------------------------------------------

@app.route(
    "/compare",
    methods=["POST"]
)
def compare():

    try:

        data = request.get_json()

        current = data["current"]
        future = data["future"]

        current_scenario = read_scenario(current)
        future_scenario = read_scenario(future)

        current_prediction = predict_facilities(
            current_scenario["students"],
            float(current.get("vehicles", 2000)),
            float(current.get("temperature", 28)),
            float(current.get("working_hours", 8))
        )
        current_prediction = apply_scenario_factors(
            current_prediction,
            current_scenario
        )

        future_prediction = predict_facilities(
            future_scenario["students"],
            float(future.get("vehicles", 2000)),
            float(future.get("temperature", 28)),
            float(future.get("working_hours", 8))
        )
        future_prediction = apply_scenario_factors(
            future_prediction,
            future_scenario
        )

        changes = {}

        for key in current_prediction:

            current_value = current_prediction[key]

            future_value = future_prediction[key]

            if current_value == 0:

                percentage_change = 0

            else:

                percentage_change = (
                    (
                        future_value -
                        current_value
                    )
                    /
                    current_value
                ) * 100

            changes[key] = {

                "current":
                    current_value,

                "future":
                    future_value,

                "change_percent":
                    round(
                        percentage_change,
                        2
                    )

            }

        recommendations = generate_recommendations(
            float(future["students"]),
            float(future.get("vehicles", 2000)),
            float(future.get("temperature", 28)),
            float(future.get("working_hours", 8)),
            future_prediction
        )

        return jsonify({

            "success": True,

            "current":
                current_prediction,

            "future":
                future_prediction,

            "changes":
                changes,

            "recommendations":
                recommendations

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 400


# ---------------------------------------------------
# START FLASK SERVER
# ---------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )