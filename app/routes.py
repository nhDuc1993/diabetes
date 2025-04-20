from flask import Blueprint, jsonify, render_template, request
import pandas as pd
import joblib

main = Blueprint('main', __name__)
DATA_PATH = 'app/data/data.csv'
model = joblib.load('app/models/random_forest_model.pkl')

@main.route("/")
def index():
    return render_template("index.html")

@main.route('/heatmap')
def heatmap():
    return render_template('heatmap.html')

@main.route('/predict')
def predict_page():
    return render_template("prediction.html")

@main.route('/factors')
def factors_page():
    return render_template("factors.html")

@main.route('/api/kpis')
def get_kpis():
    df = pd.read_csv(DATA_PATH)

    filters = {
        'Sex': request.args.get('sex'),
        'Age': request.args.get('age'),
        'Education': request.args.get('education'),
        'Income': request.args.get('income')
    }

    for col, val in filters.items():
        if val and val.upper() != 'ALL':
            df = df[df[col] == int(val)]

    count = len(df)
    percent = lambda col: round(df[col].mean() * 100, 2) if count else 0.0

    return jsonify({
        "count": count,
        "HighBP": percent('HighBP'),
        "HighChol": percent('HighChol'),
        "CholCheck": percent('CholCheck'),
        "Smoker": percent('Smoker'),
        "Stroke": percent('Stroke'),
        "HeartDisease": percent('HeartDiseaseorAttack'), 
        "PhysActivity": percent('PhysActivity'), 
        "Fruits": percent('Fruits'), 
        "Veggies": percent('Veggies'),
        "HvyAlcohol": percent('HvyAlcoholConsump'), 
        "AnyHealthcare": percent('AnyHealthcare'),
        "NoDocbcCost": percent('NoDocbcCost'),
        "DiffWalk": percent('DiffWalk'),
        "Resilient": percent('Diabetes_binary')
    })

@main.route('/api/bmi-distribution')
def get_bmi_distribution():
    df = pd.read_csv(DATA_PATH)

    # Apply filters if needed
    filters = {
        'Sex': request.args.get('sex'),
        'Age': request.args.get('age'),
        'Education': request.args.get('education'),
        'Income': request.args.get('income')
    }

    for col, val in filters.items():
        if val and val.upper() != 'ALL':
            df = df[df[col] == int(val)]

    bins = list(range(0, 100, 10))
    labels = [f"{b}-{b+10}" for b in bins]

    return jsonify({
        "labels": labels,
        "bmi_resilient": df[df['Diabetes_binary'] == 1]['BMI'].tolist(),
        "bmi_nonresilient": df[df['Diabetes_binary'] == 0]['BMI'].tolist()
    })


@main.route('/api/scatter')
def get_scatter():
    df = pd.read_csv(DATA_PATH)

    filters = {
        'Sex': request.args.get('sex'),
        'Age': request.args.get('age'),
        'Education': request.args.get('education'),
        'Income': request.args.get('income')
    }

    for col, val in filters.items():
        if val and val.upper() != 'ALL':
            df = df[df[col] == int(val)]

    resilient_df = df[df['Diabetes_binary'] == 1]
    non_resilient_df = df[df['Diabetes_binary'] == 0]
    features = ["Age", "BMI", "GenHlth", "MentHlth", "PhysHlth"]

    def get_scatter_data(group_df):
        return group_df[features].dropna().to_dict(orient="records")

    return jsonify({
        "resilient": get_scatter_data(resilient_df),
        "non_resilient": get_scatter_data(non_resilient_df)
    })

@main.route("/api/correlation")
def get_correlation():
    df = pd.read_csv(DATA_PATH)

    corr = df.corr().round(2)

    data = []
    for i in corr.columns:
        for j in corr.columns:
            data.append({
                'x': i,
                'y': j,
                'value': corr.loc[i, j]
            })

    return jsonify(data)

@main.route('/prediction', methods=['POST'])
def predict():
    data = request.json

    input_order = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
        'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
        'Sex', 'Age', 'Education', 'Income'
    ]

    input_data = [data[feature] for feature in input_order]
    input_df = pd.DataFrame([input_data], columns=input_order)

    proba = model.predict_proba(input_df)[0]
    prediction = "Diabetes" if int(model.predict(input_df)[0])  else "No Diabetes"

    return jsonify({
        'label': prediction,
        'probability': round(float(max(proba)) * 100, 4) 
    })


@main.route('/factors/bar')
def binary_charts():
    df = pd.read_csv(DATA_PATH)
    rename_dict = {
        "HighBP": "High Blood Pressure",
        "HighChol": "High Cholesterol",
        "CholCheck": "Cholesterol Check",
        "Smoker": "Smoker",
        "Stroke": "Stroke",
        "HeartDiseaseorAttack": "Heart Disease",
        "PhysActivity": "Physical Activity",
        "Fruits": "Fruits",
        "Veggies": "Veggies",
        "HvyAlcoholConsump": "Heavy Alcohol Consumption",
        "AnyHealthcare": "Healthcare",
        "NoDocbcCost": "No Doc Appointment Because of Cost",
        "DiffWalk": "Walking Difficulty"
    }

    df.rename(columns=rename_dict, inplace=True)

    binary_features = ['High Blood Pressure', 'High Cholesterol', 'Cholesterol Check', 'Smoker', 'Stroke', 'Heart Disease',
                       'Physical Activity', 'Fruits', 'Veggies', 'Heavy Alcohol Consumption', 'Healthcare',
                       'No Doc Appointment Because of Cost', 'Walking Difficulty'] 

    result = {}

    for feature in binary_features:
        if feature not in df.columns:
            continue
        grouped = df.groupby(feature)['Diabetes_binary'].mean() * 100
        result[feature] = {
            "Yes": round(grouped.get(1, 0), 2),
            "No": round(grouped.get(0, 0), 2)
        }

    return jsonify(result)