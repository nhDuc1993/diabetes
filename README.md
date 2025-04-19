# Flask + D3 Diabetes Prediction App

## Introduction
This web application is built using Flask for the backend and D3.js for dynamic visualizations. It allows users to input various health-related parameters and receive predictions about their resilience using a Random Forest model.

## Folder Structure
The folder structure of the project is as follows:

your_project/ │ ├── app/ │ ├── static/ │ │ └── lib/ │ │ └── d3/ │ │ └── d3.min.js <-- D3.js library │ ├── templates/ │ │ └── index.html <-- Main HTML page for the app │ ├── data/ │ │ └── diabetes_data.csv <-- The dataset used for visualizations │ ├── models/ │ │ └── random_forest_model.pkl <-- Pre-trained random forest model (pickled) │ ├── routes.py <-- Flask route definitions │ └── init.py <-- Initialize Flask app │ ├── run.py <-- Entry point to run the Flask app └── requirements.txt <-- Dependencies for the project


## How to Use
### 1. Clone the Repository

git clone https://github.com/nhduc1993/diabetes.git
cd diabetes

### 2. Clone the Repository

```bash
python -m venv venv
venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Flask App

```bash
python run.py
```
