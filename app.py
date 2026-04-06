from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

FORM = """
<!DOCTYPE html>
<html>
<head><title>Prédiction de défaut bancaire</title></head>
<body>
  <h2>Prédiction de défaut de prêt</h2>
  <form method="POST" action="/predict">
    <label>Lignes de crédit :</label><br>
    <input type="number" name="credit_lines_outstanding" step="1" required><br><br>
    <label>Montant du prêt :</label><br>
    <input type="number" name="loan_amt_outstanding" step="any" required><br><br>
    <label>Dette totale :</label><br>
    <input type="number" name="total_debt_outstanding" step="any" required><br><br>
    <label>Revenu :</label><br>
    <input type="number" name="income" step="any" required><br><br>
    <label>Années d'emploi :</label><br>
    <input type="number" name="years_employed" step="1" required><br><br>
    <label>Score FICO :</label><br>
    <input type="number" name="fico_score" step="1" required><br><br>
    <button type="submit">Prédire</button>
  </form>
  {% if result is not none %}
    <h3>Résultat : {{ result }}</h3>
  {% endif %}
</body>
</html>
"""

def build_features(form):
    credit_lines = float(form["credit_lines_outstanding"])
    loan_amt = float(form["loan_amt_outstanding"])
    total_debt = float(form["total_debt_outstanding"])
    income = float(form["income"])
    years_employed = float(form["years_employed"])
    fico_score = float(form["fico_score"])
    debt_to_income = total_debt / (income + 1)
    loan_to_income = loan_amt / (income + 1)
    return np.array([[credit_lines, loan_amt, total_debt, income,
                      years_employed, fico_score, debt_to_income, loan_to_income]])

@app.route("/", methods=["GET"])
def home():
    return render_template_string(FORM, result=None)

@app.route("/predict", methods=["POST"])
def predict():
    features = build_features(request.form)
    features_scaled = scaler.transform(features)
    proba = model.predict_proba(features_scaled)[0][1]
    if proba >= 0.5:
        result = f"⚠️ Risque de défaut élevé ({proba*100:.1f}%)"
    else:
        result = f"✅ Faible risque de défaut ({proba*100:.1f}%)"
    return render_template_string(FORM, result=result)

if __name__ == "__main__":
    app.run(debug=True)