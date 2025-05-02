from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import openai
import requests
import os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GPT_API_KEY = os.getenv("GPT_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Load dataset and train model
df = pd.read_csv("employees_dataset.csv")
features = df.drop(columns=['salary'])
target = df['salary']
features_encoded = pd.get_dummies(features)
train_features = features_encoded.iloc[:100]
train_target = target.iloc[:100]
model = LinearRegression()
model.fit(train_features, train_target)

model_coefficients = model.coef_
model_intercept = model.intercept_
feature_names = train_features.columns.tolist()

# ============ LLM API functions ============
def call_gemini(api_key, prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Gemini API error: {response.status_code}"
    except Exception as e:
        return f"Gemini call error: {e}"

def call_gpt(api_key, prompt):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a salary prediction expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT API call error: {e}"

# ============ Predict Route ============
@app.route('/predict', methods=['POST'])
def predict():
    try:
        row_index = int(request.form.get("row_num"))
        if row_index < 100 or row_index >= len(df):
            return f"Row number must be >= 100 and < {len(df)}", 400

        selected_features = features.iloc[[row_index]]
        selected_features_encoded = pd.get_dummies(selected_features)
        selected_features_encoded = selected_features_encoded.reindex(columns=feature_names, fill_value=0)

        # Agent A prompt
        prompt_agent_a = f"""
You are Agent A, a salary prediction expert. Based on the following trained linear regression model, please calculate the employee's salary step-by-step:

- Employee features and values:
{dict(selected_features_encoded.iloc[0])}

- Linear regression model parameters (feature: coefficient):
{dict(zip(feature_names, model_coefficients))}

- Intercept term:
{model_intercept}

Please show:
1. Step-by-step multiplication of features and coefficients.
2. Sum of contributions + intercept.
3. Final predicted salary.
"""
        agent_a_response = call_gemini(GEMINI_API_KEY, prompt_agent_a)

        # Agent B prompt
        prompt_agent_b = f"""
You are Agent B, a salary prediction expert. Here is Agent A's salary prediction:

{agent_a_response}

Your task:
1. Review Agent A's prediction carefully.
2. If you agree with Agent A's result, explain why and confirm.
3. If you find any mistakes or inconsistencies, point them out and recalculate the correct salary.
4. Show your own final salary prediction.
Be detailed and logical in your review.
"""
        agent_b_response = call_gpt(GPT_API_KEY, prompt_agent_b)

        return render_template("result.html",
                               row_num=row_index,
                               conversation=[
                                   {"round": 1, "agent": "Agent A", "message": agent_a_response},
                                   {"round": 2, "agent": "Agent B", "message": agent_b_response}
                               ])
    except Exception as e:
        return f"Internal Server Error: {e}", 500
