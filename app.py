from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import openai
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
import os
load_dotenv()

app = Flask(__name__)

# 直接设置API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")
# 加载数据集
df = pd.read_csv("employees_dataset.csv")
# 数据预处理
features = df.drop(columns=['salary'])
target = df['salary']

ordinal_cols = ['education_level']
ordinal_order = [['High School', "Bachelor's", "Master's", 'PhD']]
encoder = OrdinalEncoder(categories=ordinal_order)
features[ordinal_cols] = encoder.fit_transform(features[ordinal_cols])
categorical_cols = features.select_dtypes(include=['object']).columns
features_encoded = pd.get_dummies(features, columns=categorical_cols)

train_features = features_encoded.iloc[:100]
train_target = target.iloc[:100]
predict_features = features_encoded.iloc[100:]
predict_target = target.iloc[100:]

# 训练线性回归模型
model = LinearRegression()
model.fit(train_features, train_target)


def simulate_interactive_negotiation_with_api(features_row, y_true, risk1='seeking', risk2='seeking',
                                              threshold=100, max_rounds=10):
    base_pred = model.predict([features_row])[0]
    agent1_pred = base_pred
    agent2_pred = base_pred + np.random.normal(0, 500)

    step_map = {'averse': 0.1, 'neutral': 0.25,'seeking': 0.4}
    step1 = step_map[risk1]
    step2 = step_map[risk2]

    rounds = 0
    conversation = []

    while abs(agent1_pred - agent2_pred) > threshold and rounds < max_rounds:
        rounds += 1

        # Agent A speaks
# Agent A speaks
        prompt1 = f"""
        You are Agent A with a {risk1} risk preference. You predicted a salary of {agent1_pred:.2f}.
        Agent B predicted {agent2_pred:.2f}.

        Please explain your reasoning in 1–2 sentences (e.g., based on education, experience, or negotiation strategy),
        and then give your updated prediction (slightly adjusted toward B's value).

        Respond in this format:
        "Your reasoning..."
        Prediction: <new_prediction_number>
        """.strip()

        response1 = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant negotiating a salary prediction."},
                {"role": "user", "content": prompt1}
            ],
            temperature=0.7,
            max_tokens=200
        )

        content1 = response1.choices[0].message.content.strip()
        prediction_index = content1.find("Prediction:")
        if prediction_index != -1:
            try:
                prediction_str = content1[prediction_index + len("Prediction:"):].strip()
                new_a1 = float(prediction_str)
            except ValueError:
                new_a1 = agent1_pred + step1 * (agent2_pred - agent1_pred)
        else:
            new_a1 = agent1_pred + step1 * (agent2_pred - agent1_pred)

        conversation.append({
            "round": rounds,
            "agent": "Agent A",
            "message": content1,
            "prediction": new_a1
        })

        # Agent B speaks
        prompt2 = f"""
        You are Agent B with a {risk2} risk preference. You predicted a salary of {agent2_pred:.2f}.
        Agent A just predicted {new_a1:.2f}.

        Please explain your reasoning in 1–2 sentences (e.g., based on negotiation dynamics or confidence),
        and then give your updated prediction (slightly adjusted toward A's value).

        Respond in this format:
        "Your reasoning..."
        Prediction: <new_prediction_number>
        """.strip()

        response2 = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant negotiating a salary prediction."},
                {"role": "user", "content": prompt2}
            ],
            temperature=0.7,
            max_tokens=200
        )

        content2 = response2.choices[0].message.content.strip()
        prediction_index = content2.find("Prediction:")
        if prediction_index != -1:
            try:
                prediction_str = content2[prediction_index + len("Prediction:"):].strip()
                new_a2 = float(prediction_str)
            except ValueError:
                new_a2 = agent2_pred + step2 * (new_a1 - agent2_pred)
        else:
            new_a2 = agent2_pred + step2 * (new_a1 - agent2_pred)

        conversation.append({
            "round": rounds,
            "agent": "Agent B",
            "message": content2,
            "prediction": new_a2
        })

        agent1_pred = new_a1
        agent2_pred = new_a2

    final_pred = (agent1_pred + agent2_pred) / 2
    rmse = np.sqrt((y_true - final_pred) ** 2)
    return conversation, final_pred, rmse, rounds


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    row_num = int(request.form.get('row_num'))
    features_row = predict_features.iloc[row_num - 100]
    y_true = predict_target.iloc[row_num - 100]

    conversation, final_pred, rmse, rounds = simulate_interactive_negotiation_with_api(
        features_row,
        y_true,
        risk1='seeking',
        risk2='seeking'
    )

    return render_template('result.html', conversation=conversation, final_pred=final_pred, rmse=rmse, rounds=rounds,
                           row_num=row_num)


if __name__ == '__main__':
    app.run(debug=True)