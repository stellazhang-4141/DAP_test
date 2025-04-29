import pandas as pd
import requests
import openai
from flask import Flask, render_template, request
from dotenv import load_dotenv

app = Flask(__name__)

# 1. Load dataset
try:
    df = pd.read_csv("employees_dataset.csv")
except FileNotFoundError:
    print("❌ 未找到数据集文件，请确保 'employees_dataset.csv' 文件存在。")
    df = None

# 2. Prepare features and target
if df is not None:
    features = df.drop(columns=['salary'])
    target = df['salary']

    # 3. Encode categorical variables (One-Hot Encoding)
    features_encoded = pd.get_dummies(features)

    # 4. Split dataset
    train_features = features_encoded.iloc[:100]
    train_target = target.iloc[:100]

    predict_features = features_encoded.iloc[100:]
    predict_target = target.iloc[100:]

    # 5. Train Linear Regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(train_features, train_target)

    print("✅ Linear Regression model trained successfully!")

    model_coefficients = model.coef_
    model_intercept = model.intercept_
    feature_names = train_features.columns.tolist()

# 6. Setup API keys
import os
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GPT_API_KEY = os.getenv("GPT_API_KEY")

# 7. Define API call functions
def call_gemini(api_key, prompt):
    if not api_key or "AIza" not in api_key:
        print("❌ Gemini API Key seems invalid. Please check your key.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }
    print(f"调用 Gemini API，请求 URL: {url}，请求体: {payload}")  # 添加这行打印请求信息
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Gemini API 响应状态码: {response.status_code}，响应内容: {response.text}")  # 添加这行打印响应信息
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"❌ Gemini API call failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Gemini call error: {e}")
        return None

def call_gpt(api_key, prompt):
    if not api_key or not api_key.startswith("sk-"):
        print("❌ GPT API Key seems invalid. Please check your key.")
        return None

    client = openai.OpenAI(api_key=api_key)

    # 修正非 ASCII 字符，避免编码报错
    safe_prompt = prompt.encode('utf-8', errors='ignore').decode('utf-8')

    print(f"调用 GPT API，模型: gpt-3.5-turbo，提示内容: {safe_prompt}")  # 添加这行打印请求信息
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a salary prediction expert."},
                {"role": "user", "content": safe_prompt}
            ]
        )
        print(f"GPT API 响应内容: {response}")  # 添加这行打印响应信息
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ GPT API call failed: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            row_index = int(request.form['row_index'])
            if df is None:
                error_message = "❌ 未找到数据集文件，请检查文件是否存在。"
                return render_template('index.html', error_message=error_message)
            if row_index < 100 or row_index >= len(df):
                error_message = f"⚠️ 请输入一个 >= 100 且 < {len(df)} 的数字。"
                return render_template('index.html', error_message=error_message)

            selected_features = features.iloc[[row_index]]
            selected_features_encoded = pd.get_dummies(selected_features)
            selected_features_encoded = selected_features_encoded.reindex(columns=feature_names, fill_value=0)

            # Compose initial prompt for Agent A
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

            max_rounds = 5  # 最大对话轮数
            conversation = []

            # 第一轮：Agent A 预测
            agent_a_response = call_gemini(GEMINI_API_KEY, prompt_agent_a)
            conversation.append({"agent": "Agent A", "response": agent_a_response})

            for round in range(max_rounds - 1):
                if round % 2 == 0:  # Agent B 回复
                    prompt_agent_b = f"""
You are Agent B, a salary prediction expert. Here is the previous response from Agent A:

{agent_a_response}

Your task:
1. Review the previous prediction carefully.
2. If you agree with the result, explain why and confirm.
3. If you find any mistakes or inconsistencies, point them out and recalculate the correct salary.
4. Show your own final salary prediction.
Be detailed and logical in your review.
"""
                    agent_b_response = call_gpt(GPT_API_KEY, prompt_agent_b)
                    conversation.append({"agent": "Agent B", "response": agent_b_response})
                    agent_a_response = agent_b_response
                else:  # Agent A 回复
                    prompt_agent_a = f"""
You are Agent A, a salary prediction expert. Here is the previous response from Agent B:

{agent_b_response}

Your task:
1. Review the previous prediction carefully.
2. If you agree with the result, explain why and confirm.
3. If you find any mistakes or inconsistencies, point them out and recalculate the correct salary.
4. Show your own final salary prediction.
Be detailed and logical in your review.
"""
                    agent_a_response = call_gemini(GEMINI_API_KEY, prompt_agent_a)
                    conversation.append({"agent": "Agent A", "response": agent_a_response})

            return render_template('result.html',
                                   row_index=row_index,
                                   conversation=conversation)
        except ValueError:
            error_message = "❌ 输入无效，请输入一个有效的数字。"
            return render_template('index.html', error_message=error_message)
        except Exception as e:
            error_message = f"❌ 发生错误: {e}"
            return render_template('index.html', error_message=error_message)

    return render_template('index.html', error_message=None)

if __name__ == '__main__':
    app.run(debug=True)