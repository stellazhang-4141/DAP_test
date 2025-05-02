import pandas as pd
import requests
import openai
from dotenv import load_dotenv

# 1. Load dataset
df = pd.read_csv("employees_dataset.csv")

# 2. Prepare features and target
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

print(" Linear Regression model trained successfully!")

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
        print(" Gemini API Key seems invalid. Please check your key.")
        return None
    
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
            print(f" Gemini API call failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f" Gemini call error: {e}")
        return None

def call_gpt(api_key, prompt):
    if not api_key or not api_key.startswith("sk-"):
        print(" GPT API Key seems invalid. Please check your key.")
        return None
    
    client = openai.OpenAI(api_key=api_key)

    # 修正非ASCII字符，避免编码报错
    safe_prompt = prompt.encode('utf-8', errors='ignore').decode('utf-8')

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a salary prediction expert."},
                {"role": "user", "content": safe_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f" GPT API call failed: {e}")
        return None

# 8. Prediction & Discussion Loop
while True:
    try:
        user_input = input("\nEnter the row number you want to predict (e.g., 101), or type 'quit' to exit: ")
        if user_input.lower() == 'quit':
            print("Program exited. Goodbye!")
            break

        row_index = int(user_input)
        if row_index < 100 or row_index >= len(df):
            print(f" Please enter a number >= 100 and < {len(df)}.")
            continue

        selected_features = features.iloc[[row_index]]
        selected_features_encoded = pd.get_dummies(selected_features)
        selected_features_encoded = selected_features_encoded.reindex(columns=feature_names, fill_value=0)

        # Compose prompt for Agent A
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

        # Step 1: Agent A (Gemini) predicts salary
        print("\n Agent A (Gemini) Prediction:")
        agent_a_response = call_gemini(GEMINI_API_KEY, prompt_agent_a)
        print(agent_a_response)

        # Compose prompt for Agent B (based on Agent A's answer)
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

        # Step 2: Agent B (GPT) reviews and discusses
        print("\n Agent B (GPT) Discussion and Final Decision:")
        agent_b_response = call_gpt(GPT_API_KEY, prompt_agent_b)
        print(agent_b_response)

    except Exception as e:
        print(f" An error occurred: {e}")