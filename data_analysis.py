import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('employees_dataset.csv')
df = df.dropna()

y = df['salary']
X = df.drop('salary', axis=1)


y = np.log(y)


categorical_cols = ['gender', 'education_level', 'job_role', 'industry', 'certifications']
numerical_cols = ['age', 'years_of_experience']


ordinal_encoder = OrdinalEncoder()
X[['education_level']] = ordinal_encoder.fit_transform(X[['education_level']])


onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_transformer = Pipeline(steps=[
    ('onehot', onehot_encoder)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X = preprocessor.fit_transform(X)

# 截取前 100 个样本
X_100 = X[:100]
y_100 = y[:100]

X_train, X_test, y_train, y_test = train_test_split(X_100, y_100, test_size=0.2, random_state=42)

#random forest

from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [50, 100, 200, 500],        
   'max_depth': [None, 10, 20, 30],       
   'min_samples_split': [2, 5, 10],      
   'min_samples_leaf': [1, 2, 4]          
}


RF = RandomForestRegressor(random_state=42)


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)


grid_search.fit(X_train, y_train)


print("Best Parameters:", grid_search.best_params_)
print("Best Score (negative MSE):", grid_search.best_score_)


best_RF = grid_search.best_estimator_

y_pred = best_RF.predict(X_test)


mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
                         
#linear regression

from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(X_train, y_train)

y_pred = LR.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)