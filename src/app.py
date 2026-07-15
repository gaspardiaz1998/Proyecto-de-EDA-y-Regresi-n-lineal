from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Carga y limpieza
df = pd.read_csv('medical_insurance_cost.csv')
df = df.drop_duplicates().reset_index(drop=True)

#  2. Encoding
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# 3. Feature engineering
df['bmi_smoker'] = df['bmi'] * df['smoker']
df['obeso'] = (df['bmi'] >= 30).astype(int)
df['obeso_smoker'] = df['obeso'] * df['smoker']

# 4. Split
X = df.drop(columns=['charges'])
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Entrenamiento (modelo final)
modelo_final = LinearRegression()
modelo_final.fit(X_train, y_train)

# 6. Evaluación
y_test_pred = modelo_final.predict(X_test)
print("=== MODELO FINAL - TEST ===")
print("R²:", r2_score(y_test, y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("MAE:", mean_absolute_error(y_test, y_test_pred))

# 7. Función de predicción para nuevos clientes
def predecir_costo(age, sex, bmi, children, smoker, region):
    obeso = 1 if bmi >= 30 else 0
    smoker_bin = 1 if smoker == 'yes' else 0
    sex_bin = 1 if sex == 'male' else 0

    fila = {
        'age': age,
        'sex': sex_bin,
        'bmi': bmi,
        'children': children,
        'smoker': smoker_bin,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0,
        'bmi_smoker': bmi * smoker_bin,
        'obeso': obeso,
        'obeso_smoker': obeso * smoker_bin
    }
    X_nuevo = pd.DataFrame([fila])[X_train.columns]
    return modelo_final.predict(X_nuevo)[0]

# Ejemplo de uso
print("\nEjemplo - Cliente fumador y obeso:")
print(round(predecir_costo(40, 'male', 34, 1, 'yes', 'northeast'), 2))