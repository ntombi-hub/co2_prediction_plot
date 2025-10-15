import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

try:
    data = pd.read_csv('co2_emissions.csv')
except FileNotFoundError:
    data = pd.DataFrame({
        'Year': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
        'GDP': [1.2e12, 1.3e12, 1.4e12, 1.5e12, 1.6e12, 1.7e12, 1.8e12, 1.9e12, 2.0e12, 2.1e12],
        'Population': [6.1e9, 6.2e9, 6.3e9, 6.4e9, 6.5e9, 6.6e9, 6.7e9, 6.8e9, 6.9e9, 7.0e9],
        'Energy_Use': [400, 420, 430, 450, 470, 490, 510, 530, 550, 570],
        'CO2_Emissions': [25.0, 25.5, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0]
    })

data = data.dropna()

def normalize(column):
    return (column - column.min()) / (column.max() - column.min())

data['GDP'] = normalize(data['GDP'])
data['Population'] = normalize(data['Population'])
data['Energy_Use'] = normalize(data['Energy_Use'])
data['CO2_Emissions'] = normalize(data['CO2_Emissions'])

X = data[['GDP', 'Population', 'Energy_Use']]
y = data['CO2_Emissions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual CO2 Emissions (Normalized)')
plt.ylabel('Predicted CO2 Emissions (Normalized)')
plt.title('CO2 Emissions Prediction Results')
plt.legend()
plt.grid(True)
plt.savefig('co2_prediction_plot.png')
plt.show()

coefficients = pd.DataFrame({
    'Feature': ['GDP', 'Population', 'Energy_Use'],
    'Coefficient': model.coef_
})
print("\nModel Coefficients:")
print(coefficients)