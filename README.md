# 🌍 Air Quality Index (AQI) Prediction Model

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-green.svg)](#)

---

## 🚀 Project Overview

This project builds a **machine learning model** to predict the **Air Quality Index (AQI)** using pollutants such as CO, Ozone, NO₂, and PM2.5 values.  
It uses **Random Forest Regressor** as the primary model and evaluates performance with metrics such as **MAE, MSE, and R² Score**.  

Key highlights:
- Data cleaning & preprocessing  
- Exploratory Data Analysis (pairplot, correlation heatmap)  
- Model training with Random Forest  
- Performance evaluation & visualization of predictions  

---

## 📂 Repository Structure

```
├─ data/
│  └─ air_quality_data.csv      # dataset file
├─ notebooks/
│  └─ aqi_prediction.ipynb      # Jupyter notebook
├─ src/
│  └─ aqi_model.py              # Python script version
├─ screenshots/
│  └─ output_plot.png           # example prediction vs actual plot
├─ README.md
└─ requirements.txt
```

---

## ⚙️ Requirements

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🧭 Installation & Setup

1. Clone the repo:
```bash
git clone https://github.com/<username>/aqi-prediction.git
cd aqi-prediction
```

2. Add dataset (`air_quality_data.csv`) inside the `data/` folder.

3. Run the notebook or Python script:
```bash
jupyter notebook notebooks/aqi_prediction.ipynb
```
or
```bash
python src/aqi_model.py
```

---

## ▶️ Code Workflow

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('air_quality_data.csv')

# Data cleaning
data = data.dropna()
data.columns = [col.strip().lower() for col in data.columns]

# EDA
sns.pairplot(data)
plt.show()
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Features & Target
X = data[['co aqi value', 'ozone aqi value', 'no2 aqi value', 'pm2.5 aqi value']]
y = data['aqi value']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual AQI')
plt.plot(y_pred, label='Predicted AQI', alpha=0.7)
plt.title('Actual vs Predicted AQI')
plt.legend()
plt.show()
```

---

## 📊 Results

- **Mean Absolute Error (MAE):** ~X.XX  
- **Mean Squared Error (MSE):** ~X.XX  
- **R² Score:** ~X.XX  

📈 Example plot:  
![Prediction vs Actual](./screenshots/output_plot.png)

---

## ✅ Future Improvements

- Try Gradient Boosting, XGBoost, or Neural Networks  
- Add hyperparameter tuning with GridSearchCV  
- Deploy model using Flask / FastAPI  
- Build interactive dashboard (e.g., with Streamlit)  

---

## 🤝 Contributing

Contributions are welcome!  
1. Fork the repo  
2. Create a feature branch  
3. Commit changes  
4. Open a Pull Request  

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## ✉️ Contact

Maintainer: Your Name — [email@example.com](mailto:email@example.com)
