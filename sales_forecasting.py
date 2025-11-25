import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import joblib
import os

# ---------- 1) Load Dataset ----------
csv_path = "../data/sales.csv"
df = pd.read_csv(csv_path, parse_dates=['date'])

# Prophet requires columns: ds and y
df = df.rename(columns={'date': 'ds', 'sales': 'y'})
df = df.sort_values('ds')

# ---------- 2) Training the Model ----------
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(df)

# ---------- 3) Forecast for next 60 days ----------
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# ---------- 4) Save Outputs Folder ----------
os.makedirs("../outputs", exist_ok=True)

# Save forecast CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("../outputs/forecast.csv", index=False)

# Save Model
joblib.dump(model, "../outputs/prophet_model.joblib")

# ---------- 5) Show and Save Plots ----------
fig1 = model.plot(forecast)
plt.title("Sales Forecast")
fig1.savefig("../outputs/sales_forecast_plot.png")
plt.close()

fig2 = model.plot_components(forecast)
fig2.savefig("../outputs/sales_components_plot.png")
plt.close()

print("Task 1 completed successfully!")
print("Forecast and model saved in outputs folder.")
