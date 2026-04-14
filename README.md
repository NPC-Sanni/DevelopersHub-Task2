# DevelopersHub-Task2
DevelopersHub internship Task2
Task 1 — Bank Marketing (task1_bank_marketing.ipynb)

Loads UCI Bank Marketing dataset, handles unknown values, label-encodes all categoricals
Trains Logistic Regression and Random Forest with class balancing
Evaluates with Confusion Matrix, F1-Score, and ROC Curve
Uses SHAP TreeExplainer for global summary + 5 individual waterfall plots

Task 2 — Customer Segmentation (task2_customer_segmentation.ipynb)

Full EDA on Mall Customers dataset (distributions, scatter plots)
Finds optimal K via Elbow Method + Silhouette Score
Applies K-Means (K=5) and visualizes clusters in 2D using both PCA and t-SNE
Includes a cluster profile table with specific marketing strategies per segment

Task 3 — Energy Forecasting (task3_energy_forecasting.ipynb)

Parses and resamples the UCI Household Power Consumption dataset to hourly
Engineers lag features (1h, 24h, 168h), rolling stats, and time-based features
Trains and compares ARIMA, Prophet, and XGBoost
Evaluates with MAE & RMSE, and plots actual vs. forecasted for all 3 models
