# Market Ad Forecasting on Facebook and Instagram

## Overview
Businesses invest heavily in digital ads on Facebook and Instagram. This project builds predictive models to forecast three key performance metrics—Impressions, Clicks, and Conversions—using historical synthetic data.

## Project Structure
- **ad_forecasting_all_models.py**: End-to-end script for data generation, preprocessing, feature engineering, model training (RandomForest, XGBoost, LSTM), and evaluation.
- **ad_forecasting_tuned_comparison.py**: Script with hyperparameter tuning (RandomizedSearchCV for tabular models, Keras Tuner for LSTM) and comparison.
- **sample_predictions.csv**: Actual vs. predicted values on a held-out test set.
- **best_rf_impressions.pkl**, **best_rf_clicks.pkl**, **best_rf_conversions.pkl**: Serialized RandomForest pipelines for deployment.
- **best_lstm.h5**, **lstm_scaler.pkl**: Saved LSTM model and scaler.
- **app.py**: FastAPI service to serve online predictions.

---

## 1. Data Generation
We synthesize `n=10,000` (or `5,000` in the lighter version) ad-level records with:
- **Platform**: Facebook or Instagram
- **Budget**: Random USD between 100–1000
- **Ad_Type**: Image, Video, Carousel, SlideShow, Collection
- **Target_Age**: 14–18, 18–25, 18–30, 26–35, 35–50
- **Target_Gender**: Male, Female, All
- **Impressions, Clicks, Conversions**: Derived via randomized rules reflecting typical CTR and conversion rates

## 2. Data Preprocessing
1. **Missing Values**: Drop any rows with nulls (synthetic data contains none).
2. **Outlier Removal**: IQR method on Budget, Impressions, Clicks, Conversions.
3. **Correlation Analysis**: Visualized raw numeric correlations via heatmap.

## 3. Feature Engineering
- **CTR** = Clicks / Impressions
- **ConversionRate** = Conversions / Clicks (zero‐fill on division by zero)
- **7‑day rolling averages** per platform for each metric to capture short‑term trends.

## 4. Model Training & Tuning

### 4.1 Tabular Models (RandomForest, XGBoost)
- **Pipeline**: OneHotEncode categoricals + StandardScale numericals + Regressor
- **Hyperparameter Tuning**: Manual grid search (light version) or `RandomizedSearchCV` (tuned version)

### 4.2 Sequence Model (LSTM)
- **Aggregation**: Daily sums of Impressions, Clicks, Conversions
- **Scaling**: `MinMaxScaler` to [0,1]
- **Sliding Window**: 7‑day input, next‑day multi‑output target
- **Architecture**: LSTM(64) → Dropout(0.2) → Dense(32, relu) → Dense(3)
- **Tuning**: Keras Tuner or default settings in light version

## 5. Evaluation & Model Comparison
| Model         | Impr MAE | Impr RMSE | Clicks MAE | Clicks RMSE | Conv MAE | Conv RMSE |
|---------------|----------|-----------|------------|-------------|----------|-----------|
| RandomForest  | 1996.03  | 2569.95   | 193.61     | 253.10      | 14.10    | 18.69     |
| XGBoost       | 2050.39  | 2645.93   | 195.58     | 260.64      | 14.74    | 19.73     |
| LSTM          | 256849.75| 309138.63 | 25785.94   | 32113.44    | 2126.73  | 2563.61   |

**Selected Model**: RandomForest (lowest MAE & RMSE on all metrics).

## 6. Saving Models
```bash
# Tabular pipelines
joblib.dump(best['RF']['Impressions'],  "best_rf_impressions.pkl")
joblib.dump(best['RF']['Clicks'],       "best_rf_clicks.pkl")
joblib.dump(best['RF']['Conversions'],  "best_rf_conversions.pkl")

# LSTM and scaler
model_lstm.save("best_lstm.h5")
joblib.dump(scaler, "lstm_scaler.pkl")
```

## 7. Sample Predictions CSV
Run the following after loading the best pipelines:
```python
imp_pred = rf_imp_pipe.predict(X_test)
clk_pred = rf_clk_pipe.predict(X_test)
conv_pred= rf_conv_pipe.predict(X_test)
# Build df with actual vs predicted and save:
df_preds.to_csv("sample_predictions.csv", index=False)
```
The CSV contains all feature columns plus:
- Actual_Impressions, Predicted_Impressions
- Actual_Clicks, Predicted_Clicks
- Actual_Conversions, Predicted_Conversions

## 8. API Deployment (FastAPI)
- **app.py** loads the three `.pkl` pipelines and exposes `/predict`.
- **Request**: JSON with the 10 feature fields.
- **Response**: JSON with forecasted Impressions, Clicks, Conversions.

### Run the API
```bash
pip install fastapi uvicorn pandas scikit-learn joblib
uvicorn app:app --reload
```

## Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, tensorflow, keras-tuner, seaborn, matplotlib, joblib, fastapi, uvicorn

---

For any questions or issues, please contact the author.  

