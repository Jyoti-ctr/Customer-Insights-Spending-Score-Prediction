# Customer Insights & Spending Score Prediction 🚀

An end-to-end ML pipeline to segment customers using **K-Means++** and predict spending behavior using **XGBoost**.

## 📊 Project Overview
- **Segmentation:** Grouping customers based on income and behavior.
- **Regression:** Predicting the 'Spending Score' (1-100) for targeted marketing.
- **Accuracy:** Achieved an R2 score of 84.4.

## 🛠️ How to Run
1. Clone the repo: `git clone https://github.com/Jyoti-ctr/Customer-Insights-Spending-Score-Prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run preprocessing: `python src/preprocess.py`
4. Run clustering: `python src/cluster.py`
5. Run prediction: `python src/predict.py`


# 🛍️ Customer Insights & Spending Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![K-Means](https://img.shields.io/badge/Clustering-K--Means%2B%2B-green)

An end-to-end Machine Learning pipeline that segments customers into distinct personas and predicts their spending behavior with **84.4% accuracy**.

## 🚀 Quick Start
1. **Clone:** `git clone https://github.com/Jyoti-ctr/Customer-Insights-Spending-Score-Prediction.git`
2. **Install:** `pip install -r requirements.txt`
3. **Run Pipeline:** `python app.py`

## 💡 Business Insights
- **Cluster 0:** High Income / Low Spenders (Target for savings-based luxury ads).
- **Cluster 2:** High Income / High Spenders (VIP customers for loyalty rewards).

## 📈 Results
The model identifies 5 distinct customer personas, allowing businesses to tailor their strategies for high-value vs. budget-conscious shoppers.
