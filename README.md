# Customer Insights & Spending Score Prediction 🚀

An end-to-end ML pipeline to segment customers using **K-Means++** and predict spending behavior using **XGBoost**.

## 📊 Project Overview
- **Segmentation:** Grouping customers based on income and behavior.
- **Regression:** Predicting the 'Spending Score' (1-100) for targeted marketing.
- **Accuracy:** Achieved an R2 score of [Insert your score here].

## 🛠️ How to Run
1. Clone the repo: `git clone [your-link]`
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
1. **Clone:** `git clone https://github.com/YOUR_USERNAME/repo-name.git`
2. **Install:** `pip install -r requirements.txt`
3. **Run Pipeline:** `python main_run.py`
4. **Predict New Data:** `python predict_new.py`

## 📂 Project Structure
- `src/`: Modular Python scripts for preprocessing, clustering, and modeling.
- `models/`: Serialized `.pkl` files of the trained AI.
- `data/`: Raw and processed datasets.
- `notebooks/`: Exploratory Data Analysis (EDA).

## 💡 Business Insights
- **Cluster 0:** High Income / Low Spenders (Target for savings-based luxury ads).
- **Cluster 2:** High Income / High Spenders (VIP customers for loyalty rewards).

## 📈 Results
The model identifies 5 distinct customer personas, allowing businesses to tailor their strategies for high-value vs. budget-conscious shoppers.