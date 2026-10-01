**British-American-Tobacco--Stock-Analysis**

An unsupervised machine learning project that uses **Agglomerative Hierarchical Clustering** on historical stock market data for **British American Tobacco (BAT)** spanning from 1997 to 2023. 

By analyzing the interaction between **Closing Price** and **Trading Volume**, this project segments historical trading days into distinct market regimes—identifying periods of routine trading, price consolidation, and extreme market volatility anomalies.

---

**Project Overview**

British_American_Tobacco_Stock Data.csv file is a British American Tobacco dataset that shows the stock trends from the year 1997 to 2023. It highlights the **historical market regimes and volatility shifts**, of which the **closing price and trading volume features** were compared to help the algorithm learn the **underlying behavioral patterns** and predict outcomes in future occurrences.

Stock market data is traditionally evaluated through linear time-series forecasting. However, price movements are heavily influenced by the underlying "state" or "regime" of the market (e.g., quiet holding phases vs. high-volume panic sell-offs).

This project uses hierarchical pattern recognition to:
1. Identify natural groupings across 25+ years of historical market sessions without human labeling bias.
2. Uncover operational regimes that distinguish routine daily volume from major liquidity spikes.
3. Establish a baseline behavioral map that can be used to feed downstream predictive models and automated risk-management alerts.

---

**Dataset & Features**

* **Dataset:** `British_American_Tobacco_Stock Data.csv` (1997 – 2023)
* **Key Features Selected (2-Feature Dimensionality):**
  * `Close Price ($)`: Daily closing valuation.
  * `Volume`: Number of shares traded per session.

---

**Tech Stack & Dependencies**

* **Python 3.x**
* **Pandas**: Data ingestion and structuring.
* **Scikit-Learn**: Feature normalization (`StandardScaler`) and clustering (`AgglomerativeClustering`).
* **SciPy**: Hierarchical linkage computation and dendrogram visualization (`scipy.cluster.hierarchy`).
* **Matplotlib**: 2D scatter plots and regime visualization.

---

**Installation & Usage**

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MidnightFolio/British-American-Tobacco--Stock-Analysis.git](https://github.com/MidnightFolio/British-American-Tobacco--Stock-Analysis.git)
   cd British-American-Tobacco--Stock-Analysis
Install required packages:

Bash
pip install pandas scikit-learn scipy matplotlib
Run the script:

Bash
python clustering.py
**Methodology**
Feature Standardization (StandardScaler):
Trading volume spans into millions of shares, while closing prices range from single digits to hundreds of dollars. To prevent Euclidean distance calculations from being dominated entirely by volume magnitude, both features were standardized.

**Hierarchical Linkage & Dendrogram:**
Ward's minimum variance method was computed using SciPy to construct a dendrogram tree. Evaluating the largest vertical distances without crossing horizontal cutoffs identified the natural cluster boundaries.

**Agglomerative Clustering:**
The algorithm iteratively grouped neighboring data points based on Euclidean proximity, outputting distinct cluster labels for each historical trading date.

**Chart Interpretation & Key Insights**
Plotting Close Price vs. Trading Volume revealed distinct operational regimes across BAT's trading history:

Routine / Baseline Accumulation Regime (Dense Low-Volume Cluster):

The majority of trading sessions fall into a consistent, low-volume baseline across both low and high price ranges, indicating standard institutional holding and dividend accumulation.

Consolidation at High Price Levels:

As the stock expanded above $300 to $700+, volume remained tightly bounded, showing low speculative turnover at peak valuations.

High-Volume Liquidity Spikes (Anomalies):

Isolated clusters showed trading volume surging up to 10M+ shares within lower price bands. These represent high-impact market events, news shocks, or restructuring phases.

**Future Applications**
Automated Risk Alerts: Using the discovered clusters as labeled ground-truth data (Market_Regime) to train a supervised classifier (e.g., Random Forest or XGBoost) that flags live volatility spikes in real-time.

Regime-Adaptive Trading: Activating specific trading or hedging rules only when current market behavior enters a designated cluster.

**Repository Structure**
Plaintext

├── British_American_Tobacco_Stock Data.csv   # Historical market data (1997 - 2023)

└── README.md                                 # Project documentation (This file)

**Author**

Thomas Oluwafemi Johnson

Agricultural & Bio-Resources Engineer | Data Science & Machine Learning Practitioner
