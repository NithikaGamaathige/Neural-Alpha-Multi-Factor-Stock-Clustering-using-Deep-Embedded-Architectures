# 🤖 From Price Patterns to Portfolio DNA
### Benchmarking AI-Driven Clustering Against Traditional Methods in Equity Allocation

> **Course:** FM 3056 – Financial Simulation & AI Models  
> **Universe:** 100 S&P 500 Equities | **Period:** 2 Years of Daily Price Data  
> **Stack:** Python · TensorFlow/Keras · scikit-learn · yfinance · SciPy · Plotly

---

## 📌 Overview

This project builds an end-to-end quantitative pipeline that answers a question every portfolio manager faces:

> *Does more sophisticated pattern recognition actually translate into better portfolios?*

Three clustering frameworks — each representing a different theory of what makes stocks behave similarly — are used to construct and backtest long-only portfolios against an equal-weight benchmark, with performance evaluated on return, volatility, Sharpe ratio, and maximum drawdown.

---

## 🗂️ Project Structure

| Notebook | Description |
|---|---|
| `1_Data.ipynb` | Data ingestion, cleaning, and persistence via yfinance |
| `2_Price_Profile.ipynb` | Log return calculation, cumulative returns, structural classification |
| `3_Volatility_Profile.ipynb` | Rolling realized volatility across weekly/monthly/quarterly horizons |
| `4_KMeans.ipynb` | Traditional K-Means clustering with PCA, elbow & silhouette selection |
| `5_DEC.ipynb` | Denoising Autoencoder + Deep Embedded Clustering on latent features |
| `6_StyleDEC.ipynb` | Triple-profile fusion (Price + Volatility + Fundamentals) with StyleDEC |
| `7_Portfolios.ipynb` | Mean-variance optimisation, backtesting, and strategy comparison |

---

## 🔧 Methodology

### 1. Traditional K-Means
Clusters 100 stocks by **linear return correlations** using PCA-compressed daily log returns as features. Serves as the interpretable baseline. Optimal cluster count selected via silhouette score.

### 2. Deep Embedded Clustering (DEC)
A **denoising autoencoder** (256 → 128 → 20 bottleneck) is trained on each stock's 2-year return history, compressing it into a 20-dimensional financial fingerprint. K-Means is then applied in this learned latent space rather than raw return space — capturing nonlinear patterns invisible to traditional methods.

### 3. StyleDEC
Goes further by **fusing three factor profiles** into a single neural representation:
- 📈 **Price Profile** — PCA-compressed return dynamics
- 📉 **Volatility Profile** — Annualized realized volatility at weekly, monthly, and quarterly horizons
- 🏢 **Style Profile** — Fundamental factors: P/E ratio, D/E ratio, Dividend Yield

Each stock's cluster assignment reflects not just how it *moved*, but what kind of company it *is*.

### 4. Portfolio Construction
Each clustering output feeds a **mean-variance optimiser** (maximising utility = return − γ/2 × variance) with per-stock position caps. Portfolios are backtested on actual historical price paths and compared against a naive equal-weight benchmark.

---

## 📊 Key Findings

All three clustering-based strategies **dramatically outperformed equal-weight allocation** on a risk-adjusted basis — validating that intelligent clustering adds real value in portfolio construction.

The more interesting finding was the tradeoff *between* methods:

- **DEC and StyleDEC** delivered higher absolute returns by identifying richer behavioural patterns
- **K-Means** achieved the best Sharpe ratio through tighter, more homogeneous clusters that naturally steered the optimiser toward lower-risk positions
- **Equal-weight** was outperformed across every metric by all three methods

> The right clustering method depends on **investor mandate** — K-Means for capital preservation, StyleDEC for return-seeking investors comfortable with higher volatility.

---

## ⚠️ Limitations

- Backtest covers a **predominantly bullish 2-year window** (2024–2026). StyleDEC's fundamental factors are theoretically better suited to bear markets and sideways regimes not captured here.
- Fundamental data (P/E, D/E, Yield) fetched at a **single point in time** — does not account for factor drift over the sample period.
- All strategies use **historical covariance** for optimisation, which may not reflect future risk structure.

---

### Requirements
```
yfinance
pandas
numpy
scikit-learn
tensorflow
scipy
plotly
matplotlib
seaborn
pyarrow
```

---

## 📁 Data Files Generated

| File | Contents |
|---|---|
| `group2_stocks.parquet` | Cleaned daily adjusted close prices |
| `benchmark.parquet` | S&P 500 benchmark prices |
| `labels_kmeans.csv` | K-Means cluster assignments |
| `labels_dec.csv` | DEC cluster assignments |
| `labels_style_dec.csv` | StyleDEC cluster assignments |

---

## 🙏 Acknowledgements

Built as part of **FM 3056 – Financial Simulation & AI Models**.  
Data sourced via [yfinance](https://github.com/ranaroussi/yfinance).  
Clustering methodology inspired by [Xie et al. (2016) — Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/abs/1511.06335).
