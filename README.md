# Deep Learning for Multi-Factor Portfolio Construction
## Academic Project: FM 3056 – Financial Simulation and Artificial Intelligence Models
### Benchmarking K-Means, Deep Embedded Clustering (DEC), and Style-Aware Autoencoders
### 📌 Project Overview
Developed as part of the Financial Simulation and Artificial Intelligence Models (FM 3056) curriculum, this project explores the boundary between classical financial econometrics and modern deep learning. The objective is to construct a production-grade quantitative pipeline that clusters 101 equities (S&P 500 Group 2) and optimizes portfolios by comparing traditional statistical methods against non-linear AI architectures.
### 🏗️ The Quantitative Pipeline
The research is structured as a modular "Quant Lab" workflow, where each notebook serves a specific stage in the financial simulation process:

1. Data Ingestion & Risk Profiling


01_Data_Collection.ipynb: Automated extraction of historical price data and S&P 500 benchmarks.

02_Price_Profile.ipynb: Implementation of log-return transformations to ensure stationarity for AI model training.

03_Volatility_Profile.ipynb: Simulation of the Volatility Term Structure (Weekly, Monthly, and Quarterly horizons) to create a multi-dimensional risk signature for each asset.


2. Comparative AI Clustering ArchitecturesWe benchmark three distinct "Financial DNA" signatures to group stocks:


04_Traditional_KMeans.ipynb: Baseline clustering using linear correlations of price returns.

05_Deep_Embedded_Clustering_(DEC).ipynb: A deep learning approach using a Bottleneck Autoencoder to compress 500 days of market "noise" into a 20-dimensional latent space, identifying non-linear behavioral patterns.

06_StyleDEC_Fusion.ipynb: The most advanced model in the project. It fuses Price, Volatility, and Fundamental Style Factors (P/E, D/E, Yield) into a "Triple-Profile" input, trained on a style-aware neural network.


3. Portfolio Synthesis & Simulation


07_Portfolio_Optimization.ipynb: Final synthesis using Mean-Variance Optimization (MVO). We simulate three competing strategies (K-Means, DEC, and StyleDEC) under a 10% individual position cap constraint to maximize utility ($U = R_p - \frac{\gamma}{2}\sigma_p^2$).

### 🛠️ Tech Stack & AI Models
AI/ML: TensorFlow/Keras (Autoencoders), Scikit-Learn (Unsupervised Learning, PCA).

Simulation: SciPy.optimize (Constrained Nonlinear Optimization), NumPy, Pandas.

Data Infrastructure: yFinance API, PyArrow (Parquet Format).

Visualization: Plotly (Interactive Risk Maps), Seaborn, Matplotlib.
