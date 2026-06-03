# Outlier Detection in Network Traffic
A comparative study of outlier and anomaly detection using multiple supervised and unsupervised machine learning approaches.

## 📌 Project Overview
This project evaluates how different machine learning paradigms interpret anomaly detection using a small, specialized network dataset. We explore three distinct perspectives for both supervised and unsupervised methods to understand how model architecture impacts classification performance on simple datasets.

## 📊 Dataset Profile
* **Source:** Network Anomaly Dataset (Kaggle)
* **Size:** 1,654 entries | 5 columns
* **Features (4):** * Inbound Rate
    * Outbound Rate
    * Inbound Bandwidth Utilization
    * Outbound Bandwidth Utilization
* **Target (1):** `Label` (0 = Normal, 1 = Anomaly)
* **Data Characteristics:** The original labeling relies on a zero-based system where negative values at or below `-0.79` are classified as anomalies. Because the variance is minuscule, the dataset is highly uniform and compact.

---

## ⚙️ Methodology & Dimensionality Reduction
Due to the simplicity of having only 4 features, **Principal Component Analysis (PCA)** was applied to reduce the space to 2 principal components. This further compressed the dataset, testing the models' boundaries on low-dimensional, dense data.

---

## 🚀 Supervised Learning

We selected three models, each representing a fundamentally different mathematical approach:
1.  **Naive Bayes** (Probability-based)
2.  **LinearSVC** (Geometric-based)
3.  **Random Forest** (Ensemble-based)

### Supervised Insights
While these near-perfect scores initially hint at **overfitting**, the root cause is the dataset's scale and simplicity. With only 1,654 entries and 4 highly correlated inbound/outbound features compressed by PCA, the decision boundaries are incredibly distinct, allowing supervised models to separate the classes easily.

---

## 🔍 Unsupervised Learning (Clustering)

To evaluate how well the data separates without prior labeling, we chose three distinct clustering philosophies:
1.  **K-Means** (Centroid-based)
2.  **Gaussian Mixture Models (GMM)** (Probabilistic-based)
3.  **DBSCAN** (Density-based)

### Performance Metrics
* **K-Means:** Silhouette Score = **0.8156** (Best Performer)
* **GMM:** Silhouette Score = **0.6862**
* **DBSCAN:** Failed to find multiple clusters (Detected only 1 cluster)

### Unsupervised Insights
Intuitively, density-based algorithms like DBSCAN are favored for outlier detection. However, DBSCAN failed here because the dataset is too uniform and tightly packed. The fact that K-Means yielded the highest silhouette score confirms that the data groups cleanly around central prototypes rather than forming arbitrary density shapes. This aligns with our finding that the original target labels separate at a very specific, narrow threshold (`-0.79`).

---

## 🏁 Conclusion
* **Best Overall Model:** **Random Forest** achieved the highest accuracy (**99%**) and the best balance of precision and recall.
* Supervised models heavily outperformed unsupervised models, which is expected given that they benefit from explicit ground-truth labels. 
* The project demonstrates that for small, low-dimensional, and clean datasets, simple ensemble methods or centroid-based clustering yield highly effective boundaries.
