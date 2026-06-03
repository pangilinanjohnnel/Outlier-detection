import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import silhouette_score
import os
os.environ["OMP_NUM_THREADS"] = "1"

df=pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\data mining\activity\D15.2\archive\networkanomalydataset.csv")

stats = df.groupby('Label')['Inbound Rate(bit/s)'].agg(['min', 'max', 'mean'])
print(f"IR:{stats}")

stats = df.groupby('Label')['Outbound Rate(bit/s)'].agg(['min', 'max', 'mean'])
print(f"OR:{stats}")

stats = df.groupby('Label')['Inbound Bandwidth Utilization(%)'].agg(['min', 'max', 'mean'])
print(f"IBU:{stats}")

stats = df.groupby('Label')['Outbound Bandwidth Utilization(%)'].agg(['min', 'max', 'mean'])
print(f"OBU:{stats}")

X=df.drop(columns="Label")
y=df["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#SUPERVISED
# Naive-Bayes
nb = Pipeline([
    ("pca", PCA(n_components=2)),
    ("nb", GaussianNB(priors=[0.6, 0.4]))
])
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

print("Naive-Bayes")
print(metrics.confusion_matrix(y_test,nb_pred))
print(metrics.classification_report(y_test,nb_pred))

#LinearSVC
svm = Pipeline([
    ("pca", PCA(n_components=2)),
    ("svm", LinearSVC(max_iter=1000, random_state=42))
])
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("LinearSVC")
print(metrics.confusion_matrix(y_test,svm_pred))
print(metrics.classification_report(y_test,svm_pred))

#Random forest
rf = Pipeline([
    ("pca", PCA(n_components=2)),
    ("rf", RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_depth=3, random_state=42))
])
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest")
print(metrics.confusion_matrix(y_test, rf_pred))
print(metrics.classification_report(y_test, rf_pred))

#UNSUPERVISED
X_sample = X_scaled[:10000] if len(X_scaled) > 10000 else X_scaled

#K-MEANS
km = Pipeline([
    ("pca", PCA(n_components=2)),
    ("kmeans", KMeans(n_clusters=2, random_state=42))
])
km_labels = km.fit_predict(X_sample)
kmX_pca = km.named_steps["pca"].transform(X_sample)

print(f"K-Means: {silhouette_score(kmX_pca, km_labels):.4f}")

#GMM
gmm = Pipeline([
    ("pca", PCA(n_components=2)),
    ("gmm", GaussianMixture(n_components=2, random_state=42))
])
gmm_labels = gmm.fit_predict(X_sample)
gmX_pca = gmm.named_steps["pca"].transform(X_sample)

print(f"GMM: {silhouette_score(gmX_pca, gmm_labels):.4f}")

#DBSCAN
db = Pipeline([
    ("pca", PCA(n_components=2)),
    ("db", DBSCAN(eps=0.5, min_samples=5))
])
db_labels = db.fit_predict(X_sample)
dbX_pca = db.named_steps["pca"].transform(X_sample)

if len(set(db_labels)) > 1:
    print(f"DBSCAN: {silhouette_score(dbX_pca, db):.4f}")
else:
    print("DBSCAN: Only one cluster or only noise found.")

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
print(f"Clusters found: {n_clusters}")

