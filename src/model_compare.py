import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 1. VERİYİ OKU
# =========================
df = pd.read_csv("data/heart_disease.csv")

# Target
df["has_disease"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# Temizlik
df = df.replace("?", pd.NA).dropna()

# Feature & target
X = df[["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]]
y = df["has_disease"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 2. MODELLER
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(random_state=42)
}

accuracies = {}

# =========================
# 3. EĞİT + DEĞERLENDİR
# =========================
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.3f}")

# =========================
# 4. GRAFİK
# =========================
plt.bar(accuracies.keys(), accuracies.values())
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
