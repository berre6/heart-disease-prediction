import os

print("Çalışma dizini:", os.getcwd())
print("Data klasörü var mı?:", os.path.exists("data"))
print("CSV var mı?:", os.path.exists("data/heart_disease.csv"))


import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("TkAgg")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# 1. VERİYİ OKU
# =========================
df = pd.read_csv(
    r"C:\Users\talha\Documents\heart_disease_project\data\heart_disease.csv"
)


print("İlk 5 satır:")
print(df.head())
print("\nSütunlar:")
print(df.columns)

# =========================
# 2. TARGET OLUŞTUR
# num > 0 ise hastalık var
# =========================
df["has_disease"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

print("\nKalp hastalığı dağılımı:")
print(df["has_disease"].value_counts())

# =========================
# 3. BOŞ DEĞERLERİ SİL
# =========================
df = df.replace("?", pd.NA)
df = df.dropna()
print("==== DEBUG ====")
print("Satır sayısı:", len(df))
print("Dağılım:")
print(df["has_disease"].value_counts())
print("==== DEBUG BİTTİ ====")

print("\nTemizlendikten sonra hastalık dağılımı:")
print(df["has_disease"].value_counts())

# =========================
# 3.5 HASTALIK DAĞILIM GRAFİĞİ
# =========================
counts = df["has_disease"].value_counts()
plt.figure()
counts = df["has_disease"].value_counts()

plt.bar(counts.index.astype(str), counts.values)
plt.xlabel("Kalp Hastalığı (0 = Yok, 1 = Var)")
plt.ylabel("Kişi Sayısı")
plt.title("Kalp Hastalığı Dağılımı")
plt.show()





# =========================
# 4. FEATURE ve TARGET
# =========================
X = df[["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]]
y = df["has_disease"]

# =========================
# 5. TRAIN - TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. MODEL
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# 7. TAHMİN ve DOĞRULUK
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel doğruluğu:", accuracy)

# =========================
# 8. BASİT GRAFİK
# =========================
plt.figure()
plt.bar(["Accuracy"], [accuracy])
plt.ylim(0, 1)
plt.title("Model Accuracy")
plt.show()




