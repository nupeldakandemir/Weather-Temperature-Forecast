import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Veri Yükleme ve Temel Ön İşleme ---
df = pd.read_csv("istanbul_10000_gun.csv")
df['date'] = pd.to_datetime(df['date'])

df = df[['date', 'tavg', 'tmin', 'tmax', 'wspd', 'wdir']]
df.fillna(df.mean(numeric_only=True), inplace=True)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# --- EDA ---
print("Veri Genel Bilgisi:")
print(df.info())
print("\nEksik Değer Sayısı:")
print(df.isnull().sum())
print("\nİstatistiksel Özet:")
print(df.describe())

# Zaman Serisi Grafiği
plt.figure(figsize=(15,5))
plt.plot(df['date'], df['tavg'], label='Ortalama Sıcaklık (tavg)', color='orange')
plt.title('İstanbul Günlük Ortalama Sıcaklık (Zaman Serisi)')
plt.xlabel('Tarih')
plt.ylabel('Sıcaklık (°C)')
plt.legend()
plt.show()

# Aylara Göre Boxplot
plt.figure(figsize=(10,5))
sns.boxplot(x='month', y='tavg', data=df)
plt.title('Aylara Göre Ortalama Sıcaklık Dağılımı')
plt.xlabel('Ay')
plt.ylabel('Ortalama Sıcaklık (°C)')
plt.show()

# Korelasyon Matrisi
plt.figure(figsize=(8,6))
corr = df[['tavg', 'tmin', 'tmax', 'wspd', 'wdir']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Özellikler Arası Korelasyon Matrisi')
plt.show()

# Histogramlar
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
sns.histplot(df['tmin'], kde=True, color='blue')
plt.title('Minimum Sıcaklık Dağılımı')
plt.subplot(1,3,2)
sns.histplot(df['tavg'], kde=True, color='orange')
plt.title('Ortalama Sıcaklık Dağılımı')
plt.subplot(1,3,3)
sns.histplot(df['tmax'], kde=True, color='red')
plt.title('Maksimum Sıcaklık Dağılımı')
plt.tight_layout()
plt.show()

# Aykırı Değer Analizi
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
sns.boxplot(y=df['tmin'], color='blue')
plt.title('Minimum Sıcaklık Boxplot')
plt.subplot(1,3,2)
sns.boxplot(y=df['tavg'], color='orange')
plt.title('Ortalama Sıcaklık Boxplot')
plt.subplot(1,3,3)
sns.boxplot(y=df['tmax'], color='red')
plt.title('Maksimum Sıcaklık Boxplot')
plt.tight_layout()
plt.show()

# --- Modelleme için Veri Hazırlığı ---
features = ['tmin', 'tmax', 'wspd', 'wdir', 'month']
X = df[features]
y = df['tavg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Modeller ---
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR()
}

predictions = {}
results = {}

print("\n--- Model Eğitim ve Değerlendirme ---")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    y_mean = y_test.mean()
    rae = mae / np.mean(np.abs(y_test - y_mean))
    rrse = rmse / np.sqrt(np.mean((y_test - y_mean) ** 2))

    train_r2 = model.score(X_train_scaled, y_train)
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mean = cv_r2.mean()
    cv_std = cv_r2.std()

    results[name] = {
        "Train R²": train_r2,
        "Test R²": test_r2,
        "CV R² Mean": cv_mean,
        "CV R² Std": cv_std,
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "RAE": rae,
        "RRSE": rrse
    }

# --- Sonuçları Yazdır ---
print("\n--- Tüm Model Performansları ---")
for name, metrics in results.items():
    print(f"\n{name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# --- Grafiklerle Karşılaştırma ---
sample_range = range(100)
colors = ['red', 'green', 'blue', 'orange', 'purple']

plt.figure(figsize=(16, 12))

# 1) Tahmin vs Gerçek
plt.subplot(3,1,1)
plt.plot(sample_range, y_test.iloc[sample_range], label="Gerçek", color='black', linewidth=2)
for i, (name, y_pred) in enumerate(predictions.items()):
    plt.plot(sample_range, y_pred[sample_range], label=name, alpha=0.7, color=colors[i])
plt.title("Tahmin vs Gerçek - İlk 100 Test Örneği")
plt.ylabel("Ortalama Sıcaklık (°C)")
plt.legend()

# 2) Hata Dağılımı
plt.subplot(3,1,2)
for i, (name, y_pred) in enumerate(predictions.items()):
    errors = y_test.values - y_pred
    plt.hist(errors, bins=30, alpha=0.5, label=name, color=colors[i])
plt.title("Modellerin Tahmin Hata Dağılımı")
plt.xlabel("Hata (Gerçek - Tahmin)")
plt.ylabel("Frekans")
plt.legend()

# 3) Residual Plot
plt.subplot(3,1,3)
for i, (name, y_pred) in enumerate(predictions.items()):
    errors = y_test.values - y_pred
    plt.scatter(y_pred, errors, alpha=0.5, label=name, color=colors[i], s=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residual Plot (Tahmin Edilen vs Hata)")
plt.xlabel("Tahmin Edilen Değer")
plt.ylabel("Hata (Gerçek - Tahmin)")
plt.legend()

plt.tight_layout()
plt.show()
