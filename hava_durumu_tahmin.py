import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# --- Veri Yükleme ve Temel Ön İşleme ---
df = pd.read_csv("istanbul_10000_gun.csv")
df['date'] = pd.to_datetime(df['date'])

# -- Eksik verileri ve gereksiz sütunları kaldırıyoruz
df = df[['date', 'tavg', 'tmin', 'tmax', 'wspd', 'wdir']]
df.dropna(inplace=True)

# Yeni özellikler oluşturulan kısım
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# --- EDA (Keşifsel Veri Analizi) ---

print("Veri Genel Bilgisi:")
print(df.info())
print("\nEksik Değer Sayısı:")
print(df.isnull().sum())

print("\nİstatistiksel Özet:")
print(df.describe())

#- 1) Günlük Ortalama Sıcaklık Zaman Serisi
plt.figure(figsize=(15,5))
plt.plot(df['date'], df['tavg'], label='Ortalama Sıcaklık (tavg)', color='orange')
plt.title('İstanbul Günlük Ortalama Sıcaklık (Zaman Serisi)')
plt.xlabel('Tarih')
plt.ylabel('Sıcaklık (°C)')
plt.legend()
plt.show()

#--  2) Aylara Göre Ortalama Sıcaklık Dağılımı->Boxplot
plt.figure(figsize=(10,5))
sns.boxplot(x='month', y='tavg', data=df)
plt.title('Aylara Göre Ortalama Sıcaklık Dağılımı')
plt.xlabel('Ay')
plt.ylabel('Ortalama Sıcaklık (°C)')
plt.show()

#-- 3) Korelasyon Matrisii
plt.figure(figsize=(8,6))
corr = df[['tavg', 'tmin', 'tmax', 'wspd', 'wdir']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Özellikler Arası Korelasyon Matrisi')
plt.show()

# --4) Sıcaklık Dağılımları ->Histogramlar
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

#-5) Aykırı Değer Analizi -> Boxplot
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

# --- Modelleme için Veri Hazırlığı yaptığımız yer---
features = ['tmin', 'tmax', 'wspd', 'wdir', 'month']
X = df[features]
y = df['tavg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelleri tanımladık--
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR()
}

predictions = {}
results = {}

# --Modelleri eğittik ve değerlendirdim
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R²": r2}
    print(f"{name}: MSE={mse:.3f}, R²={r2:.4f}")

# --- Grafiklerle model karşılaştırma yaptık ---

sample_range = range(100)

plt.figure(figsize=(16, 12))

# --1) Tahmin vs Gerçek Değerler
plt.subplot(3,1,1)
plt.plot(sample_range, y_test.iloc[sample_range], label="Gerçek", color='black', linewidth=2)
colors = ['red', 'green', 'blue', 'orange', 'purple']
for i, (name, y_pred) in enumerate(predictions.items()):
    plt.plot(sample_range, y_pred[sample_range], label=name, alpha=0.7, color=colors[i])
plt.title("Tahmin vs Gerçek - İlk 100 Test Örneği")
plt.ylabel("Ortalama Sıcaklık (°C)")
plt.legend()

#-- 2) Hata Dağılımı -> Histogram
plt.subplot(3,1,2)
for i, (name, y_pred) in enumerate(predictions.items()):
    errors = y_test.values - y_pred
    plt.hist(errors, bins=30, alpha=0.5, label=name, color=colors[i])
plt.title("Modellerin Tahmin Hata Dağılımı")
plt.xlabel("Hata (Gerçek - Tahmin)")
plt.ylabel("Frekans")
plt.legend()

# -- 3) Residual Plot (Tahmin Edilen vs Hata)
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
