import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# Загружаем данные с правильными параметрами
data = pd.read_csv(
    "C:\\моё\\Универ\\5 семетср\\МО\\Wholesale customers data.csv",
    delimiter=',',
    header=0
)
print(f"Названия столбцов: {data.columns.tolist()}")
print(f"Размер: {data.shape[0]} строк, {data.shape[1]} столбцов")
print(f"\nПервые 5 строк:\n{data.head()}")


#столбцы для кластеризации
features = data
feature_names = features.columns.tolist()
print(f"Признаки для кластеризации: {feature_names}")

# Масштабирование данных
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

#ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

#метода локтя
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o', linewidth=2, markersize=8)
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Инерция')
plt.title('Метод локтя для определения оптимального k')
plt.grid(True, alpha=0.3)
plt.show()

differences = np.diff(inertia)
optimal_k = 3  
for i in range(1, len(differences)-1):
    if differences[i] > 0.7 * differences[i-1]:
        optimal_k = i + 1
        break

print(f"Оптимальное число кластеров: k = {optimal_k}")

# ПРИМЕНЕНИЕ 3 АЛГОРИТМОВ
print("4. Применение алгоритмов кластеризации:")

# Алгоритм 1: K-Means
print("  - K-Means...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_features)

# Алгоритм 2: Agglomerative Clustering
print("  - Agglomerative Clustering...")
agglo = AgglomerativeClustering(n_clusters=optimal_k)
agglo_labels = agglo.fit_predict(scaled_features)

# Алгоритм 3: DBSCAN
print("  - DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_features)

# 5. ОЦЕНКА КАЧЕСТВА
print("\n")
print("5. Оценка качества кластеризации:")

metrics = []
algorithms = ['K-Means', 'Agglomerative', 'DBSCAN']
labels_list = [kmeans_labels, agglo_labels, dbscan_labels]

for algo, labels in zip(algorithms, labels_list):
    if len(np.unique(labels)) > 1:
        score = silhouette_score(scaled_features, labels)
    else:
        score = np.nan
    
    n_clusters = len(np.unique(labels))
    if -1 in labels:
        n_clusters -= 1
    
    metrics.append([algo, n_clusters, round(score, 4) if not np.isnan(score) else 'N/A'])

# Вывод результатов
results_df = pd.DataFrame(metrics, columns=['Алгоритм', 'Кластеры', 'Silhouette Score'])
print("\n" + results_df.to_string(index=False))

# 6. ВИЗУАЛИЗАЦИЯ
print("\n")
print("6. Визуализация результатов...")

# PCA для 2D визуализации
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
variance_explained = sum(pca.explained_variance_ratio_) * 100
print(f"Объясненная дисперсия: {variance_explained:.1f}%")

# Создаем графики
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Исходные данные
axes[0,0].scatter(pca_result[:,0], pca_result[:,1], alpha=0.6, c='gray', s=50)
axes[0,0].set_title('Исходные данные')
axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0,0].grid(True, alpha=0.3)

# K-Means
scatter1 = axes[0,1].scatter(pca_result[:,0], pca_result[:,1], 
                             c=kmeans_labels, cmap='viridis', alpha=0.7, s=50)
axes[0,1].set_title(f'K-Means ({optimal_k} кластеров)')
axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0,1].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0,1])

# Agglomerative
scatter2 = axes[1,0].scatter(pca_result[:,0], pca_result[:,1], 
                             c=agglo_labels, cmap='plasma', alpha=0.7, s=50)
axes[1,0].set_title(f'Agglomerative ({optimal_k} кластеров)')
axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[1,0].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1,0])

# DBSCAN
n_dbscan_clusters = len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
scatter3 = axes[1,1].scatter(pca_result[:,0], pca_result[:,1], 
                             c=dbscan_labels, cmap='coolwarm', alpha=0.7, s=50)
axes[1,1].set_title(f'DBSCAN ({n_dbscan_clusters} кластеров)')
axes[1,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1,1].grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=axes[1,1])

plt.suptitle('Результаты кластеризации (PCA проекция)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 7. АНАЛИЗ КЛАСТЕРОВ K-MEANS
print("7. Анализ характеристик кластеров (K-Means):")

# Добавляем метки к данным
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = kmeans_labels

# Размеры кластеров
cluster_sizes = data_with_clusters['Cluster'].value_counts().sort_index()
print("\nРазмеры кластеров:")
for cluster, size in cluster_sizes.items():
    percentage = (size / len(data)) * 100
    print(f"  Кластер {cluster}: {size} объектов ({percentage:.1f}%)")

# Средние значения по кластерам
print("\nСредние значения по кластерам:")
cluster_means = data_with_clusters.groupby('Cluster')[feature_names].mean()

# Отображаем в читаемом формате
print("\nТаблица средних значений:")
print(cluster_means.round(2).to_string())

# Определяем лучший алгоритм
valid_scores = results_df[results_df['Silhouette Score'] != 'N/A']
if not valid_scores.empty:
    valid_scores['Silhouette Score'] = valid_scores['Silhouette Score'].astype(float)
    best_idx = valid_scores['Silhouette Score'].idxmax()
    best_algo = valid_scores.loc[best_idx, 'Алгоритм']
    best_score = valid_scores.loc[best_idx, 'Silhouette Score']
else:
    best_algo = "Не определен"
    best_score = "N/A"


print("ВЫВОДЫ:")
print(f"\n1. Оптимальное число кластеров: {optimal_k}")
print(f"\n2. Лучший алгоритм: {best_algo} (Silhouette Score: {best_score})")
print(f"\n3. Объясненная дисперсия PCA: {variance_explained:.1f}%")
print(f"\n4. Количество объектов: {len(data)}")
print(f"\n5. Количество признаков: {len(feature_names)}")



data_with_clusters.to_csv('clustering_results.csv', index=False, encoding='utf-8')

