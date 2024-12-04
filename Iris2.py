import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import random

data = load_iris()
iris = data.data

def compute_distances(point, centroids):
    return np.linalg.norm(point - centroids, axis=1)

# Инициализация центроидов случайными точками из датасета
def initialize_centroids(data, k):
    indices = random.sample(range(data.shape[0]), k)
    return data[indices]

# Основной алгоритм k-means
def kmeans(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    clusters = np.zeros(data.shape[0])
    steps = []  # Храним все шаги для визуализации

    for _ in range(max_iters):
        steps.append((centroids.copy(), clusters.copy()))

        # Шаг 1: Назначить точки к ближайшему кластеру
        for i, point in enumerate(data):
            distances = compute_distances(point, centroids)
            clusters[i] = np.argmin(distances)

        # Шаг 2: Пересчитать центроиды
        new_centroids = np.array([data[clusters == j].mean(axis=0) if len(data[clusters == j]) > 0 else centroids[j] for j in range(k)])

        # Проверка на сходимость (если центроиды не изменились)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    steps.append((centroids, clusters))  # Финальный шаг
    return steps

k = 3
steps = kmeans(iris, k)

# Преобразуем датасет в двумерные точки, чтобы отображать их на графике
pca = PCA(n_components=2)
iris_2d = pca.fit_transform(iris)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['red', 'blue', 'green']

def update_4d(step):
    ax.clear()
    centroids, clusters = steps[step]

    # Преобразуем центроиды в 2D
    centroids_2d = pca.transform(centroids)

    # Рисуем точки
    for i in range(k):
        cluster_points = iris_2d[clusters == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f"Cluster {i+1}", alpha=0.6)

    # Рисуем центроиды
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', marker='x', s=200, label='Centroids')
    ax.set_title(f"Step {step + 1}/{len(steps)} (4D to 2D projection)")
    ax.legend()
    ax.grid(True)

ani_4d = FuncAnimation(fig, update_4d, frames=len(steps), interval=1000)
plt.show()
