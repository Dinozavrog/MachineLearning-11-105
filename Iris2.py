import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from sklearn.datasets import load_iris
import random

# Загрузка данных Iris
data = load_iris()
X = data.data[:, :2]  # Используем только два признака для визуализации


# Функция для инициализации центроидов
def initialize_centroids(X, k):
    indices = random.sample(range(X.shape[0]), k)
    return X[indices, :]


# Функция для вычисления расстояний и назначения точек кластерам
def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
    return np.argmin(distances, axis=1)


# Функция для обновления центроидов
def update_centroids(X, labels, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for cluster in range(k):
        new_centroids[cluster] = X[labels == cluster].mean(axis=0)
    return new_centroids


# Основной алгоритм k-means с визуализацией
def k_means_with_visualization(X, k, max_iterations=10):
    centroids = initialize_centroids(X, k)
    labels = None
    images = []

    for iteration in range(max_iterations):
        # Назначаем точки кластерам
        labels = assign_clusters(X, centroids)

        # Рисуем текущую итерацию
        fig, ax = plt.subplots()
        for cluster in range(k):
            cluster_points = X[labels == cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster + 1}')
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
        ax.set_title(f'Iteration {iteration + 1}')
        ax.legend()
        plt.savefig(f'iteration_{iteration + 1}.png')
        images.append(f'iteration_{iteration + 1}.png')
        plt.close()

        # Сохраняем центроиды
        new_centroids = update_centroids(X, labels, k)

        # Проверка на сходимость
        if np.allclose(centroids, new_centroids):
            print(f'Сходимость достигнута на итерации {iteration + 1}')
            break
        centroids = new_centroids

    return images


# Запуск алгоритма
k = 3  # Число кластеров
images = k_means_with_visualization(X, k)

# Создание GIF
from PIL import Image

frames = [Image.open(image) for image in images]
frames[0].save('kmeans_visualization.gif', save_all=True, append_images=frames[1:], duration=1000, loop=0)
print("GIF создан: kmeans_visualization.gif")
