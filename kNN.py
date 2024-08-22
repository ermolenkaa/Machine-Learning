'''
Извлеки признаки и классы, класс в первом столбце, признаки в остальных
Создай генератор разбиений для кросс-валидации (5 блоков)
Вычисли точность accuracy для алгоритма knn (sklearn KNeighborsClassifier) в цикле для k от 1 до 50, найди оптимальное k
Проведи масштабирование признаков (sklearn preprocessing scale)
и снова Вычисли точность accuracy для алгоритма knn (sklearn KNeighborsClassifier)
в цикле для k от 1 до 50, найди оптимальное k, посмотри как это повлияло на точность
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

dataframe = pd.read_csv('wine.data', header=None)

class_df = dataframe.iloc[:, 0].values.reshape(-1, 1) ## reshape - преобразование в двумерный массив для Knn
sign_df = dataframe.iloc[:, 1:].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy_before_scaling = 0
best_k_before_scaling = 0

for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k) ## обучаем модель
    accuracy_scores = cross_val_score(knn, sign_df, class_df.ravel(), cv=kf) ## ravel - преобразование в одномерный массив
    mean_accuracy = accuracy_scores.mean()
    if mean_accuracy > best_accuracy_before_scaling:
        best_accuracy_before_scaling = mean_accuracy
        best_k_before_scaling = k

class_df_scaled = scale(sign_df) ## масштабирование признаков

best_accuracy_after_scaling = 0
best_k_after_scaling = 0

for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    accuracy_scores = cross_val_score(knn, class_df_scaled, class_df.ravel(), cv=kf)
    mean_accuracy = accuracy_scores.mean()
    if mean_accuracy > best_accuracy_after_scaling:
        best_accuracy_after_scaling = mean_accuracy
        best_k_after_scaling = k

print("Best accuracy before scaling:", best_accuracy_before_scaling)
print("Best k before scaling:", best_k_before_scaling)
print("Best accuracy after scaling:", best_accuracy_after_scaling)
print("Best k after scaling:", best_k_after_scaling)