import os
import numpy as np
import rasterio
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

path_train = 'd:/water/train_dataset_skoltech_train/train/images/'
path_mask = 'd:/water/train_dataset_skoltech_train/train/masks/'

# Получение списка файлов .tif
file_list = [f for f in os.listdir(path_train) if f.endswith('.tif')]

X = pd.DataFrame()
Y = []

for n in range(len(file_list)):
    # Чтение изображений
    with rasterio.open(os.path.join(path_train, file_list[n])) as src:
        I = src.read().astype(np.float32) / 256  # Нормализация значений

    M = rasterio.open(os.path.join(path_mask, file_list[n])).read(1)

    # Извлечение каналов
    BLUE = I[1]
    GREEN = I[2]
    RED = I[3]
    B05 = I[4]
    B06 = I[5]
    B07 = I[6]
    NIR = I[7]
    SWIR1 = I[8]
    SWIR2 = I[9]

    # Вычисления индексов
    NDWI = (NIR - SWIR2) / (NIR + SWIR2)
    WRI = (GREEN + RED) / (NIR + SWIR2)
    MNDWI = (GREEN - SWIR2) / (GREEN + SWIR2)
    NDSI = (GREEN - SWIR1) / (GREEN + SWIR1)
    NDTI = (RED - GREEN) / (RED + GREEN)

    # Балансировка выборки для обучения
    max_num_pixels_for_training = 1e6
    ind1 = np.where(M.flatten() == 1)[0]
    ind0 = np.where(M.flatten() == 0)[0]
    getN = min(min(len(ind0), len(ind1)), int(max_num_pixels_for_training))
    np.random.shuffle(ind1)
    np.random.shuffle(ind0)
    ind = np.concatenate((ind1[:getN], ind0[:getN]))

    # Формирование данных
    x = np.column_stack((NDWI.flatten()[ind], WRI.flatten()[ind], MNDWI.flatten()[ind],
                         NDSI.flatten()[ind], NDTI.flatten()[ind]))
    y = M.flatten()[ind]

    X = pd.concat([X, pd.DataFrame(x)], ignore_index=True)
    Y.extend(y)

# Обучение модели
tree = DecisionTreeClassifier(max_leaf_nodes=10)
tree.fit(X, Y)

# Оценка важности признаков
imp = tree.feature_importances_

# Визуализация важности признаков
plt.bar(range(len(imp)), imp)
plt.title('Predictor Importance Estimates')
plt.ylabel('Estimates')
plt.xticks(range(len(imp)), ['NDWI', 'WRI', 'MNDWI', 'NDSI', 'NDTI'], rotation=0)
plt.grid()

# Прогнозирование
label = tree.predict(X)

# Матрица смежности
confusion_mat = confusion_matrix(Y, label)
print(confusion_mat)
# plt.figure()
# plt.matshow(confusion_mat, cmap='gray')
# plt.title('Confusion Matrix')
# plt.colorbar()
plt.show()

# Сохранение модели (например, с использованием joblib)
import joblib

joblib.dump(tree, 'tree.pkl')