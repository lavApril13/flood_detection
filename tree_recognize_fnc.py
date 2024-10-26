import numpy as np
import joblib
import matplotlib.pyplot as plt
import imageio
import rasterio

def process_image_with_model(image, model_path='tree.pkl'):
    # Обнаружение воды при помощи деревьев решений
    # Обрабатывает изображение, рассчитывает индексы и делает предсказание с использованием предобученной модели.
    #:param image: Входное изображение в формате numpy.
    #:param model_path: Путь к предобученной модели.
    #:return: Изображение предсказанной маски.

    image = image.astype(np.float32) / 256

    # Загрузка предобученной модели
    tree = joblib.load(model_path)

    # Извлечение диапазонов
    BLUE, GREEN, RED, B05, B06, B07, NIR, SWIR1, SWIR2 = image[:9]

    # Расчет индексов воды
    NDWI = (NIR - SWIR2) / (NIR + SWIR2)
    WRI = (GREEN + RED) / (NIR + SWIR2)
    MNDWI = (GREEN - SWIR2) / (GREEN + SWIR2)
    NDSI = (GREEN - SWIR1) / (GREEN + SWIR1)
    NDTI = (RED - GREEN) / (RED + GREEN)

    # Подготовка набора признаков
    features = np.column_stack((NDWI.flatten(), WRI.flatten(), MNDWI.flatten(), NDSI.flatten(), NDTI.flatten()))

    # Прогнозирование с использованием модели
    predictions = tree.predict(features)

    # Формируем изображение предсказаний
    prediction_image = predictions.reshape(image.shape[1], image.shape[2])

    # Визуализация
    plt.figure(figsize=(10, 5))
    plt.imshow(prediction_image, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.show()

    return prediction_image

# Пример использования функции
file_path = 'd:/water/train_dataset_skoltech_train/train/images/1.tif'
with rasterio.open(file_path) as src:
    image = src.read().astype(np.float32) / 256  # Нормализация данных
mask = process_image_with_model(image)

