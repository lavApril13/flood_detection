from streamlit.web import cli as stcli
from streamlit import runtime
import streamlit as st
import sys
import io
import rasterio
import folium
import numpy as np
import tempfile
import cv2
from PIL import Image
import tree_recognize_fnc
import run_model
from skimage import morphology
from scipy import ndimage

PALLETE = [ [0, 0, 0], [0, 0, 255]]

def display_on_map(data, bounds):
    """Отображает GeoTIFF на карте."""
    # Нормализация данных
    data_normalized = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    # Создание карты
    m = folium.Map(location=[(bounds[3] + bounds[1]) / 2, (bounds[0] + bounds[2]) / 2], zoom_start=12)

    # Добавление растровых данных на карту
    map_data = np.array((data_normalized * 255), dtype=np.uint8)

    folium.raster_layers.ImageOverlay(image=map_data,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        opacity=0.5, interactive=True).add_to(m)
    return m


def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band - band_min) / ((band_max - band_min)))


def brighten(band):
    alpha = 0.13
    beta = 0
    return np.clip(alpha * band + beta, 0, 255)


def convert(im_path):
    with rasterio.open(im_path) as fin:
        red = fin.read(3)
        green = fin.read(2)
        blue = fin.read(1)

    red_b = brighten(red)
    blue_b = brighten(blue)
    green_b = brighten(green)

    red_bn = normalize(red_b)
    green_bn = normalize(green_b)
    blue_bn = normalize(blue_b)
    return np.dstack((blue_b, green_b, red_b)), np.dstack((red_bn, green_bn, blue_bn))

def plot_data_rgb(image_path):
    _, img = convert(image_path)
    st.image(img, "Фото в оптическом диапазоне")
    return img


def load_data(upload_file):
    with rasterio.open(io.BytesIO(upload_file.getvalue())) as src:
        data = src.read()
        bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
        meta = src.meta.copy()
        return data, bounds, meta, src


def analyse(data, src, method):
    if method==0:
        mask = tree_recognize_fnc.process_image_with_model(data)
    elif method==1:
        run = run_model.Runner()
        mask = run.run(data, 0)
    elif method==2:
        run = run_model.Runner()
        mask = run.run(data, 1)
    return mask


def save_mask_to_tif(mask, meta, name):
    # Сохранение маски в TIFF
    meta['count'] = 1
    with rasterio.open(name, 'w', **meta) as dst:
        w, h, channels = mask.shape
        if channels is not None:
            dst.write(mask[:, :, 0], 1)
        else:
            dst.write(mask, 1)


###### main function #########
def run_web_app():
    st.set_page_config(layout='wide', page_title='inNino')
    st.write(""" ### дистанционный мониторинг уровня затопления""")
    uploaded_file = st.sidebar.file_uploader("Выберите изображение для анализа", type=["tif"])
    if uploaded_file is not None:
        st.write("You selected the file:", uploaded_file.name)
        print('file ' + uploaded_file.name + ' is open')
        plot_data_rgb(uploaded_file)
        #скачиваем один tif файл целиком, чтобы дальше передать на обработку
        # в data первая размерность это число каналов (channels, m, n)
        data, bounds, meta, src = load_data(uploaded_file)

        flag_press_button = 0
        if st.sidebar.button("  Tree"):
            # считает и возвращает маску с затоплениями от 0 до 1
            mask = analyse(data, src, 0)
            flag_press_button = 1
            print('mask tree generated')
        if st.sidebar.button("UNet++"):
            mask = analyse(data, src, 1)
            print('mask Unet++ generated')
            flag_press_button = 1
        if st.sidebar.button("  UNet"):
            mask = analyse(data, src, 2)
            print('mask Unet generated')
            flag_press_button = 1
        if st.sidebar.button("   All"):
            print('mask Unet generated')
            mask1 = analyse(data, src, 0)
            mask2 = analyse(data, src, 1)
            mask3 = analyse(data, src, 2)

            # Голосование. '2из3'
            #mask = ((mask1+mask2+mask3)>2).astype(np.uint16)

            # Суммирование обнаруженных объектов нейросетями. И удаление заведомо НЕводы обнаруженной деревом
            mask = (mask2+mask3)
            mask[mask1==0] = 0
            mask[mask>0] = 1

            flag_press_button = 1

        if flag_press_button > 0:
            flag_press_button = 0
            # # Создаем структурный элемент
            # selem = morphology.disk(1)
            # # Применяем эрозию и дилатацию
            # mask = morphology.erosion(mask, selem)

            # mask = np.uint8(mask)*255
            # ret, thresh = cv2.threshold(mask, 127, 255, 0)
            # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # mask = cv2.drawContours(mask, contours, -1, (0, 255, 0), 1)
            # mask = mask.astype(np.uint16)/255

            mask = np.uint8(mask) * 255
            # Инвертируем изображение (черные области станут белыми, а белые - черными)
            inverted_image = np.invert(mask)

            # Находим все объекты (белые области) на инвертированном изображении
            #labeled_objects, num_objects = ndimage.label(inverted_image)
            max_hole_size = 1000
            # Заполняем небольшие дыры в объектах
            binary_image = inverted_image.copy()
            _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

            # Создаем маску для сохранения больших объектов
            mask = np.zeros_like(binary_image, dtype=np.uint8)
            min_object_size = 1000
            # Итерируемся по объектам и сохраняем только те, площадь которых больше или равна min_object_size
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_object_size:
                    component_mask = (labels == i).astype(np.uint8)
                    mask = cv2.bitwise_or(mask, component_mask)
            st.write(mask.max())
            # Инвертируем обратно заполненное изображение
            mask = np.invert(mask)
            mask = mask.astype(np.uint16)/255
            mask[mask!=1] = 0
            mask = mask.astype(np.uint16)
            
            # просто отрисуем маску
            st.image(mask.astype(np.uint8)*255, "Затопленные области")

            # Отображаем маску поверх карты
            map_object = display_on_map(mask, bounds)
            st.components.v1.html(map_object._repr_html_(), height=500)

            st.write('Площадь водной поверхности ' + str(np.sum(mask)*10/1000/100) + ' кв. км')
            # сохраняем как локально чтобы выгружать на проверку
            #save_mask_to_tif(mask, meta, "binary_mask.tif")
            # Сохранение бинарного изображения во временный файл

            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
                temp_filename = temp_file.name
                cv2.imwrite(temp_filename, mask)

            # Кнопка для скачивания результата
            st.download_button("Скачать бинарное изображение", data=open(temp_filename, "rb").read(), file_name=uploaded_file.name)
            print('image was sawed')


if __name__ == '__main__':
    if runtime.exists():
        run_web_app()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
