from streamlit.web import cli as stcli
from streamlit import runtime
import streamlit as st
import sys
import io
import rasterio
import folium
import numpy as np
from PIL import Image

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

def plot_mask(mask_path):
    pal = [value for color in PALLETE for value in color]
    with rasterio.open(mask_path) as fin:
        mask = fin.read(1)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(pal)
    st.image(mask, "Затопленные области")
    return mask


def load_data(upload_file):
    with rasterio.open(io.BytesIO(upload_file.getvalue())) as src:
        data = src.read()
        bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
        meta = src.meta.copy()
        return data, bounds, meta


def analyse(data):
    # пока порогом выделим маску
    # тут нейронка должна вернуть маску от 0 до 1
    # --------------------------------------------
    channels, w, h = data.shape
    blue = np.reshape(data[1, :, :], (w, h, 1))
    blue_b = brighten(blue)
    blue_bn = normalize(blue_b)

    blue_bn[blue_bn>0.5] = 1
    blue_bn[blue_bn<=0.5] = 0
    mask = blue_bn
    # ----------------------------------------------
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
        plot_data_rgb(uploaded_file)
        #скачиваем один tif файл целиком, чтобы дальше передать на обработку
        # в data первая размерность это число каналов (channels, m, n)
        data, bounds, meta = load_data(uploaded_file)

        if st.sidebar.button("Выделить области покрытые водой"):
            # считает и возвращает маску с затоплениями от 0 до 1
            mask = analyse(data)

            # просто отрисуем маску
            st.image(mask, "Затопленные области")

            # Отображаем маску поверх карты
            map_object = display_on_map(mask, bounds)
            st.components.v1.html(map_object._repr_html_(), height=500)

            # сохраняем как у них чтобы выгружать на проверку
            save_mask_to_tif(mask, meta, "binary_mask.tif")


if __name__ == '__main__':
    if runtime.exists():
        run_web_app()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())