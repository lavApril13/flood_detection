import cv2
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from model_UNET import get_model
from tqdm import tqdm
import numpy as np
from typing import List, Optional
from rasterio.windows import Window


def get_tiles_with_overlap(image_width: int, image_height: int,
                           tile_size: int, overlap: int) -> List[Window]:
    """
    Calculate the windows for tiles with specified overlap across the image.

    Parameters:
        image_width (int): The width of the input image in pixels.
        image_height (int): The height of the input image in pixels.
        tile_size (int): The size of each tile (assumes square tiles).
        overlap (int): The number of overlapping pixels between adjacent tiles.

    Returns:
        List[Window]: A list of rasterio Window objects representing each tile.
    """
    step_size = tile_size - overlap
    tiles = []
    for y in range(0, image_height, step_size):
        for x in range(0, image_width, step_size):
            window = Window(x, y, tile_size, tile_size)
            # Adjust window if it exceeds the image bounds
            window = window.intersection(Window(0, 0, image_width, image_height))
            tiles.append(window)
    return tiles

def image_padding(image, target_size=256):
    """
    Pad an image to a target size using reflection padding.
    """
    height, width = image.shape[1:3]
    pad_height = max(0, target_size - height)
    pad_width = max(0, target_size - width)
    padded_image = np.pad(image, ((0, 0), (0, pad_height),
                                  (0, pad_width)), mode='reflect')
    return padded_image

class Runner():
    def __init__(self):
        self.__model0 = torch.load('./skoltech_train/model_8th_all.pt')
        #self.__model1 = torch.load('./skoltech_train/UNet_100_6_2.pt')
        self.__model1 = get_model('./skoltech_train/unet_100_6_2.pth')
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("cuda" if torch.cuda.is_available() else "cpu")
        self.__model0.to(self.__device)
        self.__model1.to(self.__device)

    def run(self, image, model_type):
        image_array = image.read().astype(np.float32)
        image_width = image.width
        image_height = image.height
        new_img = np.zeros((image_height, image_width))
        tiles = get_tiles_with_overlap(image_width, image_height, 256, 32)
        with torch.no_grad():
            self.__model0.eval()
            self.__model1.eval()
            self.__model0.to(self.__device)
            self.__model1.to(self.__device)
            print(image_height, image_width)
            print(new_img.shape)
            for tile in tqdm(tiles):
                # print(tile.col_off, tile.row_off)
                # crop_start_col = tile.row_off
                # crop_start_row = tile.col_off
                # if crop_start_col + tile.width > image_width:
                #     crop_start_col = image_width - tile.width
                #
                # if crop_start_row + tile.height > image_height:
                #     crop_start_row = image_height - tile.height
                # if (tile.col_off + tile.width) < image_width:

                crop = image_array[:, tile.row_off:tile.row_off + tile.height, tile.col_off:tile.col_off + tile.width]
                # if crop.shape[1] < 256 or crop.shape[2] < 256:
                #     crop = pad_to_size(crop, 256, 256)

                crop = image_padding(crop, 256)
                crop = torch.from_numpy(crop).float().unsqueeze(0)

                crop = self.get_new_channels(crop, model_type)
                crop = crop.to(self.__device)

                if model_type==0:
                    out = self.__model0(crop)
                else:
                    out = self.__model1(crop)
                out = torch.sigmoid(out)
                out = torch.round(out)
                out = out[0, 0].cpu().detach().numpy()




                out = out[:tile.height, :tile.width]
                h = tile.height
                w = tile.width
                if tile.row_off + tile.height > image_height - 1:
                    h = image_height - tile.row_off

                if tile.col_off + tile.width > image_width - 1:
                    w = image_width - tile.col_off


                out = out[:h, :w]

                if tile.row_off!=0:
                    out[0:16,:] = 0
                if tile.row_off + tile.height <= image_height - 1:
                    out[h-16:h,:] = 0
                if tile.col_off != 0:
                    out[:,0:16] = 0
                if tile.col_off + tile.width <= image_width - 1:
                    out[:, w-16:w] = 0
                #print(tile.row_off, tile.col_off)
                new_img[tile.row_off:tile.row_off + h, tile.col_off:tile.col_off + w] += out

            new_img[new_img >= 1] = 1

            new_img = new_img.astype(np.uint16)
            return new_img

    def get_new_channels(self, old_channels, model_type):
        old_channels = old_channels + 1
        # Индексы каналов (предполагаем, что известны)
        NIR_idx = 6  # Индекс канала ближнего инфракрасного излучения (NIR)
        SWIR1_idx = 8  # Индекс канала коротковолнового инфракрасного излучения 1 (SWIR1)
        SWIR2_idx = 9  # Индекс канала коротковолнового инфракрасного излучения 2 (SWIR2)
        GREEN_idx = 1  # Индекс зеленого канала
        RED_idx = 2  # Индекс красного канала

        BLUE_idx = 0
        THIRD_idx = 3
        FOURTH_idx = 4
        FIFTH_idx = 5
        SEVENTH_idx = 7

        # Извлекаем соответствующие каналы
        NIR = old_channels[:, NIR_idx, :, :]
        SWIR1 = old_channels[:, SWIR1_idx, :, :]
        SWIR2 = old_channels[:, SWIR2_idx, :, :]
        GREEN = old_channels[:, GREEN_idx, :, :]
        RED = old_channels[:, RED_idx, :, :]
        BLUE = old_channels[:, BLUE_idx, :, :]
        THIRD = old_channels[:, THIRD_idx, :, :]
        FOURTH = old_channels[:, FOURTH_idx, :, :]
        FIFTH = old_channels[:, FIFTH_idx, :, :]
        SEVENTH = old_channels[:, SEVENTH_idx, :, :]

        NIR = NIR.to(torch.float32)
        SWIR1 = SWIR1.to(torch.float32)
        SWIR2 = SWIR2.to(torch.float32)
        GREEN = GREEN.to(torch.float32)
        RED = RED.to(torch.float32)
        BLUE = BLUE.to(torch.float32)
        THIRD = THIRD.to(torch.float32)
        FOURTH = FOURTH.to(torch.float32)
        FIFTH = FIFTH.to(torch.float32)
        SEVENTH = SEVENTH.to(torch.float32)

        # Вычисляем новые каналы по формулам и нормализуем их
        NDWI = (((NIR - SWIR2) / (NIR + SWIR2)) + 1) / 2

        WRI = (GREEN + RED) / (NIR + SWIR2)
        WRI_mask_4 = WRI > 4
        WRI[WRI_mask_4] = 4
        WRI = WRI / 4

        MNDWI = (((GREEN - SWIR2) / (GREEN + SWIR2)) + 1) / 2
        NDSI = (((GREEN - SWIR1) / (GREEN + SWIR1)) + 1) / 2
        NDTI = ((((RED - GREEN) / (RED + GREEN))) + 1) / 2
        # print(NDWI.shape)

        # Собираем новое изображение из новых каналов
        if model_type==0:
            k_inputs = torch.stack([BLUE / 65536.0, GREEN / 65536.0, RED / 65536.0, NDWI, WRI, MNDWI, NDSI, NDTI], dim=1)
        else:
            k_inputs = torch.stack([NDWI, WRI, MNDWI, NDSI, NDTI], dim=1)
        # print(new_inputs.shape)
        return k_inputs

if __name__ == '__main__':
    run = Runner()
    with rasterio.open('skoltech_train/train/images/9_1.tif') as src_image:
        img = run.run(src_image)