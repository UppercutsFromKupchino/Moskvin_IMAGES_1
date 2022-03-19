import math

from cv2 import imread, imwrite, imshow, waitKey, IMREAD_GRAYSCALE
import numpy as np
from math import sqrt, log, exp
from matplotlib.pyplot import hist


# Составление гистограммы изображения
# def get_hist(image):


# Загрузка изображения в память
def load_image(text):
    image = imread(f'{text}', IMREAD_GRAYSCALE)
    return image


# Обработка изображения фильтром Робертса
def roberts(image):
    mask_x = ((1, 0), (0, -1))
    mask_y = ((0, 1), (-1, 0))

    result_image = image.copy()

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):

            mean_x = 0
            mean_y = 0

            for di in range(0, 2):
                for dj in range(0, 2):
                    mean_x += image[i + di][j + dj] * mask_x[di][dj]
                    mean_y += image[i + di][j + dj] * mask_y[di][dj]

            result_image[i][j] = sqrt(mean_x ** 2 + mean_y ** 2)

    imshow('result', result_image)
    waitKey(0)


# Обработка изображения фильтром Собеля
def sobel(image):
    mask_x = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
    mask_y = ((-1, -2, -1), (0, 0, 0), (1, 2, 1))

    result_image = image.copy()

    for i in range(0, image.shape[0] - 2):
        for j in range(0, image.shape[1] - 2):

            mean_x = 0
            mean_y = 0

            for di in range(-1, 2):
                for dj in range(-1, 2):
                    mean_x += image[i + di][j + dj] * mask_x[1 + di][1 + dj]
                    mean_y += image[i + di][j + dj] * mask_y[1 + di][1 + dj]

            result_image[i][j] = sqrt(mean_x ** 2 + mean_y ** 2)

    imshow('result-sobel', result_image)
    waitKey(0)


# Обработка изображения фильтром Превитта
def previtt(image):
    mask_x = ((-1, 0, 1), (-1, 0, 1), (-1, 0, 1))
    mask_y = ((-1, -1, -1), (0, 0, 0), (1, 1, 1))

    result_image = image.copy()

    for i in range(1, image.shape[0] - 2):
        for j in range(1, image.shape[1] - 1):

            mean_x = 0
            mean_y = 0

            for di in range(-1, 2):
                for dj in range(-1, 2):
                    mean_x += image[i + di][j + dj] * mask_x[1 + di][1 + dj]
                    mean_y += image[i + di][j + dj] * mask_y[1 + di][1 + dj]

            result_image[i][j] = sqrt(mean_x ** 2 + mean_y ** 2)

    imshow('result-previtt', result_image)
    waitKey(0)


# Обработка изображений фильтра Кирша
def kirsch(image):
    mask = [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]

    z_list = []

    result_image = image.copy()

    for i in range(1, image.shape[0] - 2):
        for j in range(1, image.shape[1] - 2):

            g_x = 0

            for k in range(0, 9):

                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        g_x += image[i + di][j + dj] * mask[1 + di][1 + dj]

                temp = mask[0][0]
                mask[0][0] = mask[1][0]
                mask[1][0] = mask[2][0]
                mask[2][0] = mask[2][1]
                mask[2][1] = mask[2][2]
                mask[2][2] = mask[1][2]
                mask[1][2] = mask[0][2]
                mask[0][2] = mask[0][1]
                mask[0][1] = temp

                z_list.append(abs(g_x))

            result_image[i][j] = max(z_list)

    imshow('result-kirsch', result_image)
    waitKey(0)


# Обработка изображения фильтром Уоллеса
def wallace(image):
    result_image = image.copy()
    zalupa_image = image.copy()

    # Перед обработкой все яркости +1
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i][j] != 255:
                image[i][j] += 1

    # Первый проход
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # x = int(image[i - 1][j])
            # y = int(image[i][j - 1])
            # z = int(image[i + 1][j])
            # v = int(image[i][j + 1])
            # b = int(image[i][j])
            # bla = b ** 4 / (x * y * z * v)
            zalupa_image[i][j] = abs(float(
                log(int(image[i][j]) ** 4 / (int(image[i - 1][j]) * int(image[i][j - 1]) * int(image[i + 1][j]) *
                                             int(image[i][j + 1])), exp(1))))

    # Вычисление гистограммы
    mas = zalupa_image.ravel()
    max_h = max(mas)
    min_h = min(mas)

    coefficient = 255 / (max_h - min_h)

    for i in range(1, image.shape[0] - 2):
        for j in range(1, image.shape[1] - 2):

            result_image[i][j] = int(30 * coefficient * (zalupa_image[i][j] - min_h))

    imshow('result-wallace', result_image)
    waitKey(0)


if __name__ == '__main__':
    text1 = 'niger.jpg'
    image1 = load_image(text1)
    # roberts(image1)
    # sobel(image1)
    # previtt(image1)
    # kirsch(image1)
    wallace(image1)
