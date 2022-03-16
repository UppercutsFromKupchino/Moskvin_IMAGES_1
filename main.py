from cv2 import imread, imwrite, imshow, waitKey, IMREAD_GRAYSCALE
from math import sqrt


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

            g_x = 0
            g_y = 0

            for di in range(0, 2):
                for dj in range(0, 2):

                    g_x += image[i + di][j + dj] * mask_x[di][dj]
                    g_y += image[i + di][j + dj] * mask_y[di][dj]

            result_image[i][j] = sqrt(g_x**2 + g_y**2)

    imshow('result', result_image)
    waitKey(0)


if __name__ == '__main__':
    text1 = 'niger.jpg'
    image1 = load_image(text1)
    roberts(image1)
