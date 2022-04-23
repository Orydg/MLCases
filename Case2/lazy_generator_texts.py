"""
Для обучений моделей нужны данные
Точность обучения зависит от качества и количества данных
Большой объем дынных занимает много памяти (диска или ОЗУ)
Ленивый генератор должен обеспечить модель данными в необходимом количестве, качестве и без затрат памяти
При необходимости, можно будет сохранять сгененированные данные на диск

"""

from typing import Tuple, Iterable, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
from random import randint


ROOT_TO_FONTS = ''
OFTEN_FONTS = ['arial.ttf', 'ARIALN.TTF', 'BOOKOS.TTF', 'cour.ttf', 'georgia.ttf',
               'GARA.TTF', 'ariblk.ttf', 'comic.ttf', 'GOST_A.TTF', 'times.ttf',
               'verdana.ttf', 'lucon.ttf', 'impact.ttf']


class LazyGeneratorTexts:
    """

    Класс ленивого генератора.
    При инициализации объекта класса, необходимо указать размер батча.
    Формат батча: (количество картинок, высота картинки, ширина картинки)

    """

    def __init__(self, batch_size: int, img_size: Tuple[int, int], lang: str = "RU"):
        """

        :param batch_size: Размер входного батча модели.
        :param img_size: Размер коченчного изображения (ширина, высота).
        :param lang: Флаг генерируемого языка.
        """

        self.__BatchSize = batch_size
        self.__ImgSize = img_size
        self.__Lang = lang
        self.__Token = '!@#$%*-+/=()[]{}~“”№<>,.;:?'
        if self.__Lang == "RU":
            self.__Alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'  # RU
        else:
            self.__Alphabet = ''
        self.__ResToken = self.__Token + self.__Alphabet
        self.__ImgTrain = None
        self.__ImgRes = None

    @staticmethod
    def rand_color(start: int, end: int) -> Tuple[int, int, int]:
        rand = lambda: randint(start, end)
        return rand(), rand(), rand()

    def ImageToMask(self, class_num=1, black_color=200):
        pic = np.array(self.__ImgRes)
        self.__ImgRes = np.zeros((pic.shape[0], pic.shape[1], class_num))
        np.place(self.__ImgRes[:, :, 0], pic[:, :, 0] < black_color, 0)
        np.place(self.__ImgRes[:, :, 0], pic[:, :, 0] >= black_color, 1)

    def rand_font(self):
        # фиксированные случайные координаты для записи текста в обоих файлах
        w = randint(1, int(self.__ImgSize[0] / 10))
        h = randint(25, int(self.__ImgSize[1] / 2))

        # случайный текст картинки
        for n in range(randint(5, 10)):  # случайное количество строк текста

            # случайный текст строки
            # k = количество символов, формат str
            rand_string = "".join(random.choices(self.__ResToken, k=randint(10, 25)))

            # выбор одного случайного шрифта
            rand_font_types = "".join(random.choices(OFTEN_FONTS, k=1))  # type 'str'

            # случайный размер шрифта
            min_s = 30
            max_s = 40
            font_size = randint(min_s, max_s)

            # параметры шрифта - стиль, размер
            font = ImageFont.truetype(ROOT_TO_FONTS + 'fonts/' + rand_font_types, font_size)

            # записываем в начало координат Х У (случайные) случайную строку (случайная длина строки) случайным цветом,
            # и случайным размером шрифта
            ImageDraw.Draw(self.__ImgTrain).text((w, h + (max_s + 5) * n), rand_string,
                                                 fill=self.rand_color(0, 30), font=font)

            # записываем те же данные но в черную картинку (цвет надписи красный)
            ImageDraw.Draw(self.__ImgRes).text((w, h + (max_s + 5) * n), rand_string, fill=(255, 255, 255), font=font)

    def rand_back(self):

        w, h = self.__ImgSize

        # случайный фон картинки
        for _ in range(randint(10, 25)):  # случайно количество объектов

            # случайный прямоугольник
            coord = (randint(0, w), randint(0, h), randint(0, w), randint(0, h))
            ImageDraw.Draw(self.__ImgTrain).rectangle(coord, fill=None,
                                                      outline=self.rand_color(35, 255),
                                                      width=randint(10, 60))

    def get_img(self):
        # генерируем две одинаковые по размерам картинки
        w, h = self.__ImgSize

        # генерация картинки случайного цвета
        self.__ImgTrain = Image.new("RGB", (w, h), self.rand_color(35, 255))

        # генерация черной картинки
        self.__ImgRes = Image.new("RGB", (w, h))

        # случайный фон картинки
        self.rand_back()

        # случайный шрифт
        self.rand_font()

        # Разворот картинки на случайный угол
        angle = randint(-5, 5)
        self.__ImgTrain = self.__ImgTrain.rotate(angle, resample=Image.BICUBIC, expand=False)
        self.__ImgRes = self.__ImgRes.rotate(angle, resample=Image.BICUBIC, expand=False)

        # переформатируем в np.array
        self.__ImgTrain, self.__ImgRes = np.asarray(self.__ImgTrain), np.asarray(self.__ImgRes)

    def generator(self) -> Iterable[Any]:
        while True:

            x_batch = []
            y_batch = []

            for _ in range(self.__BatchSize):

                # получение картинки
                self.get_img()

                # фильтр, отсеивает два пустых слоя данных у RGB картинки, оставшееся переводит в [1, 0]
                self.ImageToMask()

                # запись данных в батч
                x_batch += [self.__ImgTrain]
                y_batch += [self.__ImgRes]

            # нормализация батча
            x_batch = np.array(x_batch) / 255.
            y_batch = np.array(y_batch)

            yield x_batch, y_batch


if __name__ == '__main__':

    # Tests
    import matplotlib.pyplot as plt

    # перевод маски в изображение
    def predToGrayImage(segment):
        img = np.ones((segment.shape[0], segment.shape[1], 3))
        img[:, :, 0] = segment[:, :, 0] * 255
        img[:, :, 1] = img[:, :, 0]
        img[:, :, 2] = img[:, :, 0]
        return img.astype('uint8')

    test = LazyGeneratorTexts(2, (500, 500)).generator()
    for x, y in test:
        break

    y = predToGrayImage(y[0])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 25))
    axes[0].imshow(x[0])
    axes[1].imshow(y)

    plt.show()
