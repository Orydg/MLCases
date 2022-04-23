"""
Для обучений моделей нужны данные
Точность обучения зависит от качества и количества данных
Большой объем дынных занимает много памяти (диска или ОЗУ)
Ленивый генератор должен обеспечить модель данными в необходимом количестве, качестве и без затрат памяти
При необходимости, можно будет сохранять сгененированные данные на диск

"""

from typing import Tuple, Iterable, Any
import numpy as np
import random


class LazyGeneratorTexts:
    """

    Класс ленивого генератора.
    При инициализации объекта класса, необходимо указать размер батча.
    Формат батча: (количество картинок, высота картинки, ширина картинки)

    """

    def __init__(self, batch_size: Tuple[int, int, int], lang: str = "RU"):
        """

        :param batch_size: Размер входного батча модели.
        :param lang: Флаг генерируемого языка.
        """

        self.__BatchSize = batch_size
        self.__Lang = lang
        self.__Token = '!@#$%*-+/=()[]{}~“”№<>,.;:?'
        if self.__Lang == "RU":
            self.__Alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'  # RU
        else:
            self.__Alphabet = ''
        self.__ResToken = self.__Token + self.__Alphabet

    @staticmethod
    def rand_color() -> str:
        rand = lambda: random.randint(0, 30)
        return '#%02X%02X%02X' % (rand(), rand(), rand())

    @staticmethod
    def imageOHE(image, class_num, black_color=200):
        pic = np.array(image)
        img = np.zeros((pic.shape[0], pic.shape[1], class_num))
        np.place(img[:, :, 0], pic[:, :, 0] < black_color, 0)
        np.place(img[:, :, 0], pic[:, :, 0] >= black_color, 1)
        return img

    def generator(self) -> Iterable[Any]:
        while True:
            x_batch = []
            y_batch = []
            for i in range(self.__BatchSize[0]):
                # запись данных в батч
                x_batch += [1]
                y_batch += [1]
            yield x_batch, y_batch


if __name__ == '__main__':
    # Tests
    test = LazyGeneratorTexts((2, 5, 5)).generator()
    for i in test:
        print(i)
        break
