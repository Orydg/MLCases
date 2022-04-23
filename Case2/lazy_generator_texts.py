"""
Для обучений моделей нужны данные
Точность обучения зависит от качества и количества данных
Большой объем дынных занимает много памяти (диска или ОЗУ)
Ленивый генератор должен обеспечить модель данными в необходимом количестве, качестве и без затрат памяти
При необходимости, можно будет сохранять сгененированные данные на диск

"""

from typing import Tuple, Iterable, Any
import random


class LazyGeneratorTexts:
    """

    Класс ленивого генератора.
    При инициализации объекта класса, необходимо указать размер батча.
    Формат батча: (высота картинки, ширина картинки, количество картинок)

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

    def generator(self) -> Iterable[Any]:
        while True:
            x_batch = []
            y_batch = []
            yield x_batch, y_batch


if __name__ == '__main__':
    # Tests
    test = LazyGeneratorTexts((5, 5, 1))
    for i in test.generator():
        print(i)
        break

