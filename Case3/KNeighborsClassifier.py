from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Нужны данные для обучения
# У sklearn есть небольшой архив данных для тренировок
# Выберем данные, подходящие для задачи классификации при обучении с учителем
# Классификация опухолей
data = datasets.load_breast_cancer()

# выделим из датасета данные для предсказаний и целевые
x = data.data
y = data.target

# разделим данные с помощью Scikit-Learn's train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=75,
                                                    train_size=0.5)

# классификация - поиск соседей
model = KNeighborsClassifier(n_neighbors=4)
model.fit(x_train, y_train)
print(str(round(model.score(x_test, y_test), 2)) + "%")
