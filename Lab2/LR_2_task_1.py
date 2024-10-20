import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Вхідний файл, який містить дані
input_file = 'income_data.txt'
# Читання даних
X = []
y = []

count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.dtype.kind in 'iufc':  # Numeric columns
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        label_encoder.append(le)
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Створення SVM-класифікатора
# classifier = svm.SVC(kernel='rbf', gamma=0.1, C=10.0)
classifier = OneVsOneClassifier(LinearSVC(random_state=0))


# Навчання класифікатора
classifier.fit(X, y)

# Розділення на навчальний та тестовий набір
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, andom_state=5)

# classifier = svm.SVC(kernel='rbf')

classifier.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_test_pred = classifier.predict(X_test)

# Обчислення F-міри для SVM-класифікатора
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

# Передбачення результату для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White',
              'Male',
              '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
    count += 1

input_data_encoded = np.array([input_data_encoded])

# Використання класифікатора для кодованої точки даних
# та виведення результату
predicted_class = classifier.predict(input_data_encoded)
print(f"Класифікація кодованої точки: {label_encoder[-1].inverse_transform(predicted_class)[0]}\n");

# Обчислення accuracy_score
accuracy_s = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy_s * 100:.2f}%")

# Обчислення повноти
recall = recall_score(y_test, y_test_pred)
print(f"Recall RF: {recall * 100:.2f}%")
# Обчислення точності
precision = precision_score(y_test, y_test_pred)
print(f"Precision RF: {precision * 100:.2f}%")
