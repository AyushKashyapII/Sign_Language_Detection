import pickle
from collections import Counter, defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

lengths = [len(sample) for sample in data_dict['data']]
most_common_length = Counter(lengths).most_common(1)[0][0]


filtered_data = []
filtered_labels = []
for sample, label in zip(data_dict['data'], data_dict['labels']):
    if len(sample) == most_common_length:
        filtered_data.append(sample)
        filtered_labels.append(label)


class_counts = defaultdict(int)
for label in filtered_labels:
    class_counts[label] += 1

final_data = []
final_labels = []
for sample, label in zip(filtered_data, filtered_labels):
    if class_counts[label] >= 2:
        final_data.append(sample)
        final_labels.append(label)

data = np.asarray(final_data)
labels = np.asarray(final_labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
