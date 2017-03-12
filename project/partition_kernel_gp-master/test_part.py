from sklearn.cross_validation import train_test_split
from part_kernel import FastKernel
from numpy import linspace, sin, argsort
import matplotlib.pyplot as plt

data = linspace(0, 10, 500)
labels = sin(data)
train_data, test_data, train_labels, test_labels = train_test_split(data,
                                                                    labels,
                                                                    test_size=0.1)
model = FastKernel(train_data, train_labels)
model._select_centers(data)
p = model.predict_mean(test_data, train_data, train_labels)

plt.plot(data, labels)
xs = argsort(test_data[:20])
plt.scatter(test_data[:20][xs], p[xs])
plt.show()
