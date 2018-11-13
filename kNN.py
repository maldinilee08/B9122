import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Link to R: https://www.rdocumentation.org/packages/DMwR/versions/0.4.1/topics/kNN

# Read files
dat = pd.read_csv('car_data.csv', usecols=(0, 1, 4, 7, 9, 10, 8))
print(dat.head())
print(len(dat))
print(round(dat.describe()))


# Replace missing data / unaccepted zero value with mean (or other treatments)
zero_not_accepted = dat.columns.values.tolist()
zero_not_accepted.remove('Accept?')
print(zero_not_accepted)

for column in zero_not_accepted:
    dat[column] = dat[column].replace(0, np.NaN)
    mean = int(dat[column].mean(skipna=True))
    dat[column] = dat[column].replace(np.NaN, mean)

# Split data set
X = dat.iloc[:, [0, 1, 2, 3, 5, 6]]
y = dat.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Define the model
classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean') # It remains a question on how to choose k
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Plotting
