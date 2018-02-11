from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load in dataset
iris = load_iris()

X = iris.data
y = iris.target

# set some constants
test_size = 0.2
train_size = 0.8
random_state = 0

# create test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
sc = StandardScaler()

sc.fit(X_train)


X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_train_std = X_train_std[:, [2, 3]]
X_test_std = X_test_std[:, [2, 3]]

max_iter = 45
eta0 = 0.1

# run your preceptron on train set
ppn = Perceptron(max_iter=max_iter, eta0=eta0, random_state=random_state)

ppn.fit(X_train_std, y_train)

# predict your test set 
y_pred = ppn.predict(X_test_std)
print("Accuracy", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred))
