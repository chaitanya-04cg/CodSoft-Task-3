from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#dataset
iris = load_iris()

X = iris.data
y = iris.target

#split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#model
model = KNeighborsClassifier(
    n_neighbors=5
)

#train
model.fit(
    X_train,
    y_train
)

# Predict
prediction = model.predict(
    X_test
)

# Accuracy
print(
    "Accuracy:",
    round(
        accuracy_score(y_test,prediction) * 100,2),"%")
