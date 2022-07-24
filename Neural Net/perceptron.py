from sklearn.model_selection import train_test_split
import numpy as np

images = np.load("images/images.npy")
labels = np.load("images/labels.npy")

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.20, random_state=42)


X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
X_test_flatten = X_test.reshape(X_test.shape[0], -1).T
y_train = y_train.reshape(1, y_train.shape[0])
y_test = y_test.reshape(1, y_test.shape[0])

train_set_x = X_train_flatten / 255
test_set_x = X_test_flatten / 255


def initialize_with_zeros(dim):
    w = np.zeros([dim, 1], dtype="float64")
    b = 0.0
    return w, b


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(np.sum(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))) / m

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def gradient_decent(w, b, X, Y, num_iterations=100, learning_rate=0.009):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    w = w.reshape(X.shape[0], 1)
    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0 and A[0, i] < 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = gradient_decent(
        w, b, X_train, Y_train, num_iterations, learning_rate)
    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


logistic_regression_model = model(
    train_set_x, y_train, test_set_x, y_test, num_iterations=2000, learning_rate=0.001)


# wow, 59% accuracy ... better than random guessing at least, but not very powerful. :)
