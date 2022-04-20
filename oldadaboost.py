import numpy as np
import pandas as pd
import sys

# Decision stump used as weak classifier
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class Adaboost:
    def __init__(self, k):
        self.k = k
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []

        # Iterate through classifiers
        for x in range(self.k):
            clf = DecisionStump()
            min_error = float("inf")

            # greedy search to find best threshold and feature
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # calculate alpha
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # calculate predictions and update weights
            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


# Testing
if __name__ == "__main__":
    # Imports
    #from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold

    dataset = sys.argv[1]

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

## CAR DATA ##
    if dataset == "car":
        df = pd.read_csv("car.data", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "classification"])
        df["classification"].replace(["unacc", "acc", "good", "vgood"], [0, 1, 2, 3], inplace=True)
        df["safety"].replace(["low", "med", "high"], [0, 1, 2], inplace=True)
        df["lug_boot"].replace(["small", "med", "big"], [0, 1, 2], inplace=True)
        df["persons"].replace(["more"], [2], inplace=True)
        df["doors"].replace(["5more"], [5], inplace=True)
        df["maint"].replace(["low", "med", "high", "vhigh"], [1, 2, 3, 4], inplace=True)
        df["buying"].replace(["low", "med", "high", "vhigh"], [1, 2, 3, 4], inplace=True)

        df['doors'] = df['doors'].astype(int)
        df['persons'] = df['persons'].astype(int)

        k = 2
        model = Adaboost(k)
        for i in range(0,10):
            df = df.sample(frac=1)

            x, y = df.iloc[:, 0:6].to_numpy(), df.iloc[:,6].to_numpy()
            y[y == 0] = -1

            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                model.fit(X_train, y_train)
                pred_values = model.predict(X_test)
            #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

    # Adaboost classification with 5 weak classifiers

                acc = accuracy(pred_values, y_test)
            print("Accuracy:", acc)

    elif dataset == "breast_cancer":
# BREAST CANCER DATA ##
        df = pd.read_csv("breast-cancer-wisconsin.data",names=["ID_Number", "Radius", "Texture", "Perimeter", "Area", "smoothness", "compactness","concavity", "concave_points", "symmetry", "fractal_dimension"])
        df.drop(["compactness"],axis=1)
        df["fractal_dimension"].replace([2, 4], [-1, 1], inplace=True)


        k = 2
        model = Adaboost(k)
        for i in range(0,10):
# Shuffle the dataset
            df = df.sample(frac=1)
            #x, y = df.iloc[:,1:8].to_numpy(), df.iloc[:,8].to_numpy()
            x, y = df.iloc[:,1:10].to_numpy(), df.iloc[:,10].to_numpy()   #this is giving 0 accuracy


            y[y == 0] = -1

            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                model.fit(X_train, y_train)
                pred_values = model.predict(X_test)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

    # Adaboost classification with 5 weak classifiers

                acc = accuracy(pred_values, y_test)
            print("Accuracy:", acc)


##########################################
    elif dataset == "letter":
        df = pd.read_csv("./dataset/letter-recognition.data",names=["lettr", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar","xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"])
        df['lettr'] = [ord(item) - 64 for item in df['lettr']]

        # for i in range(0, 10):
    # Shuffle the dataset

        k = 2
        model = Adaboost(k)
        for i in range(0, 10):

            df = df.sample(frac=1)
            x, y = df.iloc[:, 1:17].to_numpy(), df.iloc[:, 0].to_numpy()
            y[y == 0] = -1

            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                model.fit(X_train, y_train)
                pred_values = model.predict(X_test)
            # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

            # Adaboost classification with 5 weak classifiers

                acc = accuracy(pred_values, y_test)
            print("Accuracy:", acc)

##########################################
    elif dataset == "mushroom":
        df = pd.read_csv("mushroom.data",
                         names=["classification", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
                                "gill-attachment",
                                "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
                                "stalk-surface-above-ring", "stalk-surface-below-ring",
                                "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
                                "ring-number", "ring-type", "spore-print-color",
                                "population", "habitat"])
        df["classification"].replace(["e", "p"], [0, 1], inplace=True)
        df["cap-shape"].replace(["b", "c", "x", "f", "k", "s"], [0, 1, 2, 3, 4, 5], inplace=True)
        df["cap-surface"].replace(["f", "g", "y", "s"], [0, 1, 2, 3], inplace=True)
        df["cap-color"].replace(["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                inplace=True)
        df["bruises"].replace(["t", "f"], [0, 1], inplace=True)
        df["odor"].replace(["a", "l", "c", "y", "f", "m", "n", "p", "s"], [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        df["gill-attachment"].replace(["a", "d", "f", "n"], [0, 1, 2, 3], inplace=True)
        df["gill-spacing"].replace(["c", "w", "d"], [0, 1, 2], inplace=True)
        df["gill-size"].replace(["b", "n"], [0, 1], inplace=True)
        df["gill-color"].replace(["k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"],
                                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)
        df["stalk-shape"].replace(["e", "t"], [0, 1], inplace=True)
        df["stalk-root"].replace(["b", "c", "u", "e", "z", "r", "?"], [1, 2, 3, 4, 5, 6, 0], inplace=True)
        df["stalk-surface-above-ring"].replace(["f", "y", "k", "s"], [1, 2, 3, 4], inplace=True)
        df["stalk-surface-below-ring"].replace(["f", "y", "k", "s"], [1, 2, 3, 4], inplace=True)
        df["stalk-color-above-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"],
                                             [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        df["stalk-color-below-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"],
                                             [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        df["veil-type"].replace(["p", "u"], [1, 2], inplace=True)
        df["veil-color"].replace(["n", "o", "w", "y"], [1, 2, 3, 4], inplace=True)
        df["ring-number"].replace(["n", "o", "t"], [1, 2, 3], inplace=True)
        df["ring-type"].replace(["c", "e", "f", "l", "n", "p", "s", "z"], [1, 2, 3, 4, 5, 6, 7, 8], inplace=True)
        df["spore-print-color"].replace(["k", "n", "b", "h", "r", "o", "u", "w", "y"], [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                        inplace=True)
        df["population"].replace(["a", "c", "n", "s", "v", "y"], [1, 2, 3, 4, 5, 6], inplace=True)
        df["habitat"].replace(["g", "l", "m", "p", "u", "w", "d"], [1, 2, 3, 4, 5, 6, 7], inplace=True)


        k = 2
        model = Adaboost(k)
        for i in range(1, 11):

            df = df.sample(frac=1)
            x, y = df.iloc[:, 1:23].to_numpy(), df.iloc[:, 0].to_numpy()
            y[y == 0] = -1

            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                model.fit(X_train, y_train)
                pred_values = model.predict(X_test)
                # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

                # Adaboost classification with 5 weak classifiers

                acc = accuracy(pred_values, y_test)
            print("Accuracy:",i, acc)

print("Entire out")