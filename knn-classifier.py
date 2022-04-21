from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sys


def claculating_euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNNClassifier:
    def __init__(self):
        self.k = 2

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self.claculating_prediction(x) for x in X]
        return np.array(y_pred)

    def claculating_prediction(self, x):
        # computing the distance between selected value and rest points
        element_distance = [claculating_euclidean_distance(
            x, x_train) for x_train in self.X_train]
        k_idices = np.argsort(element_distance)[: self.k]  # sort the distance
        # fetching label of k nearest neighbo
        k_neighbor_labels = [self.y_train[i] for i in k_idices]
        most_common_label = Counter(
            k_neighbor_labels).most_common(1)
        return most_common_label[0][0]


if __name__ == "__main__":

    dataset_name = sys.argv[1]
    accuracy = []
    accuracyknn = []

    if dataset_name == "car":

        filename = './dataset/car.data'

        car_dataset = pd.read_csv("./dataset/car.data", names=[
                                  "buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])

        # Shuffle the dataset
        car_dataset = car_dataset.sample(frac=1)

        car_dataset["decision"].replace(
            ["unacc", "acc", "good", "vgood"], [0, 1, 2, 3], inplace=True)
        car_dataset["safety"].replace(
            ["low", "med", "high"], [0, 1, 2], inplace=True)
        car_dataset["lug_boot"].replace(
            ["small", "med", "big"], [0, 1, 2], inplace=True)
        car_dataset["persons"].replace(["more"], [4], inplace=True)
        car_dataset["doors"].replace(["5more"], [6], inplace=True)
        car_dataset["maint"].replace(["vhigh", "high", "med", "low"], [
            4, 3, 2, 1], inplace=True)
        car_dataset["buying"].replace(["vhigh", "high", "med", "low"], [
            4, 3, 2, 1], inplace=True)

        car_dataset['decision'] = car_dataset['decision'].astype(int)
        car_dataset['safety'] = car_dataset['safety'].astype(int)
        car_dataset['lug_boot'] = car_dataset['lug_boot'].astype(int)
        car_dataset['persons'] = car_dataset['persons'].astype(int)
        car_dataset['doors'] = car_dataset['doors'].astype(int)
        car_dataset['maint'] = car_dataset['maint'].astype(int)
        car_dataset['buying'] = car_dataset['buying'].astype(int)

        for i in range(10):
            car_dataset = car_dataset.sample(frac=1)

            knn_classifier = KNNClassifier()
            X = car_dataset.drop(["decision"], axis=1).to_numpy()
            y = car_dataset.decision.to_numpy()

            kfold_knn = KFold(n_splits=5)

            for train_index, test_index in kfold_knn.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                knn_classifier.fit(X_train, y_train)
                predicted_values = knn_classifier.predict(X_test)

                accuracy.append(accuracy_score(predicted_values, y_test))

            print('Mean Accuracy for Car Dataset: %.3f%%' %
                  (sum(accuracy)/float(len(accuracy))))

            accuracyknn.append(accuracy)

        mean_accuracy = np.sum(accuracyknn)/float(len(accuracyknn))
        std_dev = np.std(accuracyknn)

        print("Standard Deviation for Car Dataset:", std_dev, " \n ")

    elif dataset_name == "breastcancer":

        breastcancer = pd.read_csv("./dataset/breast-cancer-wisconsin.data", names=[
                                   "column1", "column2", "column3", "column4", "column5", "column6", "column7", "column8", "column9", "column10", "decision"])

        # breastcancer.drop(["column7"], axis=1)
        breastcancer["decision"].replace([2, 4], [-1, 1], inplace=True)

        breastcancer = breastcancer.sample(frac=1)

        breastcancer = breastcancer.replace("?", np.NaN)
        breastcancer = breastcancer.dropna()

        breastcancer = breastcancer.drop("column1", axis="columns")

        breastcancer['column2'] = breastcancer['column2'].astype(int)
        breastcancer['column3'] = breastcancer['column3'].astype(int)
        breastcancer['column4'] = breastcancer['column4'].astype(int)
        breastcancer['column5'] = breastcancer['column5'].astype(int)
        breastcancer['column6'] = breastcancer['column6'].astype(int)
        breastcancer['column7'] = breastcancer['column7'].astype(int)
        breastcancer['column8'] = breastcancer['column8'].astype(int)
        breastcancer['column9'] = breastcancer['column9'].astype(int)
        breastcancer['column10'] = breastcancer['column10'].astype(int)
        breastcancer['decision'] = breastcancer['decision'].astype(int)

        for i in range(10):

            breastcancer = breastcancer.sample(frac=1)

            knn_classifier = KNNClassifier()
            X = breastcancer.drop(["decision"], axis=1).to_numpy()
            y = breastcancer.decision.to_numpy()

            kfold_knn = KFold(n_splits=5)

            for train_index, test_index in kfold_knn.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                knn_classifier.fit(X_train, y_train)
                predicted_values = knn_classifier.predict(X_test)

                accuracy.append(accuracy_score(predicted_values, y_test))

            print('Mean Accuracy for Breast Cancer Dataset: %.3f%%' %
                  (sum(accuracy)/float(len(accuracy))))

            accuracyknn.append(accuracy)

        mean_accuracy = np.sum(accuracyknn)/float(len(accuracyknn))
        std_dev = np.std(accuracyknn)

        print("Standard Deviation for Breast Cancer Dataset:", std_dev, " \n ")

    elif dataset_name == "mushroom":

        mushroom = pd.read_csv("./dataset/mushroom.data", names=["decision", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",  "gill-attachment",
                                                                 "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
                                                                 "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color",
                                                                 "population", "habitat"])

        mushroom["decision"].replace(["e", "p"], [0, 1], inplace=True)
        mushroom["cap-shape"].replace(["b", "c", "x", "f",
                                      "k", "s"], [0, 1, 2, 3, 4, 5], inplace=True)
        mushroom["cap-surface"].replace(["f", "g",
                                        "y", "s"], [0, 1, 2, 3], inplace=True)
        mushroom["cap-color"].replace(["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"], [
                                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom["bruises"].replace(["t", "f"], [0, 1], inplace=True)
        mushroom["odor"].replace(["a", "l", "c", "y", "f", "m", "n", "p", "s"], [
                                 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom["gill-attachment"].replace(["a",
                                            "d", "f", "n"], [0, 1, 2, 3], inplace=True)
        mushroom["gill-spacing"].replace(["c",
                                         "w", "d"], [0, 1, 2], inplace=True)
        mushroom["gill-size"].replace(["b", "n"], [0, 1], inplace=True)
        mushroom["gill-color"].replace(["k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"], [
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)
        mushroom["stalk-shape"].replace(["e", "t"], [0, 1], inplace=True)
        mushroom["stalk-root"].replace(["b", "c", "u", "e",
                                       "z", "r", "?"], [1, 2, 3, 4, 5, 6, 0], inplace=True)
        mushroom["stalk-surface-above-ring"].replace(
            ["f", "y", "k", "s"], [1, 2, 3, 4], inplace=True)
        mushroom["stalk-surface-below-ring"].replace(
            ["f", "y", "k", "s"], [1, 2, 3, 4], inplace=True)
        mushroom["stalk-color-above-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"], [
                                                   1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom["stalk-color-below-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"], [
                                                   1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom["veil-type"].replace(["p", "u"], [1, 2], inplace=True)
        mushroom["veil-color"].replace(["n", "o",
                                       "w", "y"], [1, 2, 3, 4], inplace=True)
        mushroom["ring-number"].replace(["n", "o", "t"],
                                        [1, 2, 3], inplace=True)
        mushroom["ring-type"].replace(["c", "e", "f", "l", "n", "p", "s", "z"], [
                                      1, 2, 3, 4, 5, 6, 7, 8], inplace=True)
        mushroom["spore-print-color"].replace(["k", "n", "b", "h", "r", "o", "u", "w", "y"], [
                                              1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom["population"].replace(["a", "c", "n", "s", "v", "y"], [
                                       1, 2, 3, 4, 5, 6], inplace=True)
        mushroom["habitat"].replace(["g", "l", "m", "p", "u", "w", "d"], [
                                    1, 2, 3, 4, 5, 6, 7], inplace=True)

        # Shuffle the dataset
        mushroom = mushroom.sample(frac=1)

        mushroom['decision'] = mushroom['decision'].astype(int)
        mushroom['cap-shape'] = mushroom['cap-shape'].astype(int)
        mushroom['cap-surface'] = mushroom['cap-surface'].astype(int)
        mushroom['cap-color'] = mushroom['cap-color'].astype(int)
        mushroom['bruises'] = mushroom['bruises'].astype(int)
        mushroom['odor'] = mushroom['odor'].astype(int)
        mushroom['gill-attachment'] = mushroom['gill-attachment'].astype(int)
        mushroom['gill-spacing'] = mushroom['gill-spacing'].astype(int)
        mushroom['gill-size'] = mushroom['gill-size'].astype(int)
        mushroom['gill-color'] = mushroom['gill-color'].astype(int)
        mushroom['stalk-shape'] = mushroom['stalk-shape'].astype(int)
        mushroom['stalk-root'] = mushroom['stalk-root'].astype(int)
        mushroom['stalk-surface-above-ring'] = mushroom['stalk-surface-above-ring'].astype(
            int)
        mushroom['stalk-surface-below-ring'] = mushroom['stalk-surface-below-ring'].astype(
            int)
        mushroom['stalk-color-above-ring'] = mushroom['stalk-color-above-ring'].astype(
            int)
        mushroom['stalk-color-below-ring'] = mushroom['stalk-color-below-ring'].astype(
            int)
        mushroom['veil-type'] = mushroom['veil-type'].astype(int)
        mushroom['veil-color'] = mushroom['veil-color'].astype(int)
        mushroom['ring-number'] = mushroom['ring-number'].astype(int)
        mushroom['ring-type'] = mushroom['ring-type'].astype(int)
        mushroom['spore-print-color'] = mushroom['spore-print-color'].astype(
            int)
        mushroom['population'] = mushroom['population'].astype(int)
        mushroom['habitat'] = mushroom['habitat'].astype(int)

        for i in range(10):

            mushroom = mushroom.sample(frac=1)

            knn_classifier = KNNClassifier()
            X = mushroom.drop(["decision"], axis=1).to_numpy()
            y = mushroom.decision.to_numpy()

            kfold_knn = KFold(n_splits=5)

            for train_index, test_index in kfold_knn.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                knn_classifier.fit(X_train, y_train)
                predicted_values = knn_classifier.predict(X_test)

                accuracy.append(accuracy_score(predicted_values, y_test))

            print('Mean Accuracy for Mushroom Dataset: %.3f%%' %
                  (sum(accuracy)/float(len(accuracy))))

        accuracyknn.append(accuracy)

        mean_accuracy = np.sum(accuracyknn)/float(len(accuracyknn))
        std_dev = np.std(accuracyknn)

        print("Standard Deviation for Mushroom Dataset:", std_dev, " \n ")

    elif dataset_name == "ecoli":

        ecoli = pd.read_csv("./dataset/ecoli.data", names=["column1", "column2", "column3", "column4",
                            "column5", "column6", "column7", "column8", "decision"], delim_whitespace=True)

        # Shuffle the dataset
        ecoli = ecoli.sample(frac=1)

        ecoli = ecoli.drop("column1", axis="columns")
        ecoli["decision"].replace(["cp", "im", "imU", "imS", "imL", "om", "omL", "pp"], [
                                  0, 1, 2, 3, 4, 5, 6, 7], inplace=True)

        ecoli['column2'] = ecoli['column2'].astype(float)
        ecoli['column3'] = ecoli['column3'].astype(float)
        ecoli['column4'] = ecoli['column4'].astype(float)
        ecoli['column5'] = ecoli['column5'].astype(float)
        ecoli['column6'] = ecoli['column6'].astype(float)
        ecoli['column7'] = ecoli['column7'].astype(float)
        ecoli['column8'] = ecoli['column8'].astype(float)
        ecoli['decision'] = ecoli['decision'].astype(int)

        for i in range(10):

            ecoli = ecoli.sample(frac=1)

            knn_classifier = KNNClassifier()
            X = ecoli.drop(["decision"], axis=1).to_numpy()
            y = ecoli.decision.to_numpy()

            kfold_knn = KFold(n_splits=5)

            for train_index, test_index in kfold_knn.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                knn_classifier.fit(X_train, y_train)
                predicted_values = knn_classifier.predict(X_test)

                accuracy.append(accuracy_score(predicted_values, y_test))

            print('Mean Accuracy for Ecoli Dataset: ',
                  (sum(accuracy)/float(len(accuracy))), '% \n')

            accuracyknn.append(accuracy)

        mean_accuracy = np.sum(accuracyknn)/float(len(accuracyknn))
        std_dev = np.std(accuracyknn)

        print("Standard Deviation for Ecoli Dataset:", std_dev, " \n ")

    elif dataset_name == "letterrecognition":

        letterrecognition = pd.read_csv("./dataset/letter-recognition.data", names=["decision", "column2", "column3", "column4", "column5", "column6",
                                        "column7", "column8", "column9", "column10", "column11", "column12", "column13", "column14", "column15", "column16", "column17"])

        # Shuffle the dataset
        letterrecognition = letterrecognition.sample(frac=1)

        letterrecognition["decision"].replace(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"], [
                                              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)

        letterrecognition['decision'] = letterrecognition['decision'].astype(
            int)
        letterrecognition['column2'] = letterrecognition['column2'].astype(int)
        letterrecognition['column3'] = letterrecognition['column3'].astype(int)
        letterrecognition['column4'] = letterrecognition['column4'].astype(int)
        letterrecognition['column5'] = letterrecognition['column5'].astype(int)
        letterrecognition['column6'] = letterrecognition['column6'].astype(int)
        letterrecognition['column7'] = letterrecognition['column7'].astype(int)
        letterrecognition['column8'] = letterrecognition['column8'].astype(int)
        letterrecognition['column9'] = letterrecognition['column9'].astype(int)
        letterrecognition['column10'] = letterrecognition['column10'].astype(
            int)
        letterrecognition['column11'] = letterrecognition['column7'].astype(
            int)
        letterrecognition['column12'] = letterrecognition['column8'].astype(
            int)
        letterrecognition['column13'] = letterrecognition['column9'].astype(
            int)
        letterrecognition['column14'] = letterrecognition['column10'].astype(
            int)
        letterrecognition['column15'] = letterrecognition['column8'].astype(
            int)
        letterrecognition['column16'] = letterrecognition['column9'].astype(
            int)
        letterrecognition['column17'] = letterrecognition['column10'].astype(
            int)

        for i in range(10):

            letterrecognition = letterrecognition.sample(frac=1)

            knn_classifier = KNNClassifier()
            X = letterrecognition.drop(["decision"], axis=1).to_numpy()
            y = letterrecognition.decision.to_numpy()

            kfold_knn = KFold(n_splits=5)

            for train_index, test_index in kfold_knn.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                knn_classifier.fit(X_train, y_train)
                predicted_values = knn_classifier.predict(X_test)

                accuracy.append(accuracy_score(predicted_values, y_test))

            print('Mean Accuracy for Letter Recognization Dataset: ',
                  (sum(accuracy)/float(len(accuracy))), '% \n')

            accuracyknn.append(accuracy)

        mean_accuracy = np.sum(accuracyknn)/float(len(accuracyknn))
        std_dev = np.std(accuracyknn)

        print("Standard Deviation for Letter Recignization Dataset:", std_dev, " \n ")

    else:
        print("Give proper dataset")
