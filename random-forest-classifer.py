from collections import Counter
import numpy as np
import pandas as pd
from DecisionTreeForRandomForest import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import sys


class RandomForestClassifier:
    def __init__(self, num_of_trees=10, split_min=2, maximum_depth=100, num_of_feats=None):
        self.num_of_trees = num_of_trees
        self.split_min = split_min
        self.maximum_depth = maximum_depth
        self.num_of_feats = num_of_feats
        self.total_trees = []

    def fit(self, X_arr, y_arr):
        self.total_trees = []
        for _ in range(self.num_of_trees):
            tree = DecisionTree(
                split_min=self.split_min,
                maximum_depth=self.maximum_depth,
                num_of_feats=self.num_of_feats,
            )
            X_samp, y_samp = tree_sample(X_arr, y_arr)
            tree.fit(X_samp, y_samp)
            self.total_trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.total_trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_label_common(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)


def tree_sample(X, y):
    num_of_samples = X.shape[0]
    indices = np.random.choice(num_of_samples, num_of_samples, replace=True)
    return X[indices], y[indices]


def most_label_common(y):
    counter = Counter(y)
    most_common_label = counter.most_common(1)[0][0]
    return most_common_label


if __name__ == "__main__":

    dataset_name = sys.argv[1]

    accuracyrandomeforest = []
    accuracy = []

    if dataset_name == "mushroom":

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

            X = mushroom.drop(["decision"], axis=1).to_numpy()
            y = mushroom.decision.to_numpy()

            kfold_ran_forest = KFold(n_splits=5)
            for train_index, test_index in kfold_ran_forest.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                random_forest_classifier = RandomForestClassifier(
                    num_of_trees=20, maximum_depth=10)

                random_forest_classifier.fit(X_train, y_train)
                predicted_values = random_forest_classifier.predict(X_test)

                accuracy.append(accuracy_score(predicted_values, y_test))

            print("Accuracy for Mushroom Dataset:",
                  (sum(accuracy)/float(len(accuracy))))

            accuracyrandomeforest.append(accuracy)

        mean_accuracy = np.sum(accuracyrandomeforest) / \
            float(len(accuracyrandomeforest))
        std_dev = np.std(accuracyrandomeforest)

        print("Standard Deviation for Mushroom Dataset:", std_dev, " \n ")

    elif dataset_name == "car":

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

            X = car_dataset.drop(["decision"], axis=1).to_numpy()
            y = car_dataset.decision.to_numpy()

            kfold_ran_forest = KFold(n_splits=5)
            for train_index, test_index in kfold_ran_forest.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                random_forest_classifier = RandomForestClassifier(
                    num_of_trees=20, maximum_depth=10)

                random_forest_classifier.fit(X_train, y_train)
                predicted_values = random_forest_classifier.predict(X_test)

                accuracy.append(accuracy_score(predicted_values, y_test))

            print("Accuracy for Car Dataset:",
                  (sum(accuracy)/float(len(accuracy))))

            accuracyrandomeforest.append(accuracy)

        mean_accuracy = np.sum(accuracyrandomeforest) / \
            float(len(accuracyrandomeforest))
        std_dev = np.std(accuracyrandomeforest)

        print("Standard Deviation for Car Dataset:", std_dev, " \n ")

    elif dataset_name == "breastcancer":

        breastcancer = pd.read_csv("./dataset/breast-cancer-wisconsin.data", names=[
                                   "column1", "column2", "column3", "column4", "column5", "column6", "column7", "column8", "column9", "column10", "decision"])

        # Shuffle the dataset
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

            X = breastcancer.drop(["decision"], axis=1).to_numpy()
            y = breastcancer.decision.to_numpy()

            kfold_ran_forest = KFold(n_splits=5)
            for train_index, test_index in kfold_ran_forest.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                random_forest_classifier = RandomForestClassifier(
                    num_of_trees=20, maximum_depth=10)

                random_forest_classifier.fit(X_train, y_train)
                predicted_values = random_forest_classifier.predict(X_test)

                accuracy.append(accuracy_score(predicted_values, y_test))

            print("Accuracy for Breast Cancer Dataset:",
                  (sum(accuracy)/float(len(accuracy))))

            accuracyrandomeforest.append(accuracy)

        mean_accuracy = np.sum(accuracyrandomeforest) / \
            float(len(accuracyrandomeforest))
        std_dev = np.std(accuracyrandomeforest)

        print("Standard Deviation for Breast Cancer Dataset:", std_dev, " \n ")

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

            X = ecoli.drop(["decision"], axis=1).to_numpy()
            y = ecoli.decision.to_numpy()

            kfold_ran_forest = KFold(n_splits=5)
            for train_index, test_index in kfold_ran_forest.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                random_forest_classifier = RandomForestClassifier(
                    num_of_trees=20, maximum_depth=10)

                random_forest_classifier.fit(X_train, y_train)
                predicted_values = random_forest_classifier.predict(X_test)

                accuracy.append(accuracy_score(predicted_values, y_test))

            print("Accuracy for Ecoli Dataset:",
                  (sum(accuracy)/float(len(accuracy))))

            accuracyrandomeforest.append(accuracy)

        mean_accuracy = np.sum(accuracyrandomeforest) / \
            float(len(accuracyrandomeforest))
        std_dev = np.std(accuracyrandomeforest)

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

            X = letterrecognition.drop(["decision"], axis=1).to_numpy()
            y = letterrecognition.decision.to_numpy()

            kfold_ran_forest = KFold(n_splits=5)
            for train_index, test_index in kfold_ran_forest.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                random_forest_classifier = RandomForestClassifier(
                    num_of_trees=20, maximum_depth=10)

                random_forest_classifier.fit(X_train, y_train)
                predicted_values = random_forest_classifier.predict(X_test)

                accuracy.append(accuracy_score(predicted_values, y_test))

            print("Accuracy for Letter Recognition Dataset:",
                  (sum(accuracy)/float(len(accuracy))))

            accuracyrandomeforest.append(accuracy)

        mean_accuracy = np.sum(accuracyrandomeforest) / \
            float(len(accuracyrandomeforest))
        std_dev = np.std(accuracyrandomeforest)

        print("Standard Deviation for Letter Recognition Dataset:", std_dev, " \n ")

    else:
        print("Give proper dataset")
