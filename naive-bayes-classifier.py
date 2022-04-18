
from random import randrange
from math import sqrt
from math import exp
from math import pi
import pandas as pd
import numpy as np
import sys

# This method is used to evaluate a NB algorithm using a cross validation


def fit(dataset, nb, decision_column, *args):
    folded_dataset = cross_validation(dataset)
    accuracy_list = list()
    for index, unfold_dataset in enumerate(folded_dataset):
        X_train = np.array(folded_dataset)
        X_train = np.delete(X_train, index, axis=0)
        X_train = np.concatenate((X_train))
        X_test = list()
        for record in unfold_dataset:
            row_copy = list(record)
            X_test.append(row_copy)
            # Removing the decision node by making it as None
            row_copy[decision_column] = None
        predicted_for_each_row = nb(X_train, X_test, decision_column, *args)
        y_test = [record[decision_column] for record in unfold_dataset]
        accuracy = accuracy_precentage(y_test, predicted_for_each_row)
        accuracy_list.append(accuracy)
    return accuracy_list

# Naive Bayes Algorithm


def naive_bayes_classifier(X_train, X_test, decision_column):
    sorted_dataset = dataset_by_class_and_mean_std(X_train, decision_column)
    predictions = list()
    for record in X_test:
        resulted_probabilities = calculate_target_class_probabilities(
            sorted_dataset, record)
        best_target_class, highest_probability = None, -1
        for target_value, probability in resulted_probabilities.items():
            if best_target_class is None or probability > highest_probability:
                highest_probability = probability
                best_target_class = target_value
        predictions.append(best_target_class)
    return predictions

# This method helps to split a dataset into 5 folds.


def cross_validation(dataset):
    num_of_folds = 5
    dataset_after_cross_validation = list()
    temp_dataset = list(dataset)
    each_dataset_size = int(len(dataset) / num_of_folds)
    for f in range(num_of_folds):
        final_folded_dataset = list()
        while len(final_folded_dataset) < each_dataset_size:
            index = randrange(len(temp_dataset))
            final_folded_dataset.append(temp_dataset.pop(index))
        dataset_after_cross_validation.append(final_folded_dataset)
    return dataset_after_cross_validation


def dataset_by_class_and_mean_std(dataset, decision_column):
    # Seperating the dataset based on the class or decision and retuning the updated dataset seperated by class
    dataset_class = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        target_value = vector[decision_column]
        if (target_value not in dataset_class):
            dataset_class[target_value] = list()
        dataset_class[target_value].append(vector)
    # separated = separate_by_class(dataset, decision_column)
    result = dict()
    for target_value, record in dataset_class.items():
        # Calculate the mean, stdev and count for each column in a dataset
        calcuated_value = [(mean(column), stdev(column), len(column))
                           for column in zip(*record)]
        del(calcuated_value[decision_column])
        result[target_value] = calcuated_value
    return result

# This methos will calculate the probabilities of prediction for target class


def calculate_target_class_probabilities(sorted_dataset, row):
    total_rows = sum([sorted_dataset[label][0][2] for label in sorted_dataset])
    resulted_probabilities = dict()
    for target_value, class_sorted_dataset in sorted_dataset.items():
        resulted_probabilities[target_value] = sorted_dataset[target_value][0][2] / \
            float(total_rows)
        for i in range(len(class_sorted_dataset)):
            mean, stdev, _ = class_sorted_dataset[i]
            # Predicton of train dataset is achived by calculating the Gaussian probability distribution function for a given class
            if stdev == 0.0:
                stdev = 1
            exponent = exp(-((row[i]-mean)**2 / (2 * stdev**2)))
            calculated_probability = (1 / (sqrt(2 * pi) * stdev)) * exponent
            resulted_probabilities[target_value] *= calculated_probability
    return resulted_probabilities

# Calculating accuracy percentage


def accuracy_precentage(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Calculate the standard deviation of a list of numbers


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)


# Main function
if __name__ == "__main__":

    dataset_name = sys.argv[1]

    if dataset_name == "car":
        filename = './dataset/car.data'

        cat_dataset = pd.read_csv("./dataset/car.data", names=[
                                  "buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])

        # Shuffle the dataset
        cat_dataset = cat_dataset.sample(frac=1)

        cat_dataset["decision"].replace(
            ["unacc", "acc", "good", "vgood"], [0, 1, 2, 3], inplace=True)
        cat_dataset["safety"].replace(
            ["low", "med", "high"], [0, 1, 2], inplace=True)
        cat_dataset["lug_boot"].replace(
            ["small", "med", "big"], [0, 1, 2], inplace=True)
        cat_dataset["persons"].replace(["more"], [4], inplace=True)
        cat_dataset["doors"].replace(["5more"], [6], inplace=True)
        cat_dataset["maint"].replace(["vhigh", "high", "med", "low"], [
            4, 3, 2, 1], inplace=True)
        cat_dataset["buying"].replace(["vhigh", "high", "med", "low"], [
            4, 3, 2, 1], inplace=True)

        cat_dataset['decision'] = cat_dataset['decision'].astype(int)
        cat_dataset['safety'] = cat_dataset['safety'].astype(int)
        cat_dataset['lug_boot'] = cat_dataset['lug_boot'].astype(int)
        cat_dataset['persons'] = cat_dataset['persons'].astype(int)
        cat_dataset['doors'] = cat_dataset['doors'].astype(int)
        cat_dataset['maint'] = cat_dataset['maint'].astype(int)
        cat_dataset['buying'] = cat_dataset['buying'].astype(int)

        # Naive Bayes Classifier
        for i in range(10):
            decision_column = 6  # Last column is treated as decision column
            accuracy = fit(cat_dataset.values,
                           naive_bayes_classifier, decision_column)
            print('Accuracy for Car Dataset: %s' % accuracy)
            print('Mean Accuracy for Car Dataset: %.3f%%' %
                  (sum(accuracy)/float(len(accuracy))))

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

        # Naive Bayes Classifier
        for i in range(10):
            decision_column = 9  # Last column is treated as decision column
            accuracy = fit(breastcancer.values,
                           naive_bayes_classifier, decision_column)
            print('Accuracy for Breast Dataset Cancer: %s' % accuracy)
            print('Mean Accuracy for Breast Dataset Cancer: %.3f%%' %
                  (sum(accuracy)/float(len(accuracy))))

    elif dataset_name == "mushroom":

        mushroom = pd.read_csv("./dataset/mushroom.data", names=["decision", "column2", "column3", "column4", "column5", "column6", "column7", "column8", "column9", "column10",
                               "column11", "column12", "column13", "column14", "column15", "column16", "column17", "column18", "column19", "column20", "column21", "column22", "column23"])

        mushroom["decision"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column2"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column3"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column4"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column5"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column6"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column7"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column8"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column9"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column10"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column11"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column12"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column13"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column14"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column15"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column16"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column17"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column18"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column19"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column20"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column21"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column22"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)
        mushroom["column23"].replace(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], [
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)

        # Shuffle the dataset
        mushroom = mushroom.sample(frac=1)

        mushroom = mushroom.replace("?", np.NaN)
        mushroom = mushroom.dropna()

        mushroom['decision'] = mushroom['decision'].astype(int)
        mushroom['column2'] = mushroom['column2'].astype(int)
        mushroom['column3'] = mushroom['column3'].astype(int)
        mushroom['column4'] = mushroom['column4'].astype(int)
        mushroom['column5'] = mushroom['column5'].astype(int)
        mushroom['column6'] = mushroom['column6'].astype(int)
        mushroom['column7'] = mushroom['column7'].astype(int)
        mushroom['column8'] = mushroom['column8'].astype(int)
        mushroom['column9'] = mushroom['column9'].astype(int)
        mushroom['column10'] = mushroom['column10'].astype(int)
        mushroom['column11'] = mushroom['column2'].astype(int)
        mushroom['column12'] = mushroom['column2'].astype(int)
        mushroom['column13'] = mushroom['column3'].astype(int)
        mushroom['column14'] = mushroom['column4'].astype(int)
        mushroom['column15'] = mushroom['column5'].astype(int)
        mushroom['column16'] = mushroom['column6'].astype(int)
        mushroom['column17'] = mushroom['column7'].astype(int)
        mushroom['column18'] = mushroom['column8'].astype(int)
        mushroom['column19'] = mushroom['column9'].astype(int)
        mushroom['column20'] = mushroom['column10'].astype(int)
        mushroom['column21'] = mushroom['column8'].astype(int)
        mushroom['column22'] = mushroom['column9'].astype(int)
        mushroom['column23'] = mushroom['column10'].astype(int)

        # Change coloumn order. Moving the decision column to the end
        mushroom = mushroom[["column2", "column3", "column4", "column5", "column6", "column7", "column8", "column9", "column10", "column11", "column12",
                             "column13", "column14", "column15", "column16", "column17", "column18", "column19", "column20", "column21", "column22", "column23", "decision"]]

        # Naive Bayes Classifier
        for i in range(10):
            decision_column = 22  # First column is treated as decision column
            accuracy = fit(mushroom.values,
                           naive_bayes_classifier, decision_column)
            print('Accuracy for Mushroom Dataset: %s' % accuracy)
            print('Mean Accuracy for Mushroom Dataset: %.3f%%' %
                  (sum(accuracy)/float(len(accuracy))))

    elif dataset_name == "ecoli":

        ecoli = pd.read_csv("./dataset/ecoli.data", names=["column1", "column2", "column3", "column4",
                            "column5", "column6", "column7", "column8", "decision"], delim_whitespace=True)

        # Shuffle the dataset
        ecoli = ecoli.sample(frac=1)

        ecoli = ecoli.replace("?", np.NaN)
        ecoli.dropna()

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

        # Naive Bayes Classifier
        for i in range(10):
            decision_column = 7  # Last column is treated as decision column
            accuracy = fit(ecoli.values, naive_bayes_classifier,
                           decision_column)
            print('Accuracy for Ecoli Dataset: %s' % accuracy)
            print('Mean Accuracy for Ecoli Dataset: %.3f%%' %
                  (sum(accuracy)/float(len(accuracy))))

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

        # Change coloumn order. Moving the decision column to the end
        letterrecognition = letterrecognition[["column2", "column3", "column4", "column5", "column6", "column7", "column8",
                                               "column9", "column10", "column11", "column12", "column13", "column14", "column15", "column16", "column17", "decision"]]

        # Naive Bayes Classifier
        for i in range(10):
            decision_column = 16  # Last column is treated as decision column
            accuracy = fit(letterrecognition.values,
                           naive_bayes_classifier, decision_column)
            print('Accuracy for Letter Dataset Recognition: %s' % accuracy)
            print('Mean Accuracy for Letter Dataset Recognition: %.3f%%' %
                  (sum(accuracy)/float(len(accuracy))))

    else:
        print("Give proper dataset")
