import numpy as np
import pandas as pd

class NaiveBayes:
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# Testing
if __name__ == "__main__":

    import random
    import csv

    cols_to_remove = [6] # Column indexes to be removed (starts at 0)

    cols_to_remove = sorted(cols_to_remove, reverse=True) # Reverse so we remove from the end first
    row_count = 0 # Current amount of rows processed

    fid = open("./dataset/car.data", "r")
    li = fid.readlines()
    fid.close()
    print(li)

    random.shuffle(li)
    print(li)

    fid = open("./dataset/car_shuffled.data", "w")
    fidr = csv.writer(fid)
    fidr.writerow(["buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])
    li = ''.join([i for i in li]) 
    li = li.replace("vhigh", "3")
    li = li.replace("high", "2")
    li = li.replace("med", "1")
    li = li.replace("low", "0")
    li = li.replace("small", "0")
    li = li.replace("big", "2")
    li = li.replace("more", "4")
    li = li.replace("5more", "6")
    li = li.replace("unacc", "0")
    li = li.replace("acc", "1")
    li = li.replace("good", "2")
    li = li.replace("vgood", "3")
    for i in li.split('\n'):
        fidr.writerow(i.split(','))
    fid.close()

    with open("./dataset/car_shuffled.data", "r") as source:
        reader = csv.reader(source)
        with open("./dataset/car_shuffled.data", "w", newline='') as result:
            writer = csv.writer(result)
            for row in reader:
                row_count += 1
                print('\r{0}'.format(row_count), end='') # Print rows processed
                for col_index in cols_to_remove:
                    del row[col_index]
                writer.writerow(row)

    # cat_dataset = pd.read_csv("./dataset/car.data", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])
    
    # # Shuffle the dataset 
    # cat_dataset = cat_dataset.sample(frac=1)


    # cat_dataset["decision"].replace(["unacc", "acc", "good", "vgood"], [-1, 0, 1, 2], inplace = True)

    # predictors = cat_dataset.drop("decision", axis = "columns")
    # target = cat_dataset.decision
    
    # predictors["buying"].replace(["vhigh", "high", "med", "low"], [0.25, 0.25, 0.25, 0.25], inplace = True)
    # predictors["maint"].replace(["vhigh", "high", "med", "low"], [0.25, 0.25, 0.25, 0.25], inplace = True)
    # predictors["lug_boot"].replace(["small", "med", "big"], [0.25, 0.25, 0.25], inplace = True)
    # predictors["safety"].replace(["low", "med", "high"], [0.25, 0.25, 0.25], inplace = True)
    # predictors["persons"].replace(["more"], [4], inplace = True)
    # predictors["doors"].replace(["5more"], [6], inplace = True)

    # print(predictors.head())
    # print(target.head())

    # # Define a size for your train set 
    # predictors_train_size = int(0.6 * len(predictors))
    # target_train_size = int(0.6 * len(target))

    # # Split your dataset 
    # X_train = predictors[:predictors_train_size]
    # y_train = target[:target_train_size]

    # X_test = predictors[predictors_train_size:]
    # y_test = target[target_train_size:]

    # print(X_train.head())
    # print(y_train.head())

    # def accuracy(y_true, y_pred):
    #     accuracy = np.sum(y_true == y_pred) / len(y_true)
    #     return accuracy

    # print([X_train, X_test, y_train, y_test])

    # nb = NaiveBayes()
    # nb.fit(X_train, y_train)
    # predictions = nb.predict(X_test)

    # print("Naive Bayes classification accuracy", accuracy(y_test, predictions))