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
            self._mean[idx, :] = X_c.mean(axis=0) #[10, 10] 10 n_classes=2 n_features=10
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
        numerator = np.exp(-((x - mean) ** 2) / (2 * var)) # 10
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


def tt_split(X, y, test_size=0.6):

    i = int((1 - test_size) * X.shape[0]) 
    o = np.random.permutation(X.shape[0])
    
    X_train, X_test = np.split(np.take(X,o,axis=0), [i])
    y_train, y_test = np.split(np.take(y,o), [i])
    return X_train, X_test, y_train, y_test

# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    cat_dataset = pd.read_csv("./dataset/car.data", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])
    
    # Shuffle the dataset 
    cat_dataset = cat_dataset.sample(frac=1)

    cat_dataset["decision"].replace(["unacc", "acc", "good", "vgood"], [-1, 0, 1, 2], inplace = True)
    cat_dataset["safety"].replace(["low", "med", "high"], [0, 1, 2], inplace = True)
    cat_dataset["lug_boot"].replace(["small", "med", "big"], [0, 1, 2], inplace = True)
    cat_dataset["persons"].replace(["more"], [4], inplace = True)
    cat_dataset["doors"].replace(["5more"], [6], inplace = True)
    cat_dataset["maint"].replace(["vhigh", "high", "med", "low"], [4, 3, 2, 1], inplace = True)
    cat_dataset["buying"].replace(["vhigh", "high", "med", "low"], [4, 3, 2, 1], inplace = True)

    # predictors = cat_dataset.drop("decision", axis = "columns")
    # target = cat_dataset.decision

    # print("Shape of Predictors:",predictors.shape)
    # print(predictors)

    # print("Shape of Target:",target.shape)
    # print(target)

    print("Enter the splitting factor (i.e) ratio between train and test")

    # # Define a size for your train set 
    # predictors_train_size = int(0.6 * len(predictors))
    # target_train_size = int(0.6 * len(target))

    # # Split your dataset 
    # X_train = predictors[:predictors_train_size]
    # y_train = target[:target_train_size]

    # X_test = predictors[predictors_train_size:]
    # y_test = target[target_train_size:]

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    predictors,target = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        predictors,target, test_size=0.2, random_state=123
    )

    print("X_train:")
    print(X_train)
    print("\ny_train:")
    print(y_train)
    print("\nX_test")
    print(X_test)
    print("\ny_test")
    print(y_test)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))