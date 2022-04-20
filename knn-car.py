from collections import Counter

import numpy as np, pandas as pd
from sklearn.model_selection import KFold 

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k):
        self.k = k
        #return df

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    car_dataset = pd.read_csv("./dataset/car.data", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])
    car_dataset["buying"].replace(["vhigh", "high", "med", "low"], [3, 2, 1, 0], inplace = True)
    car_dataset["maint"].replace(["vhigh", "high", "med", "low"], [3, 2, 1, 0], inplace = True)
    car_dataset["doors"].replace(["5more"], [5], inplace = True)
    car_dataset["persons"].replace(["more"], [4], inplace = True)
    car_dataset["lug_boot"].replace(["small", "med", "big"], [0, 1, 2], inplace = True)
    car_dataset["safety"].replace(["low", "med", "high"], [0, 1, 2], inplace = True)
    car_dataset["decision"].replace(["unacc", "acc", "good", "vgood"], [1, 0, 2, 3], inplace = True)
        
    car_dataset['buying'] = car_dataset['buying'].astype(int)
    car_dataset['maint'] = car_dataset['maint'].astype(int)
    car_dataset['doors'] = car_dataset['doors'].astype(int)
    car_dataset['persons'] = car_dataset['persons'].astype(int)
    car_dataset['lug_boot'] = car_dataset['lug_boot'].astype(int)
    car_dataset['safety'] = car_dataset['safety'].astype(int)
    car_dataset['decision'] = car_dataset['decision'].astype(int)
    acc =[]
    
    for i in range(10):
        car_dataset = car_dataset.sample(frac=1)
        k = 2
        
        model = KNN(k)
        X = car_dataset.drop(["decision"], axis = 1)
        y = car_dataset.decision
        X = X.to_numpy()
        y = y.to_numpy()

        kf = KFold(n_splits=5)
        for train_index , test_index in kf.split(X):
            X_train , X_test = X[train_index,:],X[test_index,:]
            y_train , y_test = y[train_index] , y[test_index]
     
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
     
            acc.append(accuracy_score(pred_values , y_test))

    mean = sum(acc)/len(acc)

    print("KNN classification accuracy", mean)