import numpy as np
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')

class NaiveBayes:
    
    def calculate_mean_variance(self, X, y):
        num_of_traning_records, num_of_features = X.shape
        self.target_classes = np.unique(y)  # target classes with unique values
        num_of_target_classes = len(self.target_classes) # Total number of target classes

        # calculatung mean to find out liklihood probability  for each class
        self.actual_mean = np.zeros((num_of_target_classes, num_of_features), dtype=np.float64)

        # calculatung variance to find out liklihood probability  for each class
        self.actual_variance = np.zeros((num_of_target_classes, num_of_features), dtype=np.float64)

        # calculatung prior to find out liklihood probability for each class
        self.priors_value = np.zeros(num_of_target_classes, dtype=np.float64)

        # Calculated Mean, Variance and Prior for each class
        for index, target_class in enumerate(self.target_classes):
            training_class = X[y == target_class]
            self.actual_mean[index, :] = training_class.mean(axis=0)
            self.actual_variance[index, :] = training_class.var(axis=0)
            self.priors_value[index] = training_class.shape[0] / float(num_of_traning_records)
    
    def prediction(self, X):
        y_pred = [self._calculated_prediction(x) for x in X]
        return np.array(y_pred)

    def _calculated_prediction(self, x):
        liklihoods = []
        # calculating the liklihood/posterior probability for each class
        for index, c in enumerate(self.target_classes):
            prior_probability = np.log(self.priors_value[index])
            liklihood_propability = np.ma.masked_invalid(np.log(self.calculate_liklihood_probability(index, x))).sum()
            liklihood_propability = prior_probability + liklihood_propability
            liklihoods.append(liklihood_propability)

        # return class with highest posterior probability
        return self.target_classes[np.argmax(liklihoods)]
        
    def calculate_liklihood_probability(self, index, x):
        mean = self.actual_mean[index]
        variance = self.actual_variance[index]
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        result = numerator / denominator
        result[np.isnan(result)] = 0
        return result

# Testing
if __name__ == "__main__":

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

    cat_dataset['decision'] = cat_dataset['decision'].astype(int)
    cat_dataset['safety'] = cat_dataset['safety'].astype(int)
    cat_dataset['lug_boot'] = cat_dataset['lug_boot'].astype(int)
    cat_dataset['persons'] = cat_dataset['persons'].astype(int)
    cat_dataset['doors'] = cat_dataset['doors'].astype(int)
    cat_dataset['maint'] = cat_dataset['maint'].astype(int)
    cat_dataset['buying'] = cat_dataset['maint'].astype(int)

    predictors = cat_dataset.drop("decision", axis = "columns")
    target = cat_dataset.decision

    print("Shape of Predictors:",predictors.shape)
    print(predictors)

    print("Shape of Target:",target.shape)
    print(target)

    print("Enter the splitting factor (i.e) ratio between train and test")

    # Define a size for your train set 
    predictors_train_size = int(0.6 * len(predictors))
    target_train_size = int(0.6 * len(target))

    # Split your dataset 
    X_train = predictors[:predictors_train_size]
    y_train = target[:target_train_size]

    X_test = predictors[predictors_train_size:]
    y_test = target[target_train_size:]

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # X_train, X_test, y_train, y_test = train_test_split(
    #     predictors,target, test_size=0.2, random_state=123
    # )

    print("X_train:")
    print(X_train)
    print("\ny_train:")
    print(y_train)
    print("\nX_test")
    print(X_test)
    print("\ny_test")
    print(y_test)

    nb = NaiveBayes()
    nb.calculate_mean_variance(X_train, y_train)

    predictions = nb.prediction(X_test.values)

    print("Naive Bayes classification accuracy", accuracy(y_test.values, predictions))