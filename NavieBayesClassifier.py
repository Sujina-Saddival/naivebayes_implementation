from dataclasses import dataclass
import numpy as np
import pandas as pd
import sys

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

    datase_name = sys.argv[1]
    # datase_name = "breastcancer"

    if datase_name == "car":

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
        cat_dataset['buying'] = cat_dataset['buying'].astype(int)

        # Cross validation
        average_accuracy = 0

        for i in range(10):

            # Shuffle the dataset 
            cat_dataset = cat_dataset.sample(frac=1)

            predictors = cat_dataset.drop("decision", axis = "columns")
            target = cat_dataset.decision

            print("Shape of Predictors:",predictors.shape)
            print(predictors)

            print("Shape of Target:",target.shape)
            print(target)

            print("Enter the splitting factor (i.e) ratio between train and test")

            # Define a size for your train set 
            predictors_train_size1 = int(0.2 * len(predictors))
            target_train_size1 = int(0.2 * len(target))

            predictors_train_size2 = int(0.2 * len(predictors))
            target_train_size2 = int(0.2 * len(target))

            predictors_train_size3 = int(0.2 * len(predictors))
            target_train_size3 = int(0.2 * len(target))

            predictors_train_size = predictors_train_size1 + predictors_train_size2 + predictors_train_size3
            target_train_size = target_train_size1 + target_train_size2 + target_train_size3

            # predictors_train_size = int(0.6 * len(predictors))
            # target_train_size = int(0.6 * len(target))

            # Split your dataset 
            X_train = predictors[:predictors_train_size]
            y_train = target[:target_train_size]

            X_test = predictors[predictors_train_size:]
            y_test = target[target_train_size:]

            def accuracy(y_true, y_pred):
                accuracy = np.sum(y_true == y_pred) / len(y_true)
                return accuracy

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

            accuracy = accuracy(y_test.values, predictions)
            average_accuracy = ((accuracy * 100) + average_accuracy/10)

            print("\nNaive Bayes classification accuracy", accuracy * 100 , "\n")

        print("Average accuracy for 10 folds", average_accuracy)
    
    elif datase_name == "breastcancer":

        breastcancer = pd.read_csv("./dataset/breast-cancer-wisconsin.data", names=["column1", "column2", "column3", "column4", "column5", "column6", "column7", "column8", "column9", "column10", "decision"])
        
        # Shuffle the dataset 
        breastcancer = breastcancer.sample(frac=1)

        breastcancer = breastcancer.replace("?", np.NaN)
        breastcancer = breastcancer.dropna()

        breastcancer['column1'] = breastcancer['column1'].astype(int)
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

        # Cross validation
        average_accuracy = 0

        for i in range(10):

            # Shuffle the dataset 
            breastcancer = breastcancer.sample(frac=1)

            predictors = breastcancer.drop("decision", axis = "columns")
            target = breastcancer.decision

            print("Shape of Predictors:",predictors.shape)
            print(predictors)

            print("Shape of Target:",target.shape)
            print(target)

            print("Enter the splitting factor (i.e) ratio between train and test")

            # Define a size for your train set 
            predictors_train_size1 = int(0.2 * len(predictors))
            target_train_size1 = int(0.2 * len(target))

            predictors_train_size2 = int(0.2 * len(predictors))
            target_train_size2 = int(0.2 * len(target))

            predictors_train_size3 = int(0.2 * len(predictors))
            target_train_size3 = int(0.2 * len(target))

            predictors_train_size = predictors_train_size1 + predictors_train_size2 + predictors_train_size3
            target_train_size = target_train_size1 + target_train_size2 + target_train_size3

            # predictors_train_size = int(0.6 * len(predictors))
            # target_train_size = int(0.6 * len(target))

            # Split your dataset 
            X_train = predictors[:predictors_train_size]
            y_train = target[:target_train_size]

            X_test = predictors[predictors_train_size:]
            y_test = target[target_train_size:]

            def accuracy(y_true, y_pred):
                accuracy = np.sum(y_true == y_pred) / len(y_true)
                return accuracy

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

            accuracy = accuracy(y_test.values, predictions)
            average_accuracy = ((accuracy) + average_accuracy/10)

            print("\nNaive Bayes classification accuracy", accuracy * 100 , "\n")

        print("Average accuracy for 10 folds", average_accuracy * 100)

    else:
        print("Give proper dataset")
