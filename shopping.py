import csv
import sys
import pandas as pd 
import numpy as np  

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    #importing the data set 
    data = pd.read_csv("shopping.csv", header = 0)


    # mapping the months to the assigned month number 
    d = {"Jan":int(0), "Feb":int(1), "Mar":int(2), "Apr":int(3), "May":int(4), "June":int(5), "Jul":int(6), 
            "Aug":int(7), "Sep":int(8), "Oct":int(9), 
    		"Nov":int(10), "Dec":int(11)}

    #changing the month abrivations to the set values in list "d"
    data.Month = data.Month.map(d)


    # change the visitor values to be 1 if there were visitors and 0 is there were not 
    for i in range(len(data.VisitorType)):
        if data.VisitorType[i] == 'Returning_Visitor':
            data.VisitorType[i] = int(1)

        elif data.VisitorType[i] == 'New_Visitor':
            data.VisitorType[i] = int(0)

        elif data.VisitorType[i] == 'Other':
            data.VisitorType[i] = int(0)

    # changing the values for Weekend to 1 or 0 
    data.Weekend = data.Weekend.replace([True, False],[1,0])
    data.Weekend = data.Weekend.astype(int)
    for i in range(len(data.Weekend)):
        if data.Weekend[i] == True:
            data.Weekend[i] = int(1)

        elif data.Weekend[i] == False:
            data.Weekend[i] = int(0)

    # changing the revenue values to 1 or 0 
    data.Revenue = data.Revenue.replace([True, False],[1,0])
    for i in range(len(data.Revenue)):
        if data.Revenue[i] == True:
            data.Revenue[i] = int(1)

        elif data.Revenue[i] == False:
            data.Revenue[i] = int(0)


    # creating an evidence array from the data 
    evidence = data.iloc[:,:-1].values.tolist()
    # creating a label array from the data 
    labels   = data.iloc[:, -1].values.tolist()

    return (evidence, labels)


    #raise NotImplementedError


def train_model(evidence, labels):
    
    # create the model 
    model = KNeighborsClassifier(n_neighbors = 1)

    # fit the model to the data set 
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    # finding the number of positive and negative values from the labels 
    positives = labels.count(1)
    negatives = labels.count(0)

    # setting the sensitivity and specificity values equalt to zero at the beginning 
    sensitivity = 0
    specificity = 0


    # iterate over the labels and the predictions (they are the same length so zip works)
    for i, j in zip(labels, predictions):
        if i == 1:
            # correct prediction -> increase the sensitivity by 1
            if i == j:
                sensitivity += 1

        else:
            # correct prediction -> increase specificity by 1 
            if i == j:
                specificity += 1 

    # calculating the final sensitivity and specificity 
    final_sensitivity = sensitivity / positives
    final_specificity = specificity / negatives

    return (final_sensitivity, final_specificity)
    #raise NotImplementedError


if __name__ == "__main__":
    main()
