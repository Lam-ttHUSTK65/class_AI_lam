import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score


def evaluate_performance(num_trials=100):
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross-validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape

    folds = 10

    accuracy_results = []
    stump_accuracy_results = []
    dt3_accuracy_results = []

    for t in range(num_trials):
        # Shuffle the data at the beginning of each trial
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Initialize classifiers
        clf = tree.DecisionTreeClassifier()
        clf_stump = tree.DecisionTreeClassifier(max_depth=1)
        clf_dt3 = tree.DecisionTreeClassifier(max_depth=3)

        # Split the data
        for i in range(folds):
            # Choose 10% of the data for testing
            start = int(round((len(X) / 10.0) * i))
            stop = int(round((len(X) / 10.0) * (i + 1)))

            X_train = np.concatenate((X[0:start, :], X[stop:, :]))
            X_test = X[start:stop, :]

            y_train = np.concatenate((y[0:start, :], y[stop:, :]))
            y_test = y[start:stop, :]

            # Decision Tree
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy_results.append(accuracy_score(y_test, y_pred))

            # Decision Stump
            clf_stump = clf_stump.fit(X_train, y_train)
            y_pred_stump = clf_stump.predict(X_test)
            stump_accuracy_results.append(accuracy_score(y_test, y_pred_stump))

            # Level 3 Decision Tree
            clf_dt3 = clf_dt3.fit(X_train, y_train)
            y_pred_dt3 = clf_dt3.predict(X_test)
            dt3_accuracy_results.append(accuracy_score(y_test, y_pred_dt3))

    # Update statistics based on the results of the experiment
    mean_decision_tree_accuracy = np.mean(accuracy_results)
    stddev_decision_tree_accuracy = np.std(accuracy_results)

    mean_decision_stump_accuracy = np.mean(stump_accuracy_results)
    stddev_decision_stump_accuracy = np.std(stump_accuracy_results)

    mean_dt3_accuracy = np.mean(dt3_accuracy_results)
    stddev_dt3_accuracy = np.std(dt3_accuracy_results)

    # Make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = mean_decision_tree_accuracy
    stats[0, 1] = stddev_decision_tree_accuracy
    stats[1, 0] = mean_decision_stump_accuracy
    stats[1, 1] = stddev_decision_stump_accuracy
    stats[2, 0] = mean_dt3_accuracy
    stats[2, 1] = stddev_dt3_accuracy
    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluate_performance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Decision Stump Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("3-level Decision Tree = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.