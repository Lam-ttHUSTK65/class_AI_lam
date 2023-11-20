import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score


def generate_learning_curve(X_train, y_train, X_test, y_test, classifier):
    """
    Generate a learning curve for a given classifier.
    """
    train_sizes = np.arange(0.1, 1.1, 0.1)
    mean_accuracies = []
    std_accuracies = []

    for size in train_sizes:
        # Select a subset of the training data
        train_subset_size = int(len(X_train) * size)
        X_subset = X_train[:train_subset_size, :]
        y_subset = y_train[:train_subset_size, :]

        # Fit the classifier on the subset
        classifier.fit(X_subset, y_subset)

        # Make predictions on the test set
        y_pred = classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mean_accuracies.append(np.mean(accuracy))
        std_accuracies.append(np.std(accuracy))

    return train_sizes, mean_accuracies, std_accuracies


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

    # run 100 trials
    folds = 10

    accuracy_results = []
    stump_accuracy_results = []
    dt3_accuracy_results = []

    # Initialize decision tree classifiers with various depths
    depths = [1, 3, 5, 7]
    classifiers = [tree.DecisionTreeClassifier(max_depth=depth) for depth in depths]

    # Plotting setup
    plt.figure(figsize=(10, 6))
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves for Decision Trees")

    for classifier in classifiers:
        mean_accuracies_all_trials = []
        std_accuracies_all_trials = []

        for t in range(num_trials):
            # shuffle the data at the beginning of each trial
            idx = np.arange(n)
            np.random.seed(13)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            mean_accuracies, std_accuracies = [], []

            # split the data
            for i in range(folds):
                # pick 10% of the data for testing
                start = int(round((len(X) / 10.0) * i))
                stop = int(round((len(X) / 10.0) * (i + 1)))

                X_train = np.concatenate((X[0:start, :], X[stop:, :]))
                X_test = X[start:stop, :]

                y_train = np.concatenate((y[0:start, :], y[stop:, :]))
                y_test = y[start:stop, :]

                # Generate learning curve for the current fold
                train_sizes, mean_accuracy, std_accuracy = generate_learning_curve(
                    X_train, y_train, X_test, y_test, classifier
                )

                mean_accuracies.append(mean_accuracy)
                std_accuracies.append(std_accuracy)

            # Average the learning curves over all folds
            mean_accuracies_all_trials.append(np.mean(mean_accuracies, axis=0))
            std_accuracies_all_trials.append(np.mean(std_accuracies, axis=0))

        # Average the learning curves over all trials
        mean_accuracies_avg = np.mean(mean_accuracies_all_trials, axis=0)
        std_accuracies_avg = np.mean(std_accuracies_all_trials, axis=0)

        # Plot the learning curve with error bars for standard deviation
        plt.errorbar(
            train_sizes, mean_accuracies_avg, yerr=std_accuracies_avg,
            label=f"Depth {classifier.max_depth}", alpha=0.7, fmt='-o'
        )

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()


if __name__ == "__main__":
    evaluate_performance()
