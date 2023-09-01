from sklearn.ensemble import RandomForestClassifier


def algo_rf(X, y, X_test, y_test, n_estimators, max_depth):
    """
    # Define a function to perform Random Forest Classification (RFC).

    Parameters
    ----------
        X: Training data features (numpy array)
        y: Training data labels (numpy array)
        X_test: Test data features for prediction (numpy array)
        y_test: True labels for the test data (numpy array)
        n_estimators: Number of decision trees in the forest (integer)
        max_depth: Maximum depth of the individual decision trees (integer)

    Returns
    -------
        y_hat: Predicted labels for the test data (numpy array)
        score_: Accuracy score of the RFC classifier on the test data (float)
    """

    # Create a Random Forest Classification (RFC) classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # Fit the classifier to the training data
    clf.fit(X, y)

    # Predict labels for the test data
    y_hat = clf.predict(X_test)

    # Calculate the accuracy score of the classifier on the test data
    score_ = clf.score(X_test, y_test)

    # Return the predicted labels and accuracy score
    return y_hat, score_


if __name__ == '__main__':
    import numpy as np
    from Dataset import load_scg_template
    from sklearn.metrics import confusion_matrix

    # Define a function to prepare data for Random Forest algorithm
    # Using heartbeat templates for a classification task, we primarily classify
    # the S index into four categories (S = 90, S = 120, S = 150, S = 180).
    def prepare_data(signals, labels):
        # Define a function to map a continuous value 'S' to a discrete label
        def check_S(S):
            label = -1
            if 90 <= S < 105:
                label = 0
            elif 105 <= S < 135:
                label = 1
            elif 135 <= S < 165:
                label = 2
            elif S >= 165:
                label = 3
            return label

        # Initialize a list to store mapped labels
        S_ = []

        # Map continuous labels to discrete labels for each data point
        for label in labels:
            S_.append(check_S(label[-2]))

        # Find the length of the longest signal in the dataset
        longest_template = -1
        for signal in signals:
            if len(signal) > longest_template:
                longest_template = len(signal)

        # Pad all signals to have the same length as the longest signal
        padded_signals = []
        for signal in signals:
            padded_signal = np.pad(signal, (0, longest_template - len(signal)), 'constant', constant_values=0)
            padded_signals.append(padded_signal)

        # Convert the padded signals and labels to NumPy arrays
        padded_signals_np = np.array(padded_signals)
        S_np = np.array(S_)

        return padded_signals_np, S_np

    # Load training and test data using the 'load_scg_template' function
    signals_train, labels_train, duration, fs = load_scg_template(0.1, 'train')
    signals_test, labels_test, _, _ = load_scg_template(0.1, 'test')

    # Prepare the training and test data for Random Forest
    X, y = prepare_data(signals_train, labels_train)
    X_test, y_test = prepare_data(signals_test, labels_test)

    # Set the parameters for the Random Forest algorithm
    n_estimators = 500
    max_depth = 10

    # Apply the Random Forest algorithm to make predictions and compute the accuracy
    y_pred, score = algo_rf(X, y, X_test, y_test, n_estimators, max_depth)

    # Compute the confusion matrix to evaluate the Random Forest classifier
    con_matrix = confusion_matrix(y_test, y_pred)

    # Display the results of the Random Forest algorithm
    print('Result of Random Forest')
    print('---accuracy---')
    print(score)
    print('---confusion matrix---')
    print(con_matrix)
