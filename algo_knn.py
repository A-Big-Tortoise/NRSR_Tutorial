from sklearn.neighbors import KNeighborsClassifier

def algo_knn(X, y, X_test, y_test, n_neighbors):
    """
    Define a function to perform the K-nearest neighbors (KNN) algorithm for classification.
    Inputs:
        X: Training data features (numpy array)
        y: Training data labels (numpy array)
        X_test: Test data features for prediction (numpy array)
        y_test: True labels for the test data (numpy array)
        n_neighbors: Number of neighbors to consider in KNN (integer)
    Outputs:
        y_hat: Predicted labels for the test data (numpy array)
        score_: Accuracy score of the KNN classifier on the test data (float)
    """

    # Create a K-nearest neighbors classifier with the specified number of neighbors
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the classifier to the training data
    neigh.fit(X, y)

    # Predict labels for the test data
    y_hat = neigh.predict(X_test)

    # Calculate the accuracy score of the classifier on the test data
    score_ = neigh.score(X_test, y_test)

    # Return the predicted labels and accuracy score
    return y_hat, score_


if __name__ == '__main__':
    import numpy as np
    from Dataset import load_scg_template
    from sklearn.metrics import confusion_matrix

    # Define a function to prepare data for K-nearest neighbors (KNN) algorithm
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

    # Prepare the training and test data for KNN
    X, y = prepare_data(signals_train, labels_train)
    X_test, y_test = prepare_data(signals_test, labels_test)

    # Set the number of neighbors for the KNN algorithm
    n_neighbors = 10

    # Apply the KNN algorithm to make predictions and compute the accuracy
    y_pred, score = algo_knn(X, y, X_test, y_test, n_neighbors)

    # Compute the confusion matrix to evaluate the KNN classifier
    con_matrix = confusion_matrix(y_test, y_pred)

    # Display the results of the KNN algorithm
    print('Result of Knn')
    print('---accuracy---')
    print(score)
    print('---confusion matrix---')
    print(con_matrix)

