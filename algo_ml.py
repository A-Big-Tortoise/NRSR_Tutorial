# 模块尽量使用小写命名，首字母保持小写，尽量不用下划线(除非多个单词，且数量不多的情况)
import numpy as np

def algo_linearregression(data_train, label_train, data_test, label_test):
    """
    Perform linear regression on the given data and calculate Mean Absolute Error (MAE).

    Args:
    data_train (array-like): Training data with shape (n_samples, n_features).
    label_train (array-like): Training labels with shape (n_samples,) or (n_samples, n_targets).
    data_test (array-like): Test data with shape (n_samples, n_features).
    label_test (array-like): Test labels with shape (n_samples,) or (n_samples, n_targets).

    Returns:
    float: Mean Absolute Error (MAE) between predicted and actual labels.
    array-like: Predicted labels for the test data.
    """

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    # Convert input data and labels to numpy arrays
    data_train_np = np.array(data_train)
    data_test_np = np.array(data_test)
    label_train_np = np.array(label_train)
    label_test_np = np.array(label_test)

    # Reshape 1D arrays to 2D if necessary
    if data_train_np.ndim == 1:
        data_train_np = data_train_np.reshape(-1, 1)
    if data_test_np.ndim == 1:
        data_test_np = data_test_np.reshape(-1, 1)
    if label_train_np.ndim == 1:
        label_train_np = label_train_np.reshape(-1, 1)
    if label_test_np.ndim == 1:
        label_test_np = label_test_np.reshape(-1, 1)

    # Fit a Linear Regression model
    reg = LinearRegression().fit(data_train_np, label_train_np)

    # Predict labels for test data
    y_hat = reg.predict(data_test_np)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(label_test_np, y_hat)

    return mae, y_hat

# 2. random forest


