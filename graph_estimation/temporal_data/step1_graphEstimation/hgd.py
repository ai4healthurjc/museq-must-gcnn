"""Heterogeneous Gower Distance (HGD) between pairs of variables."""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def elements_in_list(large_list, small_list):
    """
    Checks if each element in the large list is present in the small list.

    Parameters:
    large_list (list): The list to search within.
    small_list (list): The list containing elements to search for.

    Returns:
    list: A list of booleans indicating presence of each element.
    """
    return [element in small_list for element in large_list]


def rescale_binary_continuous(df, binary_row, continuous_row):
    """
    Rescales continuous variables based on the binary classification (0 or 1).

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    binary_row (int): The index of the binary row in the DataFrame.
    continuous_row (int): The index of the continuous row in the DataFrame.

    Returns:
    DataFrame: The rescaled DataFrame.
    """
    df_aux = df.copy()

    max_value = np.max(df.iloc[continuous_row])
    filtered_array = [element for element in df.iloc[continuous_row] if element != -1]
    min_value = np.min(filtered_array)

    large_list = df.columns.values

    small_list = np.where(df_aux.iloc[binary_row] == 0)[0]
    df.iloc[binary_row] = df.iloc[binary_row].where(elements_in_list(large_list, small_list), max_value)

    small_list = np.where(df_aux.iloc[binary_row] == 1)[0]
    df.iloc[binary_row] = df.iloc[binary_row].where(elements_in_list(large_list, small_list), min_value)

    return df


def rescale_continuous_continuous(df):
    """
    Rescales one continuous variable based on the maximum value of another.

    Parameters:
    df (DataFrame): The DataFrame containing the data.

    Returns:
    DataFrame: The rescaled DataFrame.
    """
    max_row = df.values.argmax() // df.shape[1]
    max_value = df.loc[max_row].max()

    if max_row == 0:
        df.loc[1] = list((df.iloc[1] * max_value) / df.loc[1].max())
    else:
        df.loc[0] = list((df.iloc[0] * max_value) / df.loc[0].max())

    return df


def minmax_for_hgd(X, missing=666):
    """Rescale each variable (last axis) to [0, 1] with the training-set range,
    leaving missing entries untouched. The HGD implementation below encodes
    missing values as -1 and assumes non-negative observations, so inputs that
    were normalized with a scheme producing negative values (e.g. robust
    scaling) must be brought to [0, 1] before estimating the graphs."""
    X = np.array(X, dtype=float, copy=True)
    mask = (X == missing)
    flat = X.reshape(-1, X.shape[-1])
    fmask = mask.reshape(-1, X.shape[-1])
    for j in range(flat.shape[1]):
        obs = flat[~fmask[:, j], j]
        if obs.size == 0:
            continue
        lo, hi = obs.min(), obs.max()
        flat[~fmask[:, j], j] = (obs - lo) / (hi - lo) if hi > lo else 0.0
    X = flat.reshape(X.shape)
    X[mask] = missing
    return X


def reshape_patients_by_features(data, keys, numberOfTimeStep):
    """
    Reshapes the data based on the number of features and time steps.

    Parameters:
    data (ndarray): The original data array.
    keys (list): List of feature names.
    numberOfTimeStep (int): The number of time steps.

    Returns:
    ndarray: The reshaped data.
    """
    df = pd.DataFrame(data.reshape(int(data.shape[0]*numberOfTimeStep), data.shape[2]))
    df.replace(666, -1, inplace=True)
    df.columns = keys

    for i in range(len(keys)):
        df_trial = df[keys[i]]
        if i == 0:
            X = np.array(df_trial)
            X = X.reshape((int(df_trial.shape[0]/numberOfTimeStep), numberOfTimeStep)).T
            X = X.reshape(1, numberOfTimeStep, int(df_trial.shape[0]/numberOfTimeStep))

        else:
            X_2 = np.array(df_trial)
            X_2 = X_2.reshape((int(df_trial.shape[0]/numberOfTimeStep), numberOfTimeStep)).T
            X_2 = X_2.reshape(1, numberOfTimeStep, int(df_trial.shape[0]/numberOfTimeStep))

            X = np.append(X, X_2, axis=0)

    return X


def diagonal_to_zero(matrix):
    """
    Sets the diagonal elements of a matrix to zero.

    Parameters:
    matrix (ndarray): The input matrix.

    Returns:
    ndarray: The matrix with diagonal elements set to zero.
    """
    return np.array([[0 if i == j else element for j, element in enumerate(row)] for i, row in enumerate(matrix)])


def binary_binary_hgd(df):
    """
    Implements the Heterogeneous Gower Distance (HGD) for binary-binary variable pairs.

    Parameters:
    df (DataFrame): The DataFrame containing the binary variables.

    Returns:
    float: The HGD distance for the binary variables.
    """
    max_value = np.max(np.abs(df[df != -1].iloc[0] - df[df != -1].iloc[1]))
    if np.isnan(max_value):
        max_value = 0

    sj = []
    pat_considered = 0
    for i in range(df.shape[1]):
        if (df.iloc[0, i] != -1) and (df.iloc[1, i] != -1):
            try:
                sj.append(1 - (np.abs(df.iloc[0,i] - df.iloc[1,i]) / (max_value)))
            except ZeroDivisionError:
                sj.append(1)
            pat_considered += 1

    try:
        output = 1 - (1/pat_considered) * np.sum(sj)
    except ZeroDivisionError:
        output = 0

    return output


def binary_continuous_hgd(df):
    """
    Implements the Heterogeneous Gower Distance (HGD) for binary-continuous variable pairs.

    Parameters:
    df (DataFrame): The DataFrame containing the binary and continuous variables.

    Returns:
    float: The HGD distance for the binary-continuous variables.
    """
    # row 0 is always the binary variable (ordered in hgd_distance), so the
    # result does not depend on the order of the pair nor on missing values
    df = rescale_binary_continuous(df, 0, 1)

    sj = []
    max_value = np.max(np.abs(df[df != -1].iloc[0] - df[df != -1].iloc[1]))
    if np.isnan(max_value):
        max_value = 0

    pat_considered = 0
    for i in range(df.shape[1]):
        if (df.iloc[0, i] != -1) and (df.iloc[1, i] != -1):
            try:
                sj.append(1 - (np.abs(df.iloc[0,i] - df.iloc[1,i]) / (max_value)))
            except ZeroDivisionError:
                sj.append(1)
            pat_considered += 1

    try:
        output = 1 - (1/pat_considered) * np.sum(sj)
    except ZeroDivisionError:
        output = 0

    return output


def continuous_continuous_hgd(df):
    """
    Rescales and implements the Heterogeneous Gower Distance (HGD) for continuous-continuous variable pairs.

    Parameters:
    df (DataFrame): The DataFrame containing the continuous variables.

    Returns:
    float: The HGD distance for the continuous variables.
    """
    df = rescale_continuous_continuous(df)

    sj = []
    max_value = np.max(np.abs(df[df != -1].iloc[0] - df[df != -1].iloc[1]))
    if np.isnan(max_value):
        max_value = 0

    pat_considered = 0
    for i in range(df.shape[1]):
        if (df.iloc[0, i] >= 0) and (df.iloc[1, i] >= 0):
            sj.append(1 - ((np.abs(df.iloc[0,i] - df.iloc[1,i])) / (max_value)))
            pat_considered += 1

    try:
        output = 1 - (1/pat_considered) * np.sum(sj)
    except ZeroDivisionError:
        output = 0

    return output


def hgd_distance(feature1, feature2, f1, f2, binary, continuous):
    """
    Computes the Heterogeneous Gower Distance (HGD) between two features based on their types (binary or continuous).

    Parameters:
    feature1 (ndarray): First feature array.
    feature2 (ndarray): Second feature array.
    f1, f2 (str): Feature types (binary or continuous).
    binary (list): List of binary features.
    continuous (list): List of continuous features.

    Returns:
    float: The HGD distance between the two features.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if f1 in binary and f2 in binary:
        df = pd.DataFrame(np.array([feature1, feature2]))
        hgd_distance = binary_binary_hgd(df)

    elif (f1 in binary and f2 in continuous) or (f1 in continuous and f2 in binary):
        b, c = (feature1, feature2) if f1 in binary else (feature2, feature1)
        df = pd.DataFrame(np.array([b, c]))
        hgd_distance = binary_continuous_hgd(df)

    elif f1 in continuous and f2 in continuous:
        df = pd.DataFrame(np.array([feature1, feature2]))
        hgd_distance = continuous_continuous_hgd(df)

    else:
        raise NameError('Hi, we have a problem! Stop!')

    return hgd_distance
