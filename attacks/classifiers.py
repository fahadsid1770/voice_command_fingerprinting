"""
ML classifiers for VCFP (Voice Command Fingerprinting) attack.

This module provides functions to train and evaluate classifiers for
identifying voice commands from network traffic features.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from collections import defaultdict


def train_bayes(X_train, y_train):
    """
    Train Gaussian Naive Bayes classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained GaussianNB classifier
    """
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf


def train_svm(X_train, y_train):
    """
    Train SVM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained SVM classifier
    """
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    return clf


def jaccard_similarity(set1, set2):
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
    
    Returns:
        Jaccard similarity score (0 to 1)
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def train_jaccard(X_train, y_train):
    """
    Train Jaccard classifier (stores training sets for similarity comparison).
    
    Args:
        X_train: Training features (sets of packet sizes)
        y_train: Training labels
    
    Returns:
        Dictionary mapping labels to lists of training sets
    """
    train_sets = {}
    for i, label in enumerate(y_train):
        if label not in train_sets:
            train_sets[label] = []
        train_sets[label].append(X_train[i])
    return train_sets


def predict_jaccard(clf, X_test):
    """
    Predict using Jaccard similarity.
    
    Args:
        clf: Trained Jaccard classifier (dict of training sets)
        X_test: Test features (sets of packet sizes)
    
    Returns:
        Array of predicted labels
    """
    predictions = []
    for test_set in X_test:
        best_label = None
        best_sim = -1
        
        for label, train_sets in clf.items():
            for train_set in train_sets:
                sim = jaccard_similarity(test_set, train_set)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label
        
        predictions.append(best_label)
    
    return np.array(predictions)


def evaluate_classifier(clf, X_test, y_test, method='standard'):
    """
    Evaluate classifier performance.
    
    Args:
        clf: Trained classifier
        X_test: Test features
        y_test: True labels
        method: 'standard' for sklearn classifiers, 'jaccard' for Jaccard
    
    Returns:
        accuracy: Classification accuracy
        avg_rank: Average rank (for semantic distance, currently 0 for standard)
    """
    if method == 'jaccard':
        y_pred = predict_jaccard(clf, X_test)
    else:
        y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    correct = np.sum(y_pred == y_test)
    accuracy = correct / len(y_test)
    
    # Calculate average rank (for semantic distance)
    avg_rank = 0
    
    return accuracy, avg_rank
