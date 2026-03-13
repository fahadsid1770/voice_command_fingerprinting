"""
Cross-validation for VCFP (Voice Command Fingerprinting) attack.

This module provides functions to perform stratified n-fold cross-validation
for evaluating classifier performance.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from attacks.classifiers import train_bayes, train_svm, train_jaccard, evaluate_classifier


def n_fold_cross_validation(X, y, n_folds=5, method='bayes'):
    """
    Perform stratified n-fold cross-validation.
    
    Args:
        X: Feature matrix
        y: Labels
        n_folds: Number of folds
        method: Classification method ('bayes', 'svm', 'jaccard')
    
    Returns:
        avg_accuracy: Average accuracy across folds
        avg_rank: Average rank across folds
        fold_accuracies: List of accuracies per fold
    """

    # Count samples per class and handle classes with insufficient samples
    class_counts = Counter(y)
    min_samples = min(class_counts.values())
    
    if min_samples < n_folds:
        # Find classes with fewer samples than n_folds
        insufficient_classes = [cls for cls, count in class_counts.items() if count < n_folds]
        print(f"WARNING: The following classes have fewer than {n_folds} samples: {insufficient_classes}")
        print(f"Filtering out these classes before cross-validation...")
        
        # Create mask to filter out insufficient classes
        mask = np.array([y_i not in insufficient_classes for y_i in y])
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        # Update class counts after filtering
        new_class_counts = Counter(y_filtered)
        new_min_samples = min(new_class_counts.values())
        new_n_folds = min(n_folds, new_min_samples)
        
        print(f"After filtering: {len(X_filtered)} samples, {len(new_class_counts)} classes")
        print(f"Adjusted n_folds from {n_folds} to {new_n_folds}")
        
        X = X_filtered
        y = y_filtered
        n_folds = new_n_folds

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_ranks = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train classifier
        if method == 'bayes':
            clf = train_bayes(X_train, y_train)
        elif method == 'svm':
            clf = train_svm(X_train, y_train)
        elif method == 'jaccard':
            clf = train_jaccard(X_train, y_train)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Evaluate
        eval_method = 'jaccard' if method == 'jaccard' else 'standard'
        accuracy, rank = evaluate_classifier(clf, X_test, y_test, eval_method)
        fold_accuracies.append(accuracy)
        fold_ranks.append(rank)
        
        print(f"Fold {fold+1}: Accuracy = {accuracy:.4f}")
    
    avg_accuracy = np.mean(fold_accuracies)
    avg_rank = np.mean(fold_ranks)
    
    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    print(f"Average Rank: {avg_rank:.4f}")
    
    return avg_accuracy, avg_rank, fold_accuracies
