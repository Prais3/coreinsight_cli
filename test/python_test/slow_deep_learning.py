import numpy as np


def slow_cosine_similarity_matrix(X):
    """
    Naively compute pairwise cosine similarity matrix.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
    
    Returns:
        sim_matrix: (n_samples, n_samples)
    """
    n = X.shape[0]
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # redundant norm computation every time (slow!)
            norm_i = np.sqrt(np.sum(X[i] ** 2))
            norm_j = np.sqrt(np.sum(X[j] ** 2))
            
            if norm_i == 0 or norm_j == 0:
                sim = 0
            else:
                sim = np.dot(X[i], X[j]) / (norm_i * norm_j)
            
            sim_matrix[i, j] = sim
    
    return sim_matrix


def slow_contrastive_loss(X, labels, margin=1.0):
    """
    Naive contrastive loss using pairwise cosine similarity.
    
    Args:
        X: (n_samples, n_features)
        labels: (n_samples,) integer class labels
        margin: float
    
    Returns:
        loss: scalar
    """
    sim_matrix = slow_cosine_similarity_matrix(X)
    n = X.shape[0]
    
    loss = 0.0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            if labels[i] == labels[j]:
                # positive pair → want similarity high
                loss += (1 - sim_matrix[i, j]) ** 2
            else:
                # negative pair → want similarity low
                loss += max(0, sim_matrix[i, j] - margin) ** 2
    
    return loss / (n * (n - 1))