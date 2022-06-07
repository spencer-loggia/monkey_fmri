import torch
import numpy as np
import nibabel as nib
import graspologic as gr
import sklearn as sk
from scipy.stats import spearmanr

_distance_metrics_ = ['euclidian', 'pearson', 'spearman', 'dot', 'cosine']


def _dot_pdist(arr: torch.Tensor, normalize=False):
    """
    arr should be 2D < observations(v) x conditions (k) >
    if normalixe is true, equivilant to pairwise cosine similarity
    :param arr:
    :return:
    """
    k = arr.shape[1]
    if normalize:
        arr = arr / arr.norm(dim=0)[None, :]
    outer = arr.T @ arr  # k x k
    indices = torch.triu_indices(k, k, offset=1)
    return outer[indices[0], indices[1]]


def dissimilarity(beta: torch.Tensor, metric='dot'):
    if len(beta.shape) != 2:
        raise IndexError("beta should be 2 dimensional and have observations on dim 0 and conditions on dim 1")
    if metric not in _distance_metrics_:
        raise ValueError('metric must be one of ' + str(_distance_metrics_))
    elif metric == 'dot':
        rdm = _dot_pdist(beta, normalize=False)
    elif metric == 'cosine':
        rdm = _dot_pdist(beta, normalize=True)
    else:
        raise NotImplementedError
    return rdm


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


def searchlight_net(mean: torch.Tensor, cov: torch.Tensor,):
    raise NotImplementedError


def pairwise_rsa(beta: torch.Tensor, atlas: torch.Tensor, min_roi_dim=5):
    if type(beta) is np.ndarray:
        beta = torch.from_numpy(beta)
    if type(atlas) is np.ndarray:
        atlas = torch.from_numpy(atlas)
    atlas = atlas.int()
    unique = torch.unique(atlas)[1:]  # omit index 0 (unclassified)
    unique_filtered = []
    roi_dissimilarity = []
    for roi_id in unique:
        roi_betas = beta[atlas == roi_id]
        if len(roi_betas) > min_roi_dim:
            unique_filtered.append(roi_id)
            rdm = dissimilarity(roi_betas, metric='dot')
            roi_dissimilarity.append(rdm)
    adjacency = torch.zeros([len(unique_filtered), len(unique_filtered)])
    pvals = torch.zeros([len(unique_filtered), len(unique_filtered)])
    for i, src in enumerate(unique_filtered):
        for j, target in enumerate(unique_filtered[i + 1:]):
            j_t = j + i + 1
            print("Computed Correlation ", src, ", ", target)
            spear_corr, pval = spearmanr(roi_dissimilarity[i], roi_dissimilarity[j_t])
            # dis_i = torch.log_softmax(roi_dissimilarity[j_t], dim=0)[None, :]
            # dis_j = torch.log_softmax(roi_dissimilarity[i], dim=0)[None, :]
            adjacency[i, j_t] = spear_corr
            adjacency[j_t, i] = spear_corr
            pvals[i, j_t] = pval
            pvals[j_t, i] = pval
    return adjacency, unique_filtered, roi_dissimilarity, pvals

