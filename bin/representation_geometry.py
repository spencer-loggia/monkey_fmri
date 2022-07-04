from typing import Union, List

import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import nibabel as nib
import graspologic as gr
import sklearn as sk
from scipy.stats import spearmanr

_distance_metrics_ = ['euclidian', 'pearson', 'spearman', 'dot', 'cosine']


def _pad_to_cube(arr: np.ndarray, time_axis=3):
    """
    makes spatial dims have even size.
    :param arr:
    :param time_axis:
    :return:
    """
    if time_axis < np.ndim(arr):
        size = max(arr.shape[:time_axis] + arr.shape[time_axis+1:])
        print(size)
    else:
        size = max(arr.shape)
        # print('ruhroh')

    ax_pad = [0] * np.ndim(arr)
    for i in range(len(ax_pad)):
        if i != time_axis:
            ideal = (size - arr.shape[i]) / 2
            ax_pad[i] = (int(np.floor(ideal)), int(np.ceil(ideal)))
        else:
            ax_pad[i] = (0, 0)
    arr = np.pad(arr, ax_pad, mode='constant', constant_values=(0, 0))
    return arr

def _dot_pdist(arr: torch.Tensor, normalize=False):
    """
    arr should be 2D < observations(v) x conditions (k) >
    if normalixe is true, equivilant to pairwise cosine similarity
    :param arr:
    :return:
    """
    if len(arr.shape) == 1:
        arr = arr.reshape(1, arr.shape[0])
    k = arr.shape[1]
    if normalize:
        arr = arr / arr.norm(dim=0)[None, :]
    outer = arr.T @ arr  # k x k
    indices = torch.triu_indices(k, k, offset=1)
    return outer[indices[0], indices[1]]


def _pearson_pdist(arr: torch.Tensor):
    k = arr.shape[1]
    coef = torch.corrcoef(arr)
    indices = torch.triu_indices(k, k, offset=1)
    return coef[indices[0], indices[1]]


def dissimilarity(beta: torch.Tensor, metric='dot'):
    if len(beta.shape) != 2:
        raise IndexError("beta should be 2 dimensional and have observations on dim 0 and conditions on dim 1")
    if metric not in _distance_metrics_:
        raise ValueError('metric must be one of ' + str(_distance_metrics_))
    elif metric == 'dot':
        rdm = _dot_pdist(beta, normalize=False)
    elif metric == 'cosine':
        rdm = _dot_pdist(beta, normalize=True)
    elif metric == 'pearson':
        rdm = _pearson_pdist(beta)
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


def pairwise_rsa(beta: torch.Tensor, atlas: torch.Tensor, min_roi_dim=5, ignore_atlas_base=True, metric='cosine'):
    if type(beta) is np.ndarray:
        beta = torch.from_numpy(beta)
    if type(atlas) is np.ndarray:
        atlas = torch.from_numpy(atlas)
    atlas = atlas.int()
    unique = torch.unique(atlas)
    if ignore_atlas_base:
        # omit index 0 (unclassified)
        unique = unique[1:]
    unique_filtered = []
    roi_dissimilarity = []
    for roi_id in unique:
        roi_betas = beta[atlas == roi_id]
        if len(roi_betas) > min_roi_dim:
            unique_filtered.append(roi_id)
            rdm = dissimilarity(roi_betas, metric=metric)
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


def pca(betas: torch.Tensor, brain_mask=None, n_components=2, noisy=False):
    betas = betas[brain_mask]
    if n_components == -1:
        n_components = betas.shape[-1]
        if noisy:
            n_components -= 1
    if len(betas.shape) > 2:
        betas = betas.reshape(-1, betas.shape[-1])
    cov = torch.cov(betas.T)
    eig_vals, eig_vecs = torch.linalg.eig(cov)
    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real
    sort_idx = torch.argsort(eig_vals, descending=True)
    eig_vecs = eig_vecs[:, sort_idx]
    eig_vals = eig_vals[sort_idx]
    eig_vecs = eig_vecs * eig_vals
    if noisy:
        proj_mat = eig_vecs[:, 1:n_components+1]
    else:
        proj_mat = eig_vecs[:, :n_components]
    projected_betas = betas @ proj_mat
    return projected_betas, eig_vecs, eig_vals


class LogisticDecoder:
    """
    Class designed to fit a single task on a singe voxel set (e.g. roi)
    """
    def __init__(self, in_dim, out_dim):
        """
        Constructs a new linear decoder
        :param in_dim:
        :param out_dim:
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.beta = None
        self.softmax = torch.nn.Softmax(dim=1)

    def _reshape_time_course(self, X):
        X = X.reshape(-1, X.shape[-1]).T  # voxels x n_samples
        if X.shape[0] != self.in_dim:
            raise ValueError("time course voxels flat must be the shape of in_dim attribute.")
        return X

    def get_target_loss_stats(self, y_target, y_hat):
        loss = torch.nn.CrossEntropyLoss()
        c_entropy = loss(y_hat, y_target)
        pred_labels = torch.argmax(y_hat, dim=1)
        confusion = confusion_matrix(y_target.detach().numpy(), pred_labels.detach().numpy(),
                                     labels=np.arange(self.out_dim))
        return pred_labels, c_entropy, confusion

    def fit(self, X: torch.Tensor, targets: torch.Tensor, optim_threshold=1e-9, cutoff_epoch=10000):
        """
        fit (compute beta coefficients for) this linear model.
        :param X:
        :param optim_threshold:
        :param targets: target class ids of length n_blocks / n_samples.
        :return: beta coefficients
        """
        # reshape to n_blocks x all
        X = self._reshape_time_course(X)
        target_arr = torch.nn.functional.one_hot(targets, num_classes=self.out_dim)  # n_blocks x k
        self.beta = torch.nn.Parameter(torch.inverse(X.T @ X) @ X.T @ target_arr.double())
        optimizer = torch.optim.Adam(lr=.01, params=[self.beta])
        cross_entropy = torch.nn.CrossEntropyLoss()
        print("Initialized Logistic Optimizer...")
        grad_dt = torch.inf
        epoch = 0
        while grad_dt > optim_threshold and epoch < cutoff_epoch:
            optimizer.zero_grad()
            old_beta = self.beta.data.detach().clone()
            lin_scores = X @ self.beta
            class_probs = self.softmax(lin_scores)
            class_loss = cross_entropy(class_probs, targets)
            class_loss.backward()
            optimizer.step()
            grad_dt = (torch.linalg.norm((old_beta.flatten() - self.beta.data.flatten())).detach().clone()) / len(old_beta.flatten())
            print("Epoch", epoch, "gradient delta", grad_dt.item())
            epoch += 1
        return self.beta

    def predict(self, X: torch.Tensor, targets: Union[None, torch.Tensor] = None):
        """
        Predict class labels for data using the current model. If targets are passed, will also compute cross entropy
        and a confusion matrix on the targets.
        :param time_course:
        :param targets:
        :return: predicted classes for each sample, cross entropy and confusion matrix if targets were passed.
        """
        X = self._reshape_time_course(X)
        y_hat = self.softmax(X @ self.beta)  # n_blocks x k
        if targets is not None:
            return self.get_target_loss_stats(targets, y_hat)
        return torch.argmax(y_hat, dim=0)


class ROIDecoder:
    """
    A class to fit a linear decoder in many rois and compare.
    """

    def __init__(self, atlas: torch.Tensor, roi_lookup, out_dim):
        """

        :param atlas: atlas of rois to look at in the brain
        :param roi_lookup: dictionary mapping atlas keys to roi names
        :param in_dim: in dimmensioality of model
        :param out_dim: out dimmensionality of model (number of classes)
        """
        self.atlas: torch.Tensor = atlas
        self.roi_lookup: dict = roi_lookup
        self.out_dim: int = out_dim
        self.models: List[Union[None, LogisticDecoder]] = [None for _ in range(len(roi_lookup))]

    def fit(self, X, targets, optim_threshold=1e-9, cutoff_epoch=10000):
        """
        :param X: (spatial0, spatial1, spatial2, samples )
        :param targets:
        :return:
        """
        roi_idxs = self.roi_lookup.keys()
        for i, roi_idx in enumerate(roi_idxs):
            roi_time_course = X[self.atlas == roi_idx]
            roi_time_course = roi_time_course.reshape(roi_time_course.shape[0], -1)
            roi_model = LogisticDecoder(roi_time_course.shape[1], self.out_dim)
            roi_model.fit(roi_time_course, targets, optim_threshold=optim_threshold, cutoff_epoch=cutoff_epoch)
            self.models[i] = roi_model

    def predict(self, X, targets):
        predict_result = []
        roi_idxs = self.roi_lookup.keys()
        for i, roi_idx in enumerate(roi_idxs):
            roi_x = X[self.atlas == roi_idx]
            roi_x = roi_x.reshape(roi_x.shape[0], -1)
            res = self.models[i].predict(roi_x, targets)
            predict_result.append(res)
        return predict_result


class SearchLightDecoder:

    def __init__(self, brain_mask, roi_loookup, out_dim, kernel=4, dev='cuda:0'):
        self.kernel = kernel
        self.out_dim = out_dim
        self.lookup = roi_loookup
        self.device = torch.device(dev)
        self.softmax = torch.nn.Softmax(dim=1)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.conv1 = None
        self.conv2 = None

    def fit(self, X, targets, optim_threshold=1e-9, cutoff_epoch=5000, lr=.0001):
        X = _pad_to_cube(X) # spatial, spatial, spatial, batch
        X = torch.from_numpy(X.transpose((3, 0, 1, 2))).double()
        X = torch.unsqueeze(X, dim=1)  # batch, channels, spatial, spatial, spatial

        # DEBUG: DOWNSAMPLE: DEBUG ONLY
        # pool = torch.nn.MaxPool3d(3)
        # X = pool(X)

        print(X.shape)
        in_spatial = X.shape[2]
        stride = 1
        # computing padding that maintains spatial dims
        pad = (((in_spatial - 1) * stride) - in_spatial + self.kernel) / 2
        if not pad.is_integer() or pad >= self.kernel:
            print('kernel', self.kernel, "is invalid.")
            return
        pad = int(pad)
        print(pad)
        self.conv1 = torch.nn.Conv3d(kernel_size=self.kernel,
                                     in_channels=1,
                                     out_channels=16,
                                     padding=pad,
                                     dtype=torch.double,
                                     device=self.device)
        self.conv2 = torch.nn.Conv3d(kernel_size=1,
                                     in_channels=16,
                                     out_channels=self.out_dim,
                                     dtype=torch.double,
                                     device=self.device)
        optimizer = torch.optim.Adam(params=[self.conv1.weight, self.conv2.weight], lr=lr)
        loss_dt = torch.inf
        old_loss = torch.inf
        epoch = 0
        targets = targets.reshape(len(targets), 1, 1, 1)
        targets = torch.tile(targets, (1, in_spatial, in_spatial, in_spatial)).to(self.device)
        while loss_dt > optim_threshold and epoch < cutoff_epoch:
            optimizer.zero_grad()
            h = self.conv1(X)
            h = self.conv2(h)  # a (samples, classes, spatial1, spatial2, spatial3) tensor
            y_hat = self.softmax(h)
            ce = self.ce_loss(y_hat, targets)  # loss per hyper-voxel (voxel is actually result of linear convolution)
            loss = torch.mean(torch.pow(ce.flatten(), 1 / 3)) # 3rt the ce so we care more about making low examples lower
            loss.backward()
            optimizer.step()
            loss_dt = (old_loss - loss).detach().cpu().clone().item()
            old_loss = loss.detach().cpu().clone()
            print("CE on epoch", epoch, "is", old_loss.item())
            epoch += 1
        return ce.detach().cpu().clone()
