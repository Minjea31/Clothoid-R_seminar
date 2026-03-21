import torch
import torch.nn.utils.prune as prune
from torch import nn
from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F
from scipy.spatial import ConvexHull

class GMStructured(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'structured'

    def __init__(self, amount, dim=-1):
        prune._validate_pruning_amount_init(amount)
        self.amount = amount
        self.dim = dim
        

    def compute_mask(self, t, default_mask, module_name):
        r"""Computes and returns a mask for the input tensor ``t``.
                Starting from a base ``default_mask`` (which should be a mask of ones
                if the tensor has not been pruned yet), generate a mask to apply on
                top of the ``default_mask`` by zeroing out the channels along the
                specified dim with the lowest L\ ``n``-norm.

                Args:
                    t (torch.Tensor): tensor representing the parameter to prune
                    default_mask (torch.Tensor): Base mask from previous pruning
                        iterations, that need to be respected after the new mask is
                        applied.  Same dims as ``t``.

                Returns:
                    mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

                Raises:
                    IndexError: if ``self.dim >= len(t.shape)``
                """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        prune._validate_structured_pruning(t)
        # Check that self.dim is a valid dim to index t, else raise IndexError
        prune._validate_pruning_dim(t, self.dim)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the dim to prune
        tensor_size = t.shape[self.dim]
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        # Structured pruning prunes entire channels so we need to know the
        # L_n norm along each channel to then find the topk based on this
        # metric
        # norm = prune._compute_norm(t, self.n, self.dim)
        t_2d = t.view(t.size()[0], -1)
        C = t_2d.shape[0]    
        
        if t_2d.shape[0] < 64:
            # 모든 채널의 평균 계산 (L2 기준 사용, 중심은 실제 채널로 스냅)
            X_np = t_2d.detach().cpu().numpy()        # (C, D)
            mean_all = X_np.mean(axis=0)              # (D,)
            nearest_idx = np.argmin(np.linalg.norm(X_np - mean_all, axis=1))
            center_np = X_np[nearest_idx]             # (D,)

            center_tensor = torch.tensor(center_np, device=t.device, dtype=t.dtype)  # (D,)
            distances_tensor = torch.norm(t_2d - center_tensor, dim=1)
            tensor_sort = distances_tensor.argsort().detach().clone()

        elif 64 <= t_2d.shape[0] <= 128:
            X_np = t_2d.detach().cpu().numpy()
            K = 2
            N, D = X_np.shape
            _rng = np.random.default_rng(0)
            MAX_ITERS = 100

            # k-means++ 초기화 (L2)
            centers = np.empty((K, D), dtype=X_np.dtype)
            idx0 = _rng.integers(0, N)
            centers[0] = X_np[idx0]
            d2 = np.linalg.norm(X_np - centers[0], axis=1) ** 2
            for kk in range(1, K):
                probs = d2 / (d2.sum() + 1e-12)
                idx = _rng.choice(N, p=probs)
                centers[kk] = X_np[idx]
                d2 = np.minimum(d2, np.linalg.norm(X_np - centers[kk], axis=1) ** 2)

            labels = np.zeros(N, dtype=np.int64)
            prev_labels = None

            for it in range(MAX_ITERS):
                distances = np.linalg.norm(X_np[:, None, :] - centers[None, :, :], axis=2)  # (N, K)
                labels = np.argmin(distances, axis=1)

                for kk in range(K):
                    if not np.any(labels == kk):
                        counts = np.bincount(labels, minlength=K)
                        big = np.argmax(counts)
                        big_idx = np.where(labels == big)[0]
                        if big_idx.size > 0:
                            d_big = np.linalg.norm(X_np[big_idx] - centers[big], axis=1)
                            far_idx = big_idx[np.argmax(d_big)]
                            labels[far_idx] = kk

                new_centers = np.empty_like(centers)
                for kk in range(K):
                    members_idx = np.where(labels == kk)[0]
                    if members_idx.size == 0:
                        new_centers[kk] = centers[kk]
                        continue
                    members = X_np[members_idx]
                    mean_k = members.mean(axis=0)
                    nearest_local = np.argmin(np.linalg.norm(members - mean_k, axis=1))
                    new_centers[kk] = members[nearest_local]
                if prev_labels is not None and np.array_equal(labels, prev_labels):
                    centers = new_centers
                    break
                prev_labels = labels.copy()
                centers = new_centers

            centers_tensor = torch.tensor(centers, device=t.device, dtype=t.dtype)
            labels_tensor = torch.tensor(labels, device=t.device)
            assigned_centers = centers_tensor[labels_tensor]
            distances_tensor = torch.norm(t_2d - assigned_centers, dim=1)  # L2
            tensor_sort = distances_tensor.argsort().detach().clone()

        else:
            X_np = t_2d.detach().cpu().numpy()
            K = 4
            N, D = X_np.shape
            _rng = np.random.default_rng(0)
            MAX_ITERS = 100

            centers = np.empty((K, D), dtype=X_np.dtype)
            idx0 = _rng.integers(0, N)
            centers[0] = X_np[idx0]
            d2 = np.linalg.norm(X_np - centers[0], axis=1) ** 2
            for kk in range(1, K):
                probs = d2 / (d2.sum() + 1e-12)
                idx = _rng.choice(N, p=probs)
                centers[kk] = X_np[idx]
                d2 = np.minimum(d2, np.linalg.norm(X_np - centers[kk], axis=1) ** 2)

            labels = np.zeros(N, dtype=np.int64)
            prev_labels = None

            for it in range(MAX_ITERS):
                distances = np.linalg.norm(X_np[:, None, :] - centers[None, :, :], axis=2)
                labels = np.argmin(distances, axis=1)

                for kk in range(K):
                    if not np.any(labels == kk):
                        counts = np.bincount(labels, minlength=K)
                        big = np.argmax(counts)
                        big_idx = np.where(labels == big)[0]
                        if big_idx.size > 0:
                            d_big = np.linalg.norm(X_np[big_idx] - centers[big], axis=1)
                            far_idx = big_idx[np.argmax(d_big)]
                            labels[far_idx] = kk

                new_centers = np.empty_like(centers)
                for kk in range(K):
                    members_idx = np.where(labels == kk)[0]
                    if members_idx.size == 0:
                        new_centers[kk] = centers[kk]
                        continue
                    members = X_np[members_idx]
                    mean_k = members.mean(axis=0)
                    nearest_local = np.argmin(np.linalg.norm(members - mean_k, axis=1))
                    new_centers[kk] = members[nearest_local]

                if prev_labels is not None and np.array_equal(labels, prev_labels):
                    centers = new_centers
                    break
                prev_labels = labels.copy()
                centers = new_centers

                centers_tensor = torch.tensor(centers, device=t.device, dtype=t.dtype)
                labels_tensor = torch.tensor(labels, device=t.device)
                assigned_centers = centers_tensor[labels_tensor]
                distances_tensor = torch.norm(t_2d - assigned_centers, dim=1)  # L2
                tensor_sort = distances_tensor.argsort().detach().clone()

        # largest=True --> top k; largest=False --> bottom k
        # Keep the largest k channels along dim=self.dim
        topk = torch.topk(tensor_sort, k=nparams_tokeep, largest=True)

        keep_idx = topk.indices

        
        # topk will have .indices and .values

        # Compute binary mask by initializing it to all 0s and then filling in
        # 1s wherever topk.indices indicates, along self.dim.
        # mask has the same shape as tensor t
        def make_mask(t, dim, indices):
            # init mask to 0
            mask = torch.zeros_like(t)
            # e.g.: slc = [None, None, None], if len(t.shape) = 3
            slc = [slice(None)] * len(t.shape)
            # replace a None at position=dim with indices
            # e.g.: slc = [None, None, [0, 2, 3]] if dim=2 & indices=[0,2,3]
            slc[dim] = indices
            # use slc to slice mask and replace all its entries with 1s
            # e.g.: mask[:, :, [0, 2, 3]] = 1
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            mask = default_mask
        else:
            mask = make_mask(t, self.dim, keep_idx)
            mask *= default_mask.to(dtype=mask.dtype)
        return mask



def gm_structured(module, name, module_name, amount, dim, importance_scores=None):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
        >>> m = nn.Linear(3, 4)
        >>> foobar_unstructured(m, name='bias')
    """
    if name == None:
        pruned, _, mask = GMStructured.apply(module, name, module_name, amount, dim, importance_scores=importance_scores)
        return pruned, mask
    else:
        GMStructured.apply(module, name, module_name, amount, dim, importance_scores=importance_scores)
        return module
