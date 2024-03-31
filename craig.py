from typing import Literal, Optional, Tuple, Dict

import numpy as np
from submodlib.functions.facilityLocation import FacilityLocationFunction


class CRAIGCoreset:
    """
    This class implements the coreset selection procedure described in
    "Coresets for Data-efficient Training of Machine Learning Models" and
    "Towards Sustainable Learning: Coresets for Data-efficient Deep Learning".
    This implementation improves upon the official implementation in
    https://github.com/baharanm/craig and https://github.com/BigML-CS-UCLA/CREST:
        - it uses an object-oriented approach for the implementation
        - it provides a better documentation
        - it guarantees that a coreset of a specific size is always output while in the original implementations there
        are cases where bigger coresets are extracted
        -  the indices of the samples selected for the coreset are sorted by their marginal gain in FL objective
        (largest gain first) for each class
        - it handles errors better
    """
    def select_coreset_class(self, B: int, c: int, X: np.ndarray, y: np.ndarray,
                             metric: Literal["euclidean", "cosine"] = "euclidean",
                             mode: Literal["dense", "sparse", "clustered"] = "dense", num_n: Optional[int] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Select a coreset for a given class out of a given dataset by greedily optimising a submodular set cover problem
        with facility location function as described in
        "Coresets for Data-efficient Training of Machine Learning Models" and
        "Towards Sustainable Learning: Coresets for Data-efficient Deep Learning".

        Simply put, this function selects a subset of samples for a given class so that the weighted gradient sum of
        those samples is as similar as possible to the gradient sum of the samples of the given class in the original
        dataset. The weight of each sample in the coreset is the number of samples of the given class in the original
        dataset whose gradient is best approximated by the given sample in the coreset.

        This function outputs the indices of the samples of the given class that are selected for the coreset and
        their respective weights.

        :param B: size of the coreset for the given class, i.e. number of samples of the given class to select from the
            original dataset. If the number of samples of the given class in the original dataset is less than the size
            of the coreset, a ValueError is raised. If the number of samples of the given class in the original dataset
            is equal to the size of the coreset, all the class indices are returned along with their weights set to 1.
        :param c: target of the class
        :param X: gradient dataset with shape [N, d]. This dataset must contain the gradient of each sample in the
            original training dataset. Each gradient has size [d].
        :param y: targets of the samples in the original dataset with shape [N].
        :param metric: (optional) similarity metric to be used during the greedy submodular set cover optimisation.
            It can be `euclidean` or `cosine`.  Default is `euclidean`.
        :param mode: (optional) `dense`, `sparse` or `clustered`. It specifies whether the Facility Location function used
            to compute the coreset should operate in dense mode (using a dense similarity kernel), sparse mode (using a
            sparse similarity kernel) or clustered mode (evaluating over clusters). Default is `dense`.
        :param num_n: (optional) number of neighbors applicable for the sparse similarity kernel. It must be provided
            only if mode is `sparse`. Default is None.
        :return:  array with shape [B] of indices of the samples of the given class selected for the coreset,
            array with shape [B] of weightsof the respective samples selected for the coreset. The indices of the
            samples selected for the coreset are sorted by their marginal gain in FL objective (largest gain first).
        """
        class_indices = np.where(y == c)[0]
        X = X[class_indices]
        N = X.shape[0]
        # if the number of samples of the given class in the original dataset is less than the size of the coreset
        if N < B:
            raise ValueError(
                "The size of the coreset for the given class must be smaller than or equal to the number of"
                " samples of the given class in the original dataset")
        if N == B:
            return class_indices, np.ones(N, dtype=np.float32)

        if mode == "dense":
            num_n = None

        obj = FacilityLocationFunction(n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n)

        greedyList = obj.maximize(budget=B, optimizer="LazyGreedy", stopIfZeroGain=False, stopIfNegativeGain=False,
                                  verbose=False)
        order = list(map(lambda x: x[0], greedyList))
        order = np.asarray(order, dtype=np.int64)

        S = obj.sijs
        sz = np.zeros(B, dtype=np.float64)

        for i in range(N):
            if np.max(S[i, order]) <= 0:
                continue
            sz[np.argmax(S[i, order])] += 1
        sz[np.where(sz == 0)] = 1

        return class_indices[order], sz

    def select_coreset(self, B: int, X: np.ndarray, metric: Literal["euclidean", "cosine"] = "euclidean",
                       y: Optional[np.ndarray] = None, equal_num: bool = False,
                       mode: Literal["dense", "sparse", "clustered"] = "dense", num_n: Optional[int] = None) \
            -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        """
        Select a coreset out of a given dataset by greedily optimising a submodular set cover problem with facility
        location function as described in "Coresets for Data-efficient Training of Machine Learning Models" and
        "Towards Sustainable Learning: Coresets for Data-efficient Deep Learning".

        If the dataset contains R classes, where R > 1, the coreset selection process involves selecting R corsets, one
        for each class, and then merging them together. The size of the coreset of each class is determined by the
        `equal_num` parameter. Note that the size of the whole coreset is determined by the parameter `B`

        Simply put, this function selects a subset of samples so that the weighted gradient sum of those samples is as
        similar as possible to the gradient sum of the original dataset. The weight of each sample in the coreset is the
        number of samples in the original dataset whose gradient is best approximated by the given sample in the
        coreset.

        This function outputs all the indices of the samples that are selected for the coreset, their respective weights
        and also a dictionary containing class targets as keys and a tuple containing the indices (sorted by their
        marginal gain in FL objective (largest gain first)) selected for the given class and their respective weights
        as values.

        :param B: size of the whole coreset, i.e. total number of samples to select from the dataset. A ValueError is
            raised if the size of the coreset is greater than the size of the original dataset.
        :param X: gradient dataset with shape [N, d]. This dataset must contain the gradient of each sample in the
            original training dataset. Each gradient has size [d].
        :param metric: (optional) similarity metric to be used during the greedy submodular set cover optimisation.
            It can be `euclidean` or `cosine`.  Default is `euclidean`.
        :param y: (optional) targets of the samples in the original dataset with shape [N]. If None, the coreset is
            computed over the whole gradient dataset assuming that the original dataset only contains one class.
            Default is None.
        :param equal_num: (optional) if True, the coreset contains an equal number of samples from each class.
            Simply put, the coreset contains the concatenation of C coresets of size B/C, where C is the number of
            classes. If there are classes whose number of samples in the original dataset is less than B/C, the size of
            the coreset of these classes is set to the number of samples of those classes in the original dataset and
            the rest of samples needed to get an overall coreset of size B is taken evenly from the other classes.
            If False, the coreset contains samples from each class in proportion to the number of samples the classes
            have in the original dataset. Default is False.
        :param mode: (optional) `dense`, `sparse` or `clustered`. It specifies whether the Facility Location function
            used to compute the coreset should operate in dense mode (using a dense similarity kernel), sparse mode
            (using a sparse similarity kernel) or clustered mode (evaluating over clusters). Default is `dense`.
        :param num_n: (optional) number of neighbors applicable for the sparse similarity kernel. It must be provided
            only if mode is `sparse`. Default is None.
        :return: array with shape [B] of indices of the samples selected for the coreset, array with shape [B] of
            weights of the respective samples selected for the coreset, a dictionary containing class targets as keys
            and a tuple containing the indices (sorted by their marginal gain in FL objective (largest gain first))
            selected for the given class and their respective weights as values.
        """
        N = X.shape[0]
        # if the size of the original dataset is less than the size of the coreset
        if N < B:
            raise ValueError("The size of the coreset must be smaller than or equal to the size of the coreset")
        if y is None:
            y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
        classes = np.unique(y)
        classes = classes.astype(np.int32).tolist()
        C = len(classes)  # number of classes

        if equal_num:
            class_nums = np.asarray([sum(y == c) for c in classes], dtype=np.int32)
            num_per_class = int(np.ceil(B / C)) * np.ones(C, dtype=np.int32)
            minority = class_nums < np.ceil(B / C)
            # if there are classes whose number of samples in the original dataset is less than B/C, the size of the
            # coreset of these classes is set to the number of samples of those classes in the original dataset and
            # the rest of samples needed to get an overall coreset of size B is taken evenly from the other classes
            if sum(minority) > 0:
                num_per_class[minority] = class_nums[minority]
                extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
                minority = (class_nums - num_per_class) <= 0
                while extra > 0:
                    index_not_minority = np.arange(C, dtype=np.int32)[~minority]
                    index = index_not_minority[np.argmin(num_per_class[~minority])]
                    num_per_class[index] += 1
                    minority = (class_nums - num_per_class) <= 0
                    extra -= 1
        else:
            num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
            diff = np.sum(num_per_class) - B
            # if the sum of the samples selected for each class is greater than the whole size of the coreset, reduce by
            # one the size of the coreset selected for the first `diff` classes. This can occur due to the ceiling
            # operator
            if diff > 0:
                num_per_class -= np.concatenate((np.ones(diff, dtype=np.int32), np.zeros(C - diff, dtype=np.int32)))

        order_mg_all, cluster_sizes_all = zip(*map(lambda c: self.select_coreset_class(num_per_class[c[0]], c[1], X, y,
                                                                                       metric, mode, num_n),
                                                   enumerate(classes)))
        order_mg = np.concatenate(order_mg_all, dtype=np.int32)
        weights_mg = np.concatenate(cluster_sizes_all, dtype=np.float32)
        order_weights_classes = {c: (order_mg_all[ind], cluster_sizes_all[ind]) for ind, c in enumerate(classes)}

        return order_mg, weights_mg, order_weights_classes
