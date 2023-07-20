import os
import json

import numpy as np
import torch
from torch.utils.data import Sampler, RandomSampler, BatchSampler, DataLoader, Dataset


class GenotypeToPhenotypeDataset(Dataset):
    """
    Dataset for genotype to phenotype prediction. Can be used for both labeled and unlabeled (without phenotypes) datasets.

    The SNP and phenotype data (if labeled) should be stored in a .npz file containing the following keys:
    - x: SNP data as a n_samples x n_variants int8 2D array containing values in {0,1,2}.
    - y: Only present if the dataset is labeled. Phenotype data as a n_samples int8 1D array containing values in {0,1,...,n_classes-1}.
    The row order should match between x and y. The same number of samples should be present in both x and y.

    Parameters
    ----------
    root : str
        Root directory of the dataset.

    filename : str
        Filename of the .npz file containing the SNP and phenotype data (if labeled).

    """

    def __init__(self, root_path: str, filename: str):
        with np.load(os.path.join(root_path, filename)) as data:
            self.snps = torch.from_numpy(data['x'])
            self.phenotypes = None
            if 'y' in data:
                self.phenotypes = torch.from_numpy(data['y'])
                assert self.snps.shape[0] == self.phenotypes.shape[0]

        self.num_classes = int(self.phenotypes.max().item()) + 1 if self.phenotypes is not None else None
        self.num_samples, self.num_variants = self.snps.shape

    def __len__(self):
        """
        Returns
        -------
        n_samples : int
            Number of samples in the dataset
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns
        -------
        snps : torch.Tensor
            SNPs for the idx-th sample as a 1D tensor of size n_variants

        phenotypes : torch.Tensor | None
            Phenotypes for the idx-th sample as a 1D tensor of size 1.
            -1 if the dataset is unlabeled.
        """
        if self.phenotypes is not None:
            return self.snps[idx], self.phenotypes[idx]
        else:
            return self.snps[idx], -1


class InfiniteSampler(Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self._sampler = sampler

    def __iter__(self):
        while True:
            for batch in self._sampler:
                yield batch


class InfiniteDataLoader:
    """
    Parameters
    ----------
    dataset : Dataset
        Dataset to be loaded.

    batch_size : int
        Batch size for data loading and training / evaluation.
    """

    def __init__(self, dataset, batch_size):
        sampler = RandomSampler(dataset, replacement=False)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)  # maybe leave drop_last as a parameter for batchnorm
        data_loader = DataLoader(dataset, batch_sampler=InfiniteSampler(batch_sampler), pin_memory=True)
        self._infinite_iterator = iter(data_loader)

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0


class DataLoaders:
    """
    DataLoaders for semi-supervised domain adaptation for genotype to phenotype prediction.

    The SNP and phenotype data should be stored in .npz files containing the following keys:
    - x: SNP data as a n_samples x n_variants int8 2D array containing values in {0,1,2}.
    - y: Only present if the dataset is labeled. Phenotype data as a n_samples int8 1D array containing values in {0,1,...,n_classes-1}.
    The row order should match between x and y. The same number of samples and variants should be present in all the files.

    Parameters
    ----------
    dataset_json_path : str
        Path to the .json file storing the dataset path information.

    batch_size : int
        Batch size for data loading and training / evaluation.
    """

    def __init__(self, dataset_json_path: str, batch_size: int):

        with open(dataset_json_path) as f:
            data = json.load(f)
        
        print("Creating datasets...", flush=True)
        source_labeled_train_dataset = GenotypeToPhenotypeDataset(data["root"], data["source"]["train"])
        target_labeled_train_dataset = GenotypeToPhenotypeDataset(data["root"], data["target"]["train"])
        target_unlabeled_train_dataset = GenotypeToPhenotypeDataset(data["root"], data["target"]["unlabeled"])
        target_labeled_validation_dataset = GenotypeToPhenotypeDataset(data["root"], data["target"]["validation"])
        target_labeled_test_dataset = GenotypeToPhenotypeDataset(data["root"], data["target"]["test"])

        assert source_labeled_train_dataset.num_variants == target_labeled_train_dataset.num_variants == target_unlabeled_train_dataset.num_variants == target_labeled_validation_dataset.num_variants == target_labeled_test_dataset.num_variants, "The number of variants should be the same for all the datasets."
        assert source_labeled_train_dataset.num_classes == target_labeled_train_dataset.num_classes == target_labeled_validation_dataset.num_classes == target_labeled_test_dataset.num_classes, "The number of classes should be the same for all the datasets."
        assert source_labeled_train_dataset.num_variants is not None and source_labeled_train_dataset.num_classes is not None
        self.in_features = source_labeled_train_dataset.num_variants
        self.out_features = source_labeled_train_dataset.num_classes

        # Train data loaders
        print("Creating data loaders for training...", flush=True)
        self.source_labeled_train_inf = InfiniteDataLoader(dataset=source_labeled_train_dataset, batch_size=batch_size)
        self.s_iter = iter(self.source_labeled_train_inf)
        self.target_labeled_train_inf = InfiniteDataLoader(dataset=target_labeled_train_dataset, batch_size=batch_size)
        self.l_iter = iter(self.target_labeled_train_inf)
        self.target_unlabeled_train_inf = InfiniteDataLoader(dataset=target_unlabeled_train_dataset, batch_size=batch_size)
        self.u_iter = iter(self.target_unlabeled_train_inf)
        self.target_unlabeled_train = DataLoader(dataset=target_unlabeled_train_dataset, batch_size=batch_size)  # for ProtoClassifier
        
        # Validation and test data loaders
        print("Creating data loaders for validation and test...", flush=True)
        self.target_labeled_validation = DataLoader(dataset=target_labeled_validation_dataset, batch_size=batch_size)
        self.target_labeled_test = DataLoader(dataset=target_labeled_test_dataset, batch_size=batch_size)

        print("DataLoaders initialized", flush=True)

    def __iter__(self):
        while True:
            sx, sy = next(self.s_iter)
            sx, sy = sx.float().cuda(), sy.long().cuda()

            tx, ty = next(self.l_iter)
            tx, ty = tx.float().cuda(), ty.long().cuda()

            ux, _ = next(self.u_iter)
            ux = ux.float().cuda()

            yield (sx, sy), (tx, ty), ux

    def __len__(self):
        return 0
