import abc
import random

from grokk_replica.datasets import AbstractDataset


class AbstractDatasetCont(AbstractDataset):
    @abc.abstractmethod
    def fetch_output(self, a, b) -> float:
        pass

    def fetch_example(self, idx):
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        c = self.fetch_output(a, b)
        return self.encode([a, b]), c, None

    def fetch_train_example(self):
        idx = random.choice(self.train_pairs)
        return self.fetch_example(idx)

    def fetch_val_example(self):
        idx = random.choice(self.val_pairs)
        return self.fetch_example(idx)


class SumDatasetCont(AbstractDatasetCont):
    def __init__(self, p, frac_train):
        super().__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p

    def fetch_output(self, a, b):
        return (a + b) / self.p


class SubDatasetCont(AbstractDatasetCont):
    def __init__(self, p, frac_train):
        super().__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p

    def fetch_output(self, a, b):
        return (a - b) / self.p
