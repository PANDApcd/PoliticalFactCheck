from collections import Counter
from scipy import sparse
from typing import Dict, Union
import numpy as np
import pandas as pd
import torch
from torch import nn
import tqdm
import math


def to_sparse_tensor(matrix: sparse.csr.csr_matrix) -> torch.Tensor:
    """Transform a sparse matrix into a tensor

    Args:
        matrix (sparse.csr.csr_matrix): A row-major sparse matrix

    Returns:
        (torch.Tensor): The Tensor representation of the sparse matrix
    """
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(
        np.mat([coo.row, coo.col]), coo.data, size=matrix.shape
    ).to(torch.float)


class Preprocessor(object):
    def __init__(
        self,
        voc: Dict[str, int] = None,
        min_frequency: int = 3,
    ):
        """A class to build the vocabulary and transform matrix based on it

        Attributes:
            voc(Dict[str, int]): The vocabulary set
            tokenizer(Callable): The function to tokenize the document
            min_frequency(int): The min frequency of one word
        """
        self.voc = voc if voc else dict()
        self.min_frequency = min_frequency

    def buildVocabulary(self, train_df: pd.DataFrame):
        """Build/expend the vocabulary from the dataframe

        Args:
            train_df(pd.DataFrame): The dataframe used to build the vocabulary
        """
        ct = Counter()
        index = 0
        for row in tqdm.tqdm(train_df["content"], desc="Build Voc"):
            ct.update(Counter(row))

        for k in sorted(ct.keys()):
            if ct[k] >= self.min_frequency:
                self.voc[k] = index
                index += 1
        self.ct = ct

    def buildMatrix(self, df: pd.DataFrame) -> sparse.csr.csr_matrix:
        """Build the document matrix from the dataframe

        Args:
            df(pd.DataFrame): The document to be built from

        Returns:
            (sparse.csr.csr_matrix): The rowwise compressed sparse matrix
        """
        if not isinstance(df, pd.DataFrame):
            return sparse.csr_matrix(df)
        data = list()
        indptr = [0]
        indices = list()
        for i, row in tqdm.tqdm(enumerate(df["content"]), desc="Build Matrix"):
            for word in row:
                if word in self.voc.keys():
                    data.append(1)
                    indices.append(self.voc[word])
            data.append(1)
            indices.append(len(self.voc))
            indptr.append(len(indices))
        matrix = sparse.csr_matrix((data, indices, indptr), dtype=int)
        return matrix


class LogisticRegression(nn.Module):
    def __init__(self, processor: int):
        """A two level neruo network for logistic regression classifier

        Args:
            processor(Preprocessor): A processor with vocabular set

        Attributes:
            processor(Preprocessor):
            layer_stack (nn.Sequential): The two level neuro network structure
        """
        super().__init__()
        self.processor = processor
        self.layer_stack = nn.Sequential(
            nn.Linear(len(self.processor.voc) + 1, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """The forward method for training"""
        return self.layer_stack(x).flatten()

    def get_token_weights(self):
        """Sort the words by their weights

        Returns:
            (List[float, str]): Sorted tokens list by their weight
        """
        pri = dict()
        for k, v in self.processor.voc.items():
            pri[v] = k
        return sorted(
            [
                (val, pri[i])
                for i, val in enumerate(
                    next(self.parameters())[0].detach().numpy().flatten()[:-1]
                )
            ],
            key=lambda x: -abs(x[0]),
        )


def predict(model: object, features: Union[pd.DataFrame, np.ndarray]):
    """Make prediction based on the model and input

    Args:
        model(object): The model to be used for prediction
        features(Union[pd.DataFrame, np.ndarray]): The feature dataframe
    """
    if isinstance(model, nn.Module):
        if not isinstance(features, torch.Tensor):
            features = to_sparse_tensor(model.processor.buildMatrix(features))
        output = model(features).detach().numpy().flatten()
    else:
        output = model(features)
    return pd.Series(output)


def train(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    model: LogisticRegression,
    lr: float = 1e-4,
    epoch_num: int = 5,
    epoch_step: int = 1000,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, object]:
    """Train the model through pytorch

    Args:
        train_X(np.ndarray): The feature matrix, each row is a feature
        train_Y(np.ndarray): the label column vector
        model(LogisticRegression): The model to be trained
        lr(float): The learning rate
        epoch_num(int): The number of epoch for training
        epoch_step(int): The number of step in one epoch
        optimizer(torch.optim.Optimizer): The optimizer used for training
        fig_path (str): The path to save the loss

    Returns:
        Dict[str, object]: The loss and F1 result
    """
    criterion = torch.nn.BCELoss()
    if not optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_Y = torch.from_numpy(train_Y).to(torch.float)
    indexes = np.arange(0, len(train_Y))
    test_indexes = np.arange(0, len(train_Y))
    batch_size = math.ceil(len(train_Y) / epoch_step)
    for epoch in tqdm.tqdm(range(epoch_num), desc="Training Epoch"):
        np.random.shuffle(indexes)
        np.random.shuffle(test_indexes)
        for step in range(epoch_step):
            index = indexes[step * batch_size: (step + 1) * batch_size]
            batch_X = to_sparse_tensor(train_X[index])
            batch_Y = train_Y[index]  
            predict = model(batch_X)
            optimizer.zero_grad()
            loss = criterion(predict, batch_Y)
            loss.backward()
            optimizer.step()
            test_index = test_indexes[step *
                                      batch_size: (step +
                                                   1) *
                                      batch_size]
