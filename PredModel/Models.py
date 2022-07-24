import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from torch import optim
import logging
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from typing import Iterable

logger = logging.getLogger("LSTM")


class LinearNN(nn.Module):
    def __init__(self, input_size: int = 50):
        super(LinearNN, self).__init__()
        self.stack_layer = nn.Sequential(
            nn.Linear(input_size, 1)
        )

    def forward(self, x_seq: torch.Tensor):
        return self.stack_layer(x_seq)


class LogisticRegression(nn.Module):
    def __init__(self, input_size: int = 50):
        super(LogisticRegression, self).__init__()
        self.stack_layer = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x_seq: torch.Tensor):
        return self.stack_layer(x_seq)


class RNNModel(nn.Module):
    def __init__(self, input_size: int = 50, hidden_size: int = 10,
                 hidden_state: torch.Tensor = None):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.hidden_state = torch.zeros(
            1, 1, self.hidden_size) if hidden_state is None else hidden_state

    def forward(self, x_seq: torch.Tensor):
        x_seq = torch.unsqueeze(x_seq, 1)
        y_rnn, self.hidden_cell = self.rnn(x_seq, self.hidden_state)
        return self.linear(y_rnn[-1])


class LSTM(nn.Module):
    def __init__(
            self, input_size: int = 50, hidden_size: int = 10,
            hidden_state: torch.Tensor = None, cell_state: torch.Tensor = None):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.hidden_state = torch.zeros(
            1, 1, self.hidden_size) if hidden_state is None else hidden_state
        self.cell_state = torch.zeros(
            1, 1, self.hidden_size) if cell_state is None else cell_state

    def forward(self, x_seq: torch.Tensor):
        x_seq = torch.unsqueeze(x_seq, 1)
        y_lstm, (hidden_cell, cell_state) = self.lstm(
            x_seq, (self.hidden_state, self.cell_state))
        return self.sigmoid(self.linear(y_lstm[-1]))


class Predictor(object):
    def __init__(self, model: nn.Module, device=None):
        self.model = model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device

    def train(
            self, features: Iterable[torch.Tensor],
            labels: Iterable[float],
            lr: float = 5e-5, epochs: int = 1, batch_size: int = 0):
        self.model.to(self.device)
        if isinstance(
                self.model, RNNModel):  # RNN model's hidden state need to be put on device manually
            self.model.hidden_state = self.model.hidden_state.to(self.device)
        elif isinstance(self.model, LSTM):
            self.model.hidden_state = self.model.hidden_state.to(self.device)
            self.model.cell_state = self.model.cell_state.to(self.device)

        criterion = torch.nn.BCELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: max(
                0.995 ** step, 5e-3))
        labels = labels.to(self.device)
        for epoch in trange(epochs):
            for step, seq in enumerate(tqdm(features)):
                optimizer.zero_grad()
                output = self.model(seq.to(self.device))
                loss = criterion(
                    torch.squeeze(output),
                    torch.squeeze(
                        labels[step]))
                loss.backward()
                optimizer.step()
                if not step % 100:  # plot the loss sum every 100 step
                    scheduler.step()
                    logger.info("Step {}: Loss is {}".format(step, loss))
        return self.model

    def predict(self, features: Iterable[torch.Tensor]):
        '''Predict the return based on embedded text features'''
        if isinstance(self.model, LSTM):
            previous_state = (self.model.hidden_state, self.model.cell_state)
        elif isinstance(self.model, RNNModel):
            previous_state = self.model.hidden_state
        pred = list()
        for seq in tqdm(features):
            try:
                pred.append(self.model(seq.to(self.device)))
            except Exception:
                pred.append(0.5)
        return torch.Tensor(pred).detach().numpy()


def cal_metrics(pred, label):
    f1 = f1_score(label, pred > 0.5)
    acc = accuracy_score(label, pred > 0.5)
    prec = precision_score(label, pred > 0.5)
    recall = recall_score(label, pred > 0.5)
    metrics = {"f1": f1, "acc": acc, "prec": prec, "recall": recall}
    return metrics
