import os
import torch
import torch.nn as nn


class EarlyStopping(object):
    def __init__(self, patience: int, save_model_folder: str, save_model_name: str):
        """
        strategy for early stopping
        :param patience: max patience
        :param save_model_folder: save model folder
        :param save_model_name: save model name
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")

    def step(self, metrics: list, model: nn.Module):
        """

        :param metrics: list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
        :param model: model
        :return:
        """

        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value > self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value < self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_checkpoint(model)
            self.counter = 0
        # metrics are not better at the epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation metric increases."""
        # print(f"save model {self.save_model_path}")
        torch.save(model.state_dict(), self.save_model_path)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        print(f"load model {self.save_model_path}")
        model.load_state_dict(torch.load(self.save_model_path))
