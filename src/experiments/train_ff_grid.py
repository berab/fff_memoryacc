import mlflow
import torch
import logging
from .base import BaseTrainExp 

from utils.nn import train_epoch, eval_model
from utils.fff_stats import get_leaves, get_leaf_stats

class TrainGrid(BaseTrainExp):
    def __init__(self):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "Train"

    def get_config(self) -> dict:
        exp_conf = {'exp_name': self.exp_name,
                    "reg_alpha": self.reg_alpha,
                    "entropy_alpha": self.entropy_alpha,
                    }
        return self.model.get_config() | self.loader.get_config() | exp_conf 

    def log_exp(self, metrics) -> None:
        self.log_metrics(metrics)
        self.log_model()

    def run_exp(self) -> dict:


    def train_model(self, reg_alpha) -> dict:
        # Assuming the class is imported from your framework, e.g., from fedframework import DirichletPartitioner

        # Metrics init.
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                   }
        # Training
        for epoch in range(self.epochs):
            train_loss, train_acc, reg_loss, entropy_loss = train_epoch(self.model, self.optim, self.loader.train, self.criterion, epoch, self.device, reg_alpha, self.entropy_alpha)
            val_loss, val_acc = eval_model(self.model, self.loader.valid, self.criterion, self.device) #TODO: Change valid
            test_loss, test_acc = eval_model(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {:.4f}, train loss: {:.8f}, valid acc: {:.4f}, valid loss: {:.8f}, test acc: {:.4f}, test loss: {:.8f}".format(
                epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            metrics['train_acc'].append(train_acc)
            metrics['train_loss'].append(train_loss)
            metrics['val_acc'].append(val_acc)
            metrics['val_loss'].append(val_loss)
            self.log_epoch(epoch, metrics)
            self.model.to(self.device)

            val_leaves = get_leaves(self.model, self.loader.valid, self.device)
            val_leaf_stats = torch.tensor(get_leaf_stats(val_leaves, self.model.n_leaves))
            leaf_dev = torch.std(val_leaf_stats)
            logging.info(f"reg loss: {reg_loss}, entropy loss: {entropy_loss}, leaf dev: {leaf_dev}")
            logging.info(f"Val leaf stats: {val_leaf_stats}")


        val_leaves = get_leaves(self.model, self.loader.valid, self.device)
        val_leaf_stats = torch.tensor(get_leaf_stats(val_leaves, self.model.n_leaves))

        val_leaf_sorted_indices = torch.sort(val_leaf_stats, descending=True).indices
        val_new_leaf_indices = torch.empty_like(val_leaf_sorted_indices)
        val_new_leaf_indices[val_leaf_sorted_indices] = torch.arange(self.model.n_leaves)

        logging.info(f"Val leaf stats: {val_leaf_stats}")
        test_leaves = get_leaves(self.model, self.loader.test, self.device)
        test_leaf_stats = torch.tensor(get_leaf_stats(test_leaves, self.model.n_leaves))

        test_leaf_std = torch.std(test_leaf_stats)
        val_leaf_std = torch.std(val_leaf_stats)
        mlflow.log_metric("test_leaf_std", test_leaf_std.item())
        mlflow.log_metric("val_leaf_std", val_leaf_std.item())
        logging.info(f"Test leaf std: {test_leaf_std}, Val leaf std: {val_leaf_std}")

        torch.save(val_leaf_stats, self.out_dir/f"leaf_stats.pt") # TODO: Add more checkpoints
        torch.save(torch.tensor(test_leaves), self.out_dir/f"test_leaves.pt") # TODO: Add more checkpoints
        torch.save(torch.tensor(val_leaves), self.out_dir/f"val_leaves.pt") # TODO: Add more checkpoints
        torch.save(torch.tensor(val_new_leaf_indices), self.out_dir/f"val_opt_indices.pt") # TODO: Add more checkpoints
        mlflow.log_artifact(str(self.out_dir/'leaf_stats.pt'))
        mlflow.log_artifact(str(self.out_dir/'test_leaves.pt'))
        mlflow.log_artifact(str(self.out_dir/'val_leaves.pt'))
        mlflow.log_artifact(str(self.out_dir/'val_opt_indices.pt'))

        # # Testing
        test_loss, test_acc= eval_model(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid logging.info("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
        logging.info("FINAL TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        self.log_test(test_loss, test_acc)
        return metrics
