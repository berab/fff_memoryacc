import mlflow
import torch
import logging

from .base import BaseTrainExp 

from utils.nn import moe_train_epoch, eval_model
from utils.moe_stats import get_experts, get_expert_stats


class TrainMoE(BaseTrainExp):
    def __init__(self):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "Train"

    def get_config(self) -> dict:
        exp_conf = {'exp_name': self.exp_name,
                    }
        return self.model.get_config() | self.loader.get_config() | exp_conf 

    def log_exp(self, metrics) -> None:
        self.log_metrics(metrics)
        self.log_moe()

    def run_exp(self) -> dict:
        # Metrics init.
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                   }
        # Training
        for epoch in range(self.epochs):
            train_loss, train_acc = moe_train_epoch(self.model, self.optim, self.loader.train, self.criterion, epoch, self.device)
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

            val_experts = get_experts(self.model, self.loader.valid, self.device)
            val_expert_stats = torch.tensor(get_expert_stats(val_experts, self.model.n_experts))
            expert_dev = torch.std(val_expert_stats)
            logging.info(f"expert dev: {expert_dev}")
            logging.info(f"Val expert stats: {val_expert_stats}")


        val_experts = get_experts(self.model, self.loader.valid, self.device)
        val_expert_stats = torch.tensor(get_expert_stats(val_experts, self.model.n_experts))

        val_expert_sorted_indices = torch.sort(val_expert_stats, descending=True).indices
        val_new_expert_indices = torch.empty_like(val_expert_sorted_indices)
        val_new_expert_indices[val_expert_sorted_indices] = torch.arange(self.model.n_experts)

        logging.info(f"Val expert stats: {val_expert_stats}")
        test_experts = get_experts(self.model, self.loader.test, self.device)
        test_expert_stats = torch.tensor(get_expert_stats(test_experts, self.model.n_experts))

        test_expert_std = torch.std(test_expert_stats)
        val_expert_std = torch.std(val_expert_stats)
        mlflow.log_metric("test_expert_std", test_expert_std.item())
        mlflow.log_metric("val_expert_std", val_expert_std.item())
        logging.info(f"Test expert std: {test_expert_std}, Val expert std: {val_expert_std}")

        torch.save(val_expert_stats, self.out_dir/f"expert_stats.pt") # TODO: Add more checkpoints
        torch.save(torch.tensor(test_experts), self.out_dir/f"test_experts.pt") # TODO: Add more checkpoints
        torch.save(torch.tensor(val_experts), self.out_dir/f"val_experts.pt") # TODO: Add more checkpoints
        torch.save(torch.tensor(val_new_expert_indices), self.out_dir/f"val_opt_indices.pt") # TODO: Add more checkpoints
        mlflow.log_artifact(str(self.out_dir/'expert_stats.pt'))
        mlflow.log_artifact(str(self.out_dir/'test_experts.pt'))
        mlflow.log_artifact(str(self.out_dir/'val_experts.pt'))
        mlflow.log_artifact(str(self.out_dir/'val_opt_indices.pt'))

        # # Testing
        test_loss, test_acc= eval_model(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid logging.info("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
        logging.info("FINAL TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        self.log_test(test_loss, test_acc)
        return metrics
