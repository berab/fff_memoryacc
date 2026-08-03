import mlflow
import torch
import logging

from .base import BaseTrainExp 

from utils.nn import moe_train_epoch, eval_model
from utils.moe_stats import get_experts, get_expert_stats, get_partition_count


MCU_CONFIG = {
    "apollo": [384, 1000], # TCM, SRAM
    "stm": [96, 1000], # SRAM, FLASH
}

class TrainMoE(BaseTrainExp):
    def __init__(self):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "TrainMoE"

    def get_config(self) -> dict:
        exp_conf = {'exp_name': self.exp_name,
                    "reg_alpha": self.reg_alpha,
                    "entropy_alpha": self.entropy_alpha,
                    "dist_warmup": self.dist_warmup,
                    "mem_alpha": self.mem_alpha,
                    "mem_tune": self.mem_tune,
                    "dist_alpha": self.dist_alpha,
                    }
        return self.model.get_config() | self.loader.get_config() | exp_conf 

    def log_exp(self, metrics) -> None:
        self.log_metrics(metrics)
        self.log_moe()

    def run_exp(self) -> dict:
        # Metrics init.
        mem1, mem2 = MCU_CONFIG[self.target_mcu]
        router_size, expert_size = self.model.get_element_memory_kb()
        n_mem1 = get_partition_count(mem1, mem2, self.model.n_experts, expert_size, router_size)
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                   "reg_loss": [], "entropy_loss": [],
                   "expert_std": [], "mem_loss": [],
                   "p_mem1": [], "p_mem2": [], "pmem1/pmem2": [],
                   }
        # Training
        for epoch in range(self.epochs):
            train_loss, train_acc, reg_loss, entropy_loss, mem_loss = moe_train_epoch(self.model, self.optim, self.loader.train, self.criterion, epoch, self.device, self.reg_alpha, self.entropy_alpha, self.dist_reg, self.dist_alpha, n_mem1, self.mem_alpha)
            val_loss, val_acc = eval_model(self.model, self.loader.valid, self.criterion, self.device) #TODO: Change valid
            test_loss, test_acc = eval_model(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {:.4f}, train loss: {:.8f}, valid acc: {:.4f}, valid loss: {:.8f}, test acc: {:.4f}, test loss: {:.8f}".format(
                epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            metrics['train_acc'].append(train_acc)
            metrics['train_loss'].append(train_loss)
            metrics['reg_loss'].append(reg_loss)
            metrics['entropy_loss'].append(entropy_loss)
            metrics['val_acc'].append(val_acc)
            metrics['val_loss'].append(val_loss)
            metrics["mem_loss"].append(mem_loss)

            val_experts = get_experts(self.model, self.loader.valid, self.device)
            val_expert_stats = torch.tensor(get_expert_stats(val_experts, self.model.n_experts))
            val_expert_stats_sorted, val_experts_sorted = val_expert_stats.sort(descending=True)
            mem1_expert_stats, mem2_expert_stats = val_expert_stats_sorted[:n_mem1], val_expert_stats_sorted[n_mem1:]
            mem1_experts, mem2_experts = val_experts_sorted[:n_mem1], val_experts_sorted[n_mem1:]
            expert_dev = torch.std(val_expert_stats)
            metrics['expert_std'].append(expert_dev)
            metrics['p_mem1'].append(sum(mem1_expert_stats))
            metrics['p_mem2'].append(sum(mem2_expert_stats))
            metrics['pmem1/pmem2'].append(sum(mem1_expert_stats)/sum(mem2_expert_stats))
            logging.info("Mem loss: {}, pmem1: {}, pmem2: {}, pmem1/pmem2: {}".format(mem_loss, metrics["p_mem1"][-1], metrics["p_mem2"][-1], metrics["pmem1/pmem2"][-1]))

            self.log_epoch(epoch, metrics)
            self.model.to(self.device)

            val_experts = get_experts(self.model, self.loader.valid, self.device)
            val_expert_stats = torch.tensor(get_expert_stats(val_experts, self.model.n_experts))
            expert_dev = torch.std(val_expert_stats)
            logging.info(f"reg loss: {reg_loss}, entropy loss: {entropy_loss}, expert dev: {expert_dev}")
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
