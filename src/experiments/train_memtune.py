import mlflow
import torch
import logging
from .base import BaseTrainExp 

from utils.nn import train_epoch, eval_model, train_epoch_mem
from utils.fff_stats import get_leaves, get_leaf_stats, get_partition_count

#
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import DirichletPartitioner
MCU_CONFIG = {
    "apollo": [384, 1000], # TCM, SRAM
    "stm": [96, 1000], # SRAM, FLASH
}


class TrainMemTune(BaseTrainExp):
    def __init__(self):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "TrainMemTune"

    def get_config(self) -> dict:
        exp_conf = {'exp_name': self.exp_name,
                    "reg_alpha": self.reg_alpha,
                    "entropy_alpha": self.entropy_alpha,
                    "dist_alpha": self.dist_alpha,
                    "dist_reg": self.dist_reg,
                    "dist_warmup": self.dist_warmup,
                    "mem_alpha": self.mem_alpha,
                    "mem_tune": self.mem_tune,
                    }
        return self.model.get_config() | self.loader.get_config() | exp_conf 

    def log_exp(self, metrics) -> None:
        self.log_metrics(metrics)
        self.log_model()

    def run_exp(self) -> dict:
        mem1, mem2 = MCU_CONFIG[self.target_mcu]
        router_size, leaf_size = self.model.get_element_memory_kb()
        n_mem1 = get_partition_count(mem1, mem2, self.model.depth, leaf_size, router_size)


        # Metrics init.
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                   "reg_loss": [], "entropy_loss": [],
                   "dist_loss": [], "leaf_std": [],
                   "p_mem1": [], "p_mem2": [], "pmem1/pmem2": [],
                   "mem_loss": [],
                   }
        # Training
        val_leaves, all_val_leaf_stats = [], []
        mem1_leaf_stats = None
        for epoch in range(self.epochs):
            train_loss, train_acc, reg_loss, entropy_loss, dist_loss, _ = train_epoch(self.model, self.optim, self.loader.train, self.criterion, epoch, self.device, self.reg_alpha, self.entropy_alpha, self.dist_reg, self.dist_alpha)
            val_loss, val_acc = eval_model(self.model, self.loader.valid, self.criterion, self.device) #TODO: Change valid
            test_loss, test_acc = eval_model(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {:.4f}, train loss: {:.8f}, valid acc: {:.4f}, valid loss: {:.8f}, test acc: {:.4f}, test loss: {:.8f}".format(
                epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            metrics['train_acc'].append(train_acc)
            metrics['train_loss'].append(train_loss)
            metrics['reg_loss'].append(reg_loss)
            metrics['entropy_loss'].append(entropy_loss)
            metrics['dist_loss'].append(dist_loss)
            metrics['val_acc'].append(val_acc)
            metrics['val_loss'].append(val_loss)
            metrics['mem_loss'].append(0.0)

            val_leaves = get_leaves(self.model, self.loader.valid, self.device)
            val_leaf_stats = torch.tensor(get_leaf_stats(val_leaves, self.model.n_leaves))
            val_leaf_stats_sorted, val_leaves_sorted = val_leaf_stats.sort(descending=True)
            mem1_leaf_stats, mem2_leaf_stats = val_leaf_stats_sorted[:n_mem1], val_leaf_stats_sorted[n_mem1:]
            mem1_leaves, mem2_leaves = val_leaves_sorted[:n_mem1], val_leaves_sorted[n_mem1:]
            leaf_dev = torch.std(val_leaf_stats)
            metrics['leaf_std'].append(leaf_dev)
            metrics['p_mem1'].append(sum(mem1_leaf_stats))
            metrics['p_mem2'].append(sum(mem2_leaf_stats))
            metrics['pmem1/pmem2'].append(sum(mem1_leaf_stats)/sum(mem2_leaf_stats))
            logging.info("pmem1: {}, pmem2: {}, pmem1/pmem2: {}".format(metrics["p_mem1"][-1], metrics["p_mem2"][-1], metrics["pmem1/pmem2"][-1]))

            self.log_epoch(epoch, metrics)
            self.model.to(self.device)

            logging.info(f"reg loss: {reg_loss}, entropy loss: {entropy_loss}, leaf dev: {leaf_dev}")
            logging.info(f"Val leaf stats: {val_leaf_stats}")
            all_val_leaf_stats.append(val_leaf_stats)

        logging.info("Memory finetuning starts...")
        for epoch in range(self.mem_tune):
        # for epoch in range(self.epochs):
            train_loss, train_acc, reg_loss, entropy_loss, dist_loss, mem_loss = train_epoch_mem(self.model, self.optim, self.loader.train, self.criterion, epoch, self.device, self.reg_alpha, self.entropy_alpha, self.dist_reg, self.dist_alpha, n_mem1, self.mem_alpha, mem1_leaves)
            val_loss, val_acc = eval_model(self.model, self.loader.valid, self.criterion, self.device) #TODO: Change valid
            test_loss, test_acc = eval_model(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {:.4f}, train loss: {:.8f}, valid acc: {:.4f}, valid loss: {:.8f}, test acc: {:.4f}, test loss: {:.8f}".format(
                epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            metrics['train_acc'].append(train_acc)
            metrics['train_loss'].append(train_loss)
            metrics['reg_loss'].append(reg_loss)
            metrics['entropy_loss'].append(entropy_loss)
            metrics['dist_loss'].append(dist_loss)
            metrics['val_acc'].append(val_acc)
            metrics['val_loss'].append(val_loss)
            metrics['mem_loss'].append(mem_loss)

            val_leaves = get_leaves(self.model, self.loader.valid, self.device)
            val_leaf_stats = torch.tensor(get_leaf_stats(val_leaves, self.model.n_leaves))
            val_leaf_stats_sorted, val_leaves_sorted = val_leaf_stats.sort(descending=True)
            mem1_leaf_stats, mem2_leaf_stats = val_leaf_stats_sorted[:n_mem1], val_leaf_stats_sorted[n_mem1:]
            mem1_leaves, mem2_leaves = val_leaves_sorted[:n_mem1], val_leaves_sorted[n_mem1:]
            leaf_dev = torch.std(val_leaf_stats)
            metrics['leaf_std'].append(leaf_dev)
            metrics['p_mem1'].append(sum(mem1_leaf_stats))
            metrics['p_mem2'].append(sum(mem2_leaf_stats))
            metrics['pmem1/pmem2'].append(sum(mem1_leaf_stats)/sum(mem2_leaf_stats))
            logging.info("Mem loss: {}, pmem1: {}, pmem2: {}, pmem1/pmem2: {}".format(mem_loss, metrics["p_mem1"][-1], metrics["p_mem2"][-1], metrics["pmem1/pmem2"][-1]))

            self.log_epoch(epoch, metrics)
            self.model.to(self.device)

            logging.info(f"reg loss: {reg_loss}, entropy loss: {entropy_loss}, leaf dev: {leaf_dev}")
            logging.info(f"Val leaf stats: {val_leaf_stats}")
            all_val_leaf_stats.append(val_leaf_stats)


        val_leaf_stats = all_val_leaf_stats[-1]
        all_val_leaf_stats = torch.cat(all_val_leaf_stats)
        torch.save(all_val_leaf_stats, self.out_dir/f"all_leaf_stats.pt") # TODO: Add more checkpoints
        mlflow.log_artifact(str(self.out_dir/'all_leaf_stats.pt'))

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

        # Testing
        for i in range(len(val_leaf_stats)):
            mlflow.log_metric(f"val_leaf_stat{i}", val_leaf_stats[i].item())
            mlflow.log_metric(f"leaf_stat{i}", test_leaf_stats[i].item())
        for i, l in enumerate(mem1_leaves):
            mlflow.log_metric(f"mem1_leaf{i}", l)
        for i, l in enumerate(mem2_leaves):
            mlflow.log_metric(f"mem2_leaf{i}", l)
        test_loss, test_acc = eval_model(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid logging.info("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
        logging.info("FINAL TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        self.log_test(test_loss, test_acc)
        return metrics
