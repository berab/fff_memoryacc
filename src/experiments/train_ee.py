import mlflow
import torch
import logging
from .base import BaseTrainExp 

from utils.nn import train_epoch_ee, eval_model_ee, train_epoch_ee_v2

class TrainEE(BaseTrainExp):
    def __init__(self):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "TrainEE"

    def get_config(self) -> dict:
        exp_conf = {'exp_name': self.exp_name,
                    }
        return self.model.get_config() | self.loader.get_config() | exp_conf 

    def log_exp(self, metrics) -> None:
        self.log_metrics(metrics)
        self.log_ee()

    def run_exp(self) -> dict:
        # Metrics init.
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                   }
        # Training
        for epoch in range(self.epochs):
            train_loss, train_acc, train_early = train_epoch_ee(self.model, self.optim, self.loader.train, self.criterion, epoch, self.device)
            val_loss, val_acc, val_early = eval_model_ee(self.model, self.loader.valid, self.criterion, self.device) #TODO: Change valid
            test_loss, test_acc, test_early = eval_model_ee(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {:.4f}, train loss: {:.8f}, valid acc: {:.4f}, valid loss: {:.8f}, test acc: {:.4f}, test loss: {:.8f}".format(
                epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            logging.info("train early: {}, val early: {}, test early: {}".format(train_early, val_early, test_early))
            metrics['train_acc'].append(train_acc)
            metrics['train_loss'].append(train_loss)
            metrics['val_acc'].append(val_acc)
            metrics['val_loss'].append(val_loss)
            self.log_epoch(epoch, metrics)
            self.model.to(self.device)

        # # Testing
        test_loss, test_acc, test_early = eval_model_ee(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid logging.info("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
        logging.info("FINAL TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        self.log_test(test_loss, test_acc)
        return metrics

    def run_exp_v2(self) -> dict:
        # Metrics init.
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                   }
        # Training
        for epoch in range(self.epochs):
            train_loss, train_acc = train_epoch_ee_v2(self.model, self.optim, self.loader.train, self.criterion, epoch, self.device)
            val_loss, val_acc, val_early = eval_model_ee(self.model, self.loader.valid, self.criterion, self.device) #TODO: Change valid
            test_loss, test_acc, test_early = eval_model_ee(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {:.4f}, train loss: {:.8f}, valid acc: {:.4f}, valid loss: {:.8f}, test acc: {:.4f}, test loss: {:.8f}".format(
                epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            logging.info("train early: {:.4f}, val early: {:.4f}, test early: {:.4f}".format(train_early, val_early, test_early))
            metrics['train_acc'].append(train_acc)
            metrics['train_loss'].append(train_loss)
            metrics['val_acc'].append(val_acc)
            metrics['val_loss'].append(val_loss)
            self.log_epoch(epoch, metrics)
            self.model.to(self.device)

        # # Testing
        test_loss, test_acc, test_early = eval_model_ee(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid logging.info("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
        logging.info("FINAL TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        self.log_test(test_loss, test_acc)
        return metrics
