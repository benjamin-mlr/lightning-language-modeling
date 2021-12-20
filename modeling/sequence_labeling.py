


from argparse import ArgumentParser
import pytorch_lightning as pl
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoConfig,
)
from transformers.optimization import AdamW
from data import LMDataModule

from uuid import uuid4
import torch

import logging

from pathlib import Path

class SequenceLablling(pl.LightningModule):
    def __init__(self, model_name_or_path, learning_rate, adam_beta1, adam_beta2, adam_epsilon):
        super().__init__()

        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(
            model_name_or_path, return_dict=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            config=config)

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('valid_loss', loss, on_step=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          self.hparams.learning_rate,
                          betas=(self.hparams.adam_beta1,
                                 self.hparams.adam_beta2),
                          eps=self.hparams.adam_epsilon,)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=5e-5)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        return parser



