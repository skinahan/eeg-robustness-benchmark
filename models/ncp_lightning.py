import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from moabb.datasets import BNCI2014_001
from moabb.evaluations import WithinSessionEvaluation
from ncps.torch import CfC
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config import get_paradigm
from globals import set_seeds


class SpeedCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = (time.time() - self.start_time) * 1.0 / 60  # TODO: calibrate to GPU
        print(f"\nEpoch {trainer.current_epoch} took {elapsed:.3f} minutes")


class CfCClassifier(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, use_ltc=False):
        super().__init__()
        self.cfc = CfC(
            input_size=in_features,
            units=hidden_size,
            proj_size=out_features,
            return_sequences=False,
            batch_first=True,
            mixed_memory=True,
            mode='default'
        )
        self.out_feature = out_features
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        x:     [batch, seq_len, in_features]
        """
        x, _ = self.cfc(x)
        return x


class LightningCfC(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # assume your input has D channels → in_features = D
        self.model = CfCClassifier(
            in_features=self.hparams.in_features,
            hidden_size=self.hparams.hidden_size,
            out_features=2,
            use_ltc=False
        )
        self.validation_step_outputs = []
        self.accuracy = torchmetrics.classification.Accuracy(task='binary', num_classes=2)
        self.auroc = torchmetrics.AUROC(task="binary")
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, self.hparams.class_weight]))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x).view(-1, 2)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x).view(-1, 2)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        auc = self.auroc(preds, y)
        self.validation_step_outputs.append(auc)
        probs = torch.softmax(logits, dim=1)[:, 1]
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return {"probs": probs, "labels": y}

    def on_validation_epoch_end(self):
        # all_probs = torch.cat([o["probs"] for o in outputs])
        # all_labels = torch.cat([o["labels"] for o in outputs])
        auc = torch.stack(self.validation_step_outputs).mean()
        self.log("val_rocauc", auc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        optim_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "rmsprop": torch.optim.RMSprop}[
            self.hparams.optim]
        optimizer = optim_cls(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: self.hparams.decay_lr ** epoch)
        return [optimizer], [scheduler]


class TemporalWindow(nn.Module):
    """Learned temporal window to limit input sequence length"""
    def __init__(self, max_seq_len, learn_window=True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.learn_window = learn_window
        
        if learn_window:
            # Learnable parameters for window selection
            self.window_start = nn.Parameter(torch.tensor(0.0))
            self.window_length = nn.Parameter(torch.tensor(1.0))
        else:
            # Fixed window parameters
            self.window_start = 0.0
            self.window_length = 1.0
    
    def forward(self, x):
        """
        x: [batch, seq_len, features]
        Returns: [batch, windowed_seq_len, features]
        """
        batch_size, seq_len, features = x.shape
        
        if self.learn_window:
            # Apply learned window parameters
            start_idx = torch.clamp(self.window_start * seq_len, 0, seq_len - 1).long()
            window_len = torch.clamp(self.window_length * seq_len, 1, min(self.max_seq_len, seq_len)).long()
            end_idx = torch.clamp(start_idx + window_len, 0, seq_len)
            
            # Extract windowed sequence
            windowed_x = []
            for i in range(batch_size):
                start = start_idx.item()
                end = end_idx.item()
                windowed_x.append(x[i, start:end])
            
            # Pad or truncate to max_seq_len
            windowed_x = torch.stack(windowed_x)
            if windowed_x.shape[1] > self.max_seq_len:
                windowed_x = windowed_x[:, :self.max_seq_len, :]
            elif windowed_x.shape[1] < self.max_seq_len:
                # Pad with zeros
                pad_size = self.max_seq_len - windowed_x.shape[1]
                windowed_x = torch.cat([windowed_x, torch.zeros(batch_size, pad_size, features, device=x.device)], dim=1)
        else:
            # Use fixed window
            windowed_x = x[:, :self.max_seq_len, :]
        
        return windowed_x


class ImprovedCfCClassifier(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, max_seq_len=1000, 
                 dropout_rate=0.3, use_ltc=False, use_temporal_window=True):
        super().__init__()
        
        # Input normalization
        self.input_norm = nn.LayerNorm(in_features)
        
        # Temporal windowing
        self.use_temporal_window = use_temporal_window
        if use_temporal_window:
            self.temporal_window = TemporalWindow(max_seq_len, learn_window=True)
        
        # CfC layer with dropout
        self.cfc = CfC(
            input_size=in_features,
            units=hidden_size,
            proj_size=hidden_size,  # Changed to use hidden_size for intermediate representation
            return_sequences=False,
            batch_first=True,
            mixed_memory=True,
            mode='default'
        )
        
        # Additional regularization layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Classification head with regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, out_features)
        )
        
        self.out_feature = out_features
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        x: [batch, seq_len, in_features]
        """
        # Input normalization
        x = self.input_norm(x)
        
        # Apply temporal windowing if enabled
        if self.use_temporal_window:
            x = self.temporal_window(x)
        
        # CfC processing
        x, _ = self.cfc(x)
        
        # Apply dropout after CfC
        x = self.dropout1(x)
        
        # Classification head
        x = self.classifier(x)
        
        return x


class ImprovedLightningCfC(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Enhanced model with regularization
        self.model = ImprovedCfCClassifier(
            in_features=self.hparams.in_features,
            hidden_size=self.hparams.hidden_size,
            out_features=2,
            max_seq_len=getattr(self.hparams, 'max_seq_len', 1000),
            dropout_rate=getattr(self.hparams, 'dropout_rate', 0.3),
            use_ltc=False,
            use_temporal_window=getattr(self.hparams, 'use_temporal_window', True)
        )
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.accuracy = torchmetrics.classification.Accuracy(task='binary', num_classes=2)
        self.auroc = torchmetrics.AUROC(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        
        # Enhanced loss function with label smoothing
        if hasattr(self.hparams, 'label_smoothing') and self.hparams.label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.Tensor([1.0, self.hparams.class_weight]),
                label_smoothing=self.hparams.label_smoothing
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, self.hparams.class_weight]))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        
        # Log additional metrics
        f1_score = self.f1(preds, y)
        self.log_dict({
            "train_loss": loss, 
            "train_acc": acc,
            "train_f1": f1_score
        }, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        auc = self.auroc(preds, y)
        f1_score = self.f1(preds, y)
        
        self.validation_step_outputs.append({
            "val_loss": loss,
            "val_acc": acc,
            "val_auc": auc,
            "val_f1": f1_score,
            "probs": torch.softmax(logits, dim=1)[:, 1],
            "labels": y
        })
        
        self.log_dict({
            "val_loss": loss, 
            "val_acc": acc,
            "val_f1": f1_score
        }, prog_bar=True)
        
        return {"probs": torch.softmax(logits, dim=1)[:, 1], "labels": y}

    def on_validation_epoch_end(self):
        # Calculate mean metrics across all validation steps
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in self.validation_step_outputs]).mean()
        avg_auc = torch.stack([x["val_auc"] for x in self.validation_step_outputs]).mean()
        avg_f1 = torch.stack([x["val_f1"] for x in self.validation_step_outputs]).mean()
        
        self.log("val_rocauc", avg_auc, prog_bar=True)
        self.log("val_loss_epoch", avg_loss)
        self.log("val_acc_epoch", avg_acc)
        self.log("val_f1_epoch", avg_f1)
        
        # Clear outputs for next epoch
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        auc = self.auroc(preds, y)
        f1_score = self.f1(preds, y)
        
        self.test_step_outputs.append({
            "test_loss": loss,
            "test_acc": acc,
            "test_auc": auc,
            "test_f1": f1_score
        })
        
        return {"probs": torch.softmax(logits, dim=1)[:, 1], "labels": y}

    def on_test_epoch_end(self):
        # Calculate mean test metrics
        avg_loss = torch.stack([x["test_loss"] for x in self.test_step_outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in self.test_step_outputs]).mean()
        avg_auc = torch.stack([x["test_auc"] for x in self.test_step_outputs]).mean()
        avg_f1 = torch.stack([x["test_f1"] for x in self.test_step_outputs]).mean()
        
        self.log("test_loss", avg_loss)
        self.log("test_acc", avg_acc)
        self.log("test_rocauc", avg_auc)
        self.log("test_f1", avg_f1)
        
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optim_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "rmsprop": torch.optim.RMSprop}[
            self.hparams.optim]
        
        # Enhanced optimizer with better weight decay
        weight_decay = getattr(self.hparams, 'weight_decay', 1e-4)
        optimizer = optim_cls(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=weight_decay
        )
        
        # Enhanced learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_rocauc"
            }
        }

import seaborn as sns

def plot_loss_curve(log_dir):
    df = pd.read_csv(f"{log_dir}/metrics.csv")
    df = df.dropna(subset=["step"])  # drop early rows with NaNs

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='step', y='train_loss', label="Train Loss")
    sns.lineplot(data=df, x='step', y='val_loss', label="Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/loss_curve.png", dpi=300)
    plt.show()

from sklearn.model_selection import StratifiedKFold
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_features", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay_lr", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optim", choices=["adam", "adamw", "rmsprop"], default="adam")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--use_ltc", action="store_true")
    parser.add_argument("--class_weight", type=float, default=1.0)
    parser.add_argument("--gpus", type=int, default=1)
    
    # New regularization parameters
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--max_seq_len", type=int, default=1000)
    parser.add_argument("--use_temporal_window", action="store_true", default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    
    args = parser.parse_args()

    seed = 42
    set_seeds(seed)
    model_name = 'cnn_ncp_lightning'
    mode = 'tune'
    is_perturbed = False
    noise_type = None
    dataset = BNCI2014_001()
    subj = 1
    dataset.subject_list = [subj]
    subject_list = dataset.subject_list
    paradigm = get_paradigm()
    model = ImprovedLightningCfC(vars(args))
    ckpt_cb = ModelCheckpoint(monitor="val_rocauc", mode="max", save_top_k=1)
    logger = CSVLogger("logs", name="cfc_only")
    torch.use_deterministic_algorithms(False)
    X, y, metadata = paradigm.get_data(dataset, subjects=[subj])
    y_encoded = LabelEncoder().fit_transform(y)
    # extract metadata
    groups = metadata.subject.values
    sessions = metadata.session.values
    n_subjects = len(dataset.subject_list)

    train_mask = metadata["session"] == '0train'
    test_mask = metadata["session"] == '1test'

    X_train = X[train_mask]
    y_train = y_encoded[train_mask]

    X_train_t = torch.from_numpy(X_train).float().permute(0, 2, 1)
    y_train_t = torch.from_numpy(y_train).long()

    X_test = X[test_mask]
    y_test = y_encoded[test_mask]

    X_test_t = torch.from_numpy(X_test).float().permute(0, 2, 1)
    y_test_t = torch.from_numpy(y_test).long()

    n_splits = 3  # MOABB v1.5.1 WithinSessionEvaluation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_t, y_train_t)):
        fold_num = fold + 1
        print(f"\n===== Fold {fold_num} / {n_splits} =====")

        # Split data
        X_train, y_train = X_train_t[train_idx], y_train_t[train_idx]
        X_val, y_val = X_train_t[val_idx], y_train_t[val_idx]

        # Create loaders
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)

        # Init model and callbacks
        model = ImprovedLightningCfC(vars(args))
        logger = CSVLogger("logs", name=f"cfc_fold_{fold_num}")
        ckpt_cb = ModelCheckpoint(monitor="val_rocauc", mode="max", save_top_k=1)
        early_stop_cb = EarlyStopping(
            monitor="val_rocauc", 
            mode="max", 
            patience=args.early_stopping_patience,
            verbose=True
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu",
            devices=1,
            logger=logger,
            callbacks=[SpeedCallback(), ckpt_cb, early_stop_cb],
            log_every_n_steps=3,
            gradient_clip_val=args.gradient_clip_val,
            deterministic=False
        )

        trainer.fit(model, train_loader, val_loader)
        test_result = trainer.validate(model, dataloaders=val_loader, verbose=False)
        fold_results.append(test_result[0]["val_rocauc"])

        latest_version = 0
        while os.path.exists(f"./logs/cfc_fold_{fold_num}/version_{latest_version+1}"):
            latest_version += 1
        folder_path = f"./logs/cfc_fold_{fold_num}/version_{latest_version}"
        plot_loss_curve(folder_path)



    #
    # # 3. Create TensorDatasets
    # train_ds = TensorDataset(X_train_t, y_train_t)
    # test_ds = TensorDataset(X_test_t, y_test_t)
    # batch_size = args.batch_size
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True,
    # )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    #
    # trainer = pl.Trainer(
    #     max_epochs=args.epochs,
    #     devices=1,
    #     accelerator="gpu",
    #     gradient_clip_val=0.0,
    #     callbacks=[SpeedCallback(), ckpt_cb],
    #     logger=logger,
    # )
    # trainer.fit(model, train_loader)
    results = trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
