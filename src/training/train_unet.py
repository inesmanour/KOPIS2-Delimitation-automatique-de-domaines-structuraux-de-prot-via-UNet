#!/usr/bin/env python3
"""
Entraînement du U-Net sur les cartes de contacts + domain_maps.

Usage typique (depuis la racine du projet) :

    conda activate kopis2
    python src/training/train_unet.py

Ce script :

  1) Récupère la liste des IDs disponibles (intersection contacts/domains).
  2) Crée (ou recharge) un split train / val / test.
  3) Construit les DataLoader (batch_size=1 car N variable).
  4) Initialise un U-Net 2D.
  5) Entraîne avec BCEWithLogitsLoss + early stopping sur la val_loss.
  6) Sauvegarde :
       - best_model.pt
       - training_log.csv
"""

from __future__ import annotations
from tqdm.auto import tqdm
import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# --- Pour pouvoir importer models.unet et training.dataset sans package explicite ---
import sys

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]            # .../src
ROOT = THIS_FILE.parents[2]               # racine projet
sys.path.append(str(SRC_DIR))

# Dossier des données U-Net (même logique que dans dataset.py)
ML_UNET_ROOT = ROOT / "kopis_data" / "ml" / "unet"

from models.unet import UNet2D            # type: ignore
from training.dataset import (            # type: ignore
    ContactDomainDataset,
    list_available_ids,
)


# ----------------------------------------------------------------------------- #
#   Utils : seed, splits, métriques
# ----------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_or_load_splits(
    all_ids: List[str],
    split_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    split_root.mkdir(parents=True, exist_ok=True)
    train_file = split_root / "train_ids.txt"
    val_file = split_root / "val_ids.txt"
    test_file = split_root / "test_ids.txt"

    if train_file.exists() and val_file.exists() and test_file.exists():
        def read_ids(p: Path) -> List[str]:
            return [l.strip() for l in p.read_text().splitlines() if l.strip()]

        return read_ids(train_file), read_ids(val_file), read_ids(test_file)

    rng = random.Random(seed)
    ids = list(all_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids   = ids[n_train:n_train + n_val]
    test_ids  = ids[n_train + n_val:]

    train_file.write_text("\n".join(train_ids) + "\n")
    val_file.write_text("\n".join(val_ids) + "\n")
    test_file.write_text("\n".join(test_ids) + "\n")

    return train_ids, val_ids, test_ids


def compute_batch_metrics(logits: torch.Tensor, targets: torch.Tensor):
    """
    Calcule accuracy + F1 binaire sur un batch.

    logits  : [B, 1, N, N]
    targets : [B, 1, N, N] (0/1)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        y_true = targets.view(-1)
        y_pred = preds.view(-1)

        # TP, FP, FN
        tp = (y_true * y_pred).sum().item()
        fp = ((1 - y_true) * y_pred).sum().item()
        fn = (y_true * (1 - y_pred)).sum().item()
        tn = ((1 - y_true) * (1 - y_pred)).sum().item()

        # accuracy
        acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)

        # F1
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

    return acc, f1


# ----------------------------------------------------------------------------- #
#   Train / Val loop
# ----------------------------------------------------------------------------- #
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch_idx: int,
    num_epochs: int,
):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    n_samples = 0

    pbar = tqdm(
        loader,
        desc=f"[Train] Epoch {epoch_idx}/{num_epochs}",
        unit="batch",
        leave=False,
    )

    for X, Y in pbar:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        # 🔹 métriques réelles
        acc, f1 = compute_batch_metrics(logits.detach(), Y)
        running_acc += acc * batch_size
        running_f1 += f1 * batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(1, n_samples)
    epoch_acc = running_acc / max(1, n_samples)
    epoch_f1 = running_f1 / max(1, n_samples)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    epoch_idx: int,
    num_epochs: int,
):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    n_samples = 0

    pbar = tqdm(
        loader,
        desc=f"[Val]   Epoch {epoch_idx}/{num_epochs}",
        unit="batch",
        leave=False,
    )

    for X, Y in pbar:
        X = X.to(device)
        Y = Y.to(device)

        logits = model(X)
        loss = criterion(logits, Y)

        batch_size = X.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        acc, f1 = compute_batch_metrics(logits, Y)
        running_acc += acc * batch_size
        running_f1 += f1 * batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(1, n_samples)
    val_acc = running_acc / max(1, n_samples)
    val_f1 = running_f1 / max(1, n_samples)

    return epoch_loss, val_acc, val_f1


# ----------------------------------------------------------------------------- #
#   Main
# ----------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Entraînement U-Net sur contact maps.")
    parser.add_argument("--epochs", type=int, default=5, help="Nombre d'epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (laisser 1)")
    parser.add_argument("--patience", type=int, default=10, help="Patience early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # --- préparation des IDs & splits ---
    all_ids = list_available_ids()
    print(f"{len(all_ids)} protéines avec contacts + domaines trouvées.")

    split_root = ML_UNET_ROOT / "splits"
    train_ids, val_ids, test_ids = make_or_load_splits(all_ids, split_root)
    
    # TEST Sous-échantillon pour debug (200 / 50 / 50) voir si mes mectrcis st goods
    train_ids = train_ids[:200]
    val_ids   = val_ids[:50]
    test_ids  = test_ids[:50]
    ####
    
    print(
        f"Split : train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)} "
        f"(fichiers dans {split_root})"
    )

    # --- datasets & dataloaders ---
    train_dataset = ContactDomainDataset(train_ids)
    val_dataset = ContactDomainDataset(val_ids)
    # le test_dataset sera utilisé plus tard, après entraînement
    test_dataset = ContactDomainDataset(test_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    # on prépare quand même, même si on ne l'utilise pas ici
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --- modèle, loss, optim ---
    model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- dossiers de sortie ---
    results_root = ROOT / "results" / "unet"
    results_root.mkdir(parents=True, exist_ok=True)
    best_model_path = results_root / "best_model.pt"
    log_path = results_root / "training_log.csv"

    # initialisation du fichier de log
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "train_f1",
                    "val_loss",
                    "val_acc",
                    "val_f1",
                ]
            )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # --------------------------------------------------------------------- #
    #   Boucle d'entraînement
    # --------------------------------------------------------------------- #
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch_idx=epoch,
            num_epochs=args.epochs,
        )

        val_loss, val_acc, val_f1 = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch_idx=epoch,
            num_epochs=args.epochs,
        )

        print(
            f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, f1={train_f1:.4f}"
        )
        print(
            f"  Val  : loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}"
        )

        # log CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{train_f1:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                    f"{val_f1:.6f}",
                ]
            )

        # Early stopping / checkpoint
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_model_path,
            )
            print(f"  -> Nouveau meilleur modèle sauvegardé : {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  -> Pas d'amélioration (patience {epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print("  -> Early stopping déclenché.")
                break

    print("\nEntraînement terminé.")
    print(f"Meilleur modèle : {best_model_path}")
    print(f"Log d'entraînement : {log_path}")
    print(f"Pour l'évaluation finale, recharge le modèle et utilise le test_loader.")


if __name__ == "__main__":
    main()