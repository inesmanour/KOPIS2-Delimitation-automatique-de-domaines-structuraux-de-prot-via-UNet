#!/usr/bin/env python3
"""
Dataset PyTorch pour les cartes de contacts et les domain_maps préparées
dans kopis_data/ml/unet.
c'est a dire 
→ sait lire tes .npy dans kopis_data/ml/unet/contacts/ et domains/
→ renvoie (X, Y) = (contact_map, domain_map) à PyTorch.

Organisation des données  :

kopis_data/
  ml/
    unet/
      contacts/
        A0A0A0RDR2.npy
        ...
      domains/
        A0A0A0RDR2.npy
        ...
      labels/       (optionnel, pas encore utilisé)
        ...

Ce dataset :
  - charge contacts/ID.npy  -> X : [1, N, N] float32
  - charge domains/ID.npy   -> Y : [1, N, N] float32 (0/1)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Racine du projet : src/training/dataset.py -> parents[2] = project root
ROOT = Path(__file__).resolve().parents[2]
ML_UNET_ROOT = ROOT / "kopis_data" / "ml" / "unet"
CONTACTS_DIR = ML_UNET_ROOT / "contacts"
DOMAINS_DIR = ML_UNET_ROOT / "domains"


def list_available_ids() -> List[str]:
    """
    Retourne la liste des IDs disponibles, définie comme
    l'intersection (stem) des fichiers contacts/*.npy et domains/*.npy.
    """
    contact_ids = {p.stem for p in CONTACTS_DIR.glob("*.npy")}
    domain_ids = {p.stem for p in DOMAINS_DIR.glob("*.npy")}
    ids = sorted(contact_ids & domain_ids)
    return ids


class ContactDomainDataset(Dataset):
    """
    Dataset minimal pour (contact_map, domain_map).

    Parameters
    ----------
    ids : sequence de str
        UniProt IDs correspondant aux noms de fichiers .npy.
    root : Path, optionnel
        Racine ml/unet. Par défaut : ML_UNET_ROOT global.
    """

    def __init__(
        self,
        ids: Sequence[str],
        root: Optional[Path] = None,
    ):
        super().__init__()
        self.ids = list(ids)
        if root is None:
            root = ML_UNET_ROOT
        self.root = root
        self.contacts_dir = root / "contacts"
        self.domains_dir = root / "domains"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        prot_id = self.ids[idx]

        # charge les npy
        contact_path = self.contacts_dir / f"{prot_id}.npy"
        domain_path = self.domains_dir / f"{prot_id}.npy"

        contact = np.load(contact_path)  # (N, N), float32
        domain_map = np.load(domain_path)  # (N, N), uint8 (0/1)

        # conversion en tensors PyTorch
        x = torch.from_numpy(contact).unsqueeze(0)  # -> [1, N, N]
        y = torch.from_numpy(domain_map.astype(np.float32)).unsqueeze(0)  # [1, N, N]

        return x, y