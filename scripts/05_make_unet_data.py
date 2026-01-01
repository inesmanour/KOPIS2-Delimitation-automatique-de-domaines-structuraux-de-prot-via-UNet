#!/usr/bin/env python3
"""
Préparation des données pour l'U-Net à partir des sorties SWORD2.

Pour chaque protéine dans kopis_data/sword/UNIPROTID/, ce script :

  1) Cherche le fichier Peeling.log.
  2) En déduit les domaines (bornes) à partir de la DERNIÈRE ligne
     (celle avec le CI maximal).
  3) Charge la matrice de contacts résidu–résidu à partir de
     file_proba_contact.mat (matrice de probabilités N×N).
  4) Construit une domain_map N×N (1 si deux résidus sont dans le même domaine).
  5) Sauvegarde :
       - kopis_data/ml/unet/contacts/UNIPROTID.npy
       - kopis_data/ml/unet/domains/UNIPROTID.npy
"""

import numpy as np
from pathlib import Path

# ---------- Config de base ----------

# Racine du projet (remonte depuis scripts/...)
ROOT = Path(__file__).resolve().parents[1]

# Dossiers d'entrée / sortie
SWORD_ROOT = ROOT / "kopis_data" / "sword"
OUT_ROOT = ROOT / "kopis_data" / "ml" / "unet"
CONTACTS_ROOT = OUT_ROOT / "contacts"
DOMAINS_ROOT = OUT_ROOT / "domains"

# On utilise la matrice de probabilités résidu–résidu
CONTACT_MATRIX_FILENAME = "file_proba_contact.mat"


# ---------- Loaders ----------

def load_proba_matrix(txt_path: Path) -> np.ndarray:
    """
    Charge file_proba_contact.mat : matrice dense de probabilités N×N (float32).

    Format observé :
      - 1 ligne de titre "Contact probability matrix"
      - puis N lignes, chacune avec N valeurs flottantes séparées par des espaces.
    """
    rows = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # on saute la/les lignes de titre sans chiffres
            if not any(c.isdigit() for c in line):
                continue
            rows.append([float(x) for x in line.split()])

    if not rows:
        raise ValueError(f"Aucune ligne numérique trouvée dans {txt_path}")

    mat = np.array(rows, dtype=np.float32)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrice non carrée dans {txt_path}: {mat.shape}")

    return mat


def parse_domains_from_peeling(peeling_path: Path):
    """
    Parse Peeling.log et retourne une liste de domaines [(start, end), ...]
    avec indices 1-based (comme SWORD2).

    Hypothèse (vérifiée sur tes fichiers) :
      - les lignes sont triées par CI croissant,
      - la DERNIÈRE ligne = meilleure solution (CI maximal).
    On prend donc la dernière ligne de données.
    """
    with open(peeling_path) as f:
        # garder seulement les lignes non vides et non commentées
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    if not lines:
        return []

    header = lines[0].split()
    data_lines = [l.split() for l in lines[1:]]
    if not data_lines:
        return []

    # On prend simplement la dernière ligne de Peeling.log
    best = data_lines[-1]

    # Colonnes attendues :
    # Max_CR, Min_Density, CI, R, Num_PUs, PU_Delineations...
    try:
        num_pus = int(best[4])
        bounds = list(map(int, best[5:5 + 2 * num_pus]))
    except (IndexError, ValueError):
        return []

    if len(bounds) != 2 * num_pus:
        # format étrange, on abandonne proprement
        return []

    domains = [(bounds[2 * k], bounds[2 * k + 1]) for k in range(num_pus)]
    # trier par position de début (par sécurité)
    domains.sort(key=lambda d: d[0])
    return domains


# ---------- Main ----------

def main():
    # Création des dossiers de sortie
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONTACTS_ROOT.mkdir(parents=True, exist_ok=True)
    DOMAINS_ROOT.mkdir(parents=True, exist_ok=True)

    if not SWORD_ROOT.exists():
        raise SystemExit(f"Dossier inexistant : {SWORD_ROOT}")

    protein_dirs = sorted([d for d in SWORD_ROOT.iterdir() if d.is_dir()])
    total = len(protein_dirs)
    print(f"Trouvé {total} dossiers de protéines dans {SWORD_ROOT}")

    n_ok = 0

    for idx, prot_dir in enumerate(protein_dirs, start=1):
        uniprot_id = prot_dir.name
        print(f"[{idx}/{total}] Traitement {uniprot_id}...")

        # 1) chercher Peeling.log (où qu'il soit sous ce dossier)
        peeling_files = list(prot_dir.glob("**/Peeling/Peeling.log"))
        if not peeling_files:
            print("  -> Pas de Peeling.log trouvé, on saute.")
            continue
        peeling_path = peeling_files[0]

        # 2) extraire les domaines à partir de la dernière ligne
        domains = parse_domains_from_peeling(peeling_path)
        if not domains:
            print("  -> Pas de domaines valides dans Peeling.log, on saute.")
            continue

        # 3) retrouver le dossier contenant la matrice de probas
        # Peeling.log est en .../PDBs_Clean/XXXX/Peeling/Peeling.log
        # La matrice est en .../PDBs_Clean/XXXX/file_proba_contact.mat
        clean_dir = peeling_path.parents[1]  # le dossier "XXXX"
        proba_path = clean_dir / CONTACT_MATRIX_FILENAME

        if not proba_path.exists():
            print(f"  -> Pas de {CONTACT_MATRIX_FILENAME}, on saute.")
            continue

        # 4) charger la matrice de probas
        try:
            contact = load_proba_matrix(proba_path)
        except Exception as e:
            print(f"  -> Erreur lors du chargement de la matrice de probas: {e}")
            continue

        N = contact.shape[0]

        # 5) construire la domain_map N×N
        domain_map = np.zeros((N, N), dtype=np.uint8)

        out_of_bounds = False
        for start, end in domains:
            # indices SWORD2 (1-based, inclusifs) -> indices Python [start-1 : end]
            if start < 1 or end > N or start > end:
                out_of_bounds = True
                break
            i0 = start - 1
            i1 = end
            domain_map[i0:i1, i0:i1] = 1

        if out_of_bounds:
            print("  -> Bornes de domaines hors limites par rapport à la matrice, on saute.")
            continue

        # 6) sauvegarde des fichiers .npy dans des sous-dossiers séparés
        out_contacts = CONTACTS_ROOT / f"{uniprot_id}.npy"
        out_domains = DOMAINS_ROOT / f"{uniprot_id}.npy"

        np.save(out_contacts, contact)
        np.save(out_domains, domain_map)

        n_ok += 1
        print(f"  -> OK, sauvegardé : {out_contacts}, {out_domains}")

    print(f"\nTerminé : {n_ok}/{total} protéines traitées avec succès.")


if __name__ == "__main__":
    main()
    
 
