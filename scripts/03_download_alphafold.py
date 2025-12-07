#!/usr/bin/env python
import argparse
import csv
import time
from pathlib import Path

import requests


def load_uniprot_ids(mapping_tsv: Path):
    """
    Lit mapping_pdb_uniprot.tsv et renvoie un set d'UniProt IDs uniques.
    """
    uniprots = set()
    with mapping_tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids = row.get("uniprot_ids", "")
            if not ids:
                continue
            for uid in ids.split(","):
                uid = uid.strip()
                if uid:
                    uniprots.add(uid)
    return uniprots


def fetch_cif_url_from_af_api(uniprot_id: str):
    """
    Utilise l'API officielle AlphaFold :
      GET https://alphafold.ebi.ac.uk/api/prediction/<uniprot_id>

    La réponse est une liste de prédictions ; on prend la première
    et on récupère l'URL du mmCIF (cifUrl ou mmcifUrl).
    """
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        resp = requests.get(api_url, timeout=30)
    except Exception as e:
        print(f"[{uniprot_id}] ERREUR API : {e}")
        return None

    if resp.status_code != 200:
        # pas de prédiction pour cet UniProt
        # (404 = rien dans AlphaFold DB)
        # on renvoie None
        return None

    try:
        data = resp.json()
    except Exception as e:
        print(f"[{uniprot_id}] ERREUR JSON API : {e}")
        return None

    if not data:
        return None

    entry = data[0]  # on prend la première prédiction
    # suivant la version de l'API, le champ peut s'appeler cifUrl ou mmcifUrl
    cif_url = entry.get("cifUrl") or entry.get("mmcifUrl")
    return cif_url


def download_alphafold_model(uniprot_id: str, out_dir: Path, sleep: float = 0.2):
    """
    Télécharge le modèle AlphaFold (mmCIF) via l'API.

    - Interroge l'API /api/prediction/<uniprot_id>
    - Récupère l'URL du mmCIF
    - Télécharge le fichier dans out_dir
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # nom de fichier local : AF-<UNIPROT>.cif (on ne se complique pas avec v4/v6)
    out_path = out_dir / f"AF-{uniprot_id}.cif"
    if out_path.exists():
        return "exists"

    cif_url = fetch_cif_url_from_af_api(uniprot_id)
    if not cif_url:
        # pas de modèle dispo dans AF DB
        print(f"[{uniprot_id}] Aucun mmCIF trouvé via l'API AlphaFold.")
        return "missing"

    try:
        resp = requests.get(cif_url, timeout=60)
    except Exception as e:
        print(f"[{uniprot_id}] ERREUR téléchargement CIF : {e}")
        return "error"

    if resp.status_code == 200:
        out_path.write_bytes(resp.content)
        time.sleep(sleep)
        return "ok"
    else:
        print(f"[{uniprot_id}] HTTP {resp.status_code} pour {cif_url}")
        return "error"


def main(mapping_tsv: Path, out_dir: Path, max_proteins=None):
    """
    max_proteins : int ou None
      - None -> tous les UniProt
      - sinon -> on limite au nombre donné (ex : 50 pour tests)
    """
    uniprots = sorted(load_uniprot_ids(mapping_tsv))
    print(f"{len(uniprots)} UniProt uniques trouvés dans {mapping_tsv}")

    if max_proteins is not None:
        uniprots = uniprots[:max_proteins]
        print(f"On limite le téléchargement aux {len(uniprots)} premiers pour les tests.")

    n_ok = n_exists = n_missing = n_error = 0

    for i, uid in enumerate(uniprots, start=1):
        status = download_alphafold_model(uid, out_dir)
        if status == "ok":
            n_ok += 1
        elif status == "exists":
            n_exists += 1
        elif status == "missing":
            n_missing += 1
        else:
            n_error += 1

        if i % 50 == 0:
            print(f"{i}/{len(uniprots)} traités...")

    print("Résumé téléchargement AlphaFold :")
    print(f"  Nouveaux fichiers : {n_ok}")
    print(f"  Déjà présents   : {n_exists}")
    print(f"  Sans modèle AF  : {n_missing}")
    print(f"  Erreurs         : {n_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Télécharge les modèles AlphaFold (mmCIF) à partir du mapping ASTRAL+SIFTS."
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("kopis_data/sifts/mapping_pdb_uniprot.tsv"),
        help="Fichier TSV issu de 02_join_astral_sifts.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("kopis_data/alphafold"),
        help="Dossier de sortie pour les fichiers mmCIF",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=50,  # 50 pour tester,  mettre 0 ou None pour tout
        help="Nombre max d'UniProt à télécharger (0 -> tous)",
    )
    args = parser.parse_args()

    max_proteins = args.max_proteins
    if max_proteins == 0:
        max_proteins = None

    main(args.mapping, args.out_dir, max_proteins)