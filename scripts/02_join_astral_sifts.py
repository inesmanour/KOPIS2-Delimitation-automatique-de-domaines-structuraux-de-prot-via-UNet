#!/usr/bin/env python
import argparse
import csv
import gzip
from pathlib import Path


def open_maybe_gz(path: Path):
    """
    Ouvre un fichier normal ou .gz en mode texte.
    """
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    else:
        return path.open("r")
 

def load_sifts_mapping(sifts_path: Path):
    """
    Lit pdb_chain_uniprot.tsv(.gz) et retourne un dict :
        (pdb_id, chain_id) -> set(uniprot_ids)
    """

    mapping = {}

    with open_maybe_gz(sifts_path) as f:
        # trouver la vraie ligne d'en-tête (celle qui contient "PDB")
        header = None
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                # ligne de commentaire, on saute
                continue
            header = line.strip().split("\t")
            break

        if header is None:
            raise RuntimeError(f"Aucune ligne d'en-tête trouvée dans {sifts_path}")

        try:
            pdb_idx = header.index("PDB")
            chain_idx = header.index("CHAIN")
            uniprot_idx = header.index("SP_PRIMARY")
        except ValueError as e:
            raise RuntimeError(
                f"Colonnes attendues 'PDB', 'CHAIN', 'SP_PRIMARY' introuvables dans {sifts_path}"
            ) from e

        # maintenant on lit le reste du fichier (les vraies données)
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.strip().split("\t")
            if len(parts) <= max(pdb_idx, chain_idx, uniprot_idx):
                continue

            pdb_id = parts[pdb_idx].lower()  # ex : 1DLW -> 1dlw
            chain_id = parts[chain_idx]
            uniprot_id = parts[uniprot_idx]

            key = (pdb_id, chain_id)
            if uniprot_id:
                mapping.setdefault(key, set()).add(uniprot_id)

    return mapping


def join_astral_sifts(astral_tsv: Path, sifts_path: Path, out_tsv: Path):
    """
    Joint astral_cleaned.tsv avec SIFTS pour ajouter les uniprot_ids.
    """

    print(f"Chargement du mapping SIFTS depuis {sifts_path} ...")
    sifts_map = load_sifts_mapping(sifts_path)
    print(f"  -> {len(sifts_map)} couples (PDB, CHAINE) trouvés dans SIFTS.")

    with astral_tsv.open() as fin, out_tsv.open("w", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = reader.fieldnames + ["uniprot_ids"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        n_total = 0
        n_matched = 0

        for row in reader:
            n_total += 1
            key = (row["pdb_id"].lower(), row["chain_id"])
            uniprot_ids = sifts_map.get(key, set())
            if uniprot_ids:
                n_matched += 1
                row["uniprot_ids"] = ",".join(sorted(uniprot_ids))
            else:
                row["uniprot_ids"] = ""
            writer.writerow(row)

    print(f"Jointure terminée : {n_matched}/{n_total} domaines avec au moins un UniProt.")
    print(f"Fichier écrit dans {out_tsv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Joint ASTRAL et SIFTS (PDB+CHAINE -> UniProt)"
    )
    parser.add_argument(
        "--astral",
        type=Path,
        default=Path("kopis_data/astral/astral_cleaned.tsv"),
        help="Table ASTRAL nettoyée (TSV)",
    )
    parser.add_argument(
        "--sifts",
        type=Path,
        
        default=Path("kopis_data/sifts/pdb_chain_uniprot.tsv"),
        help="Fichier SIFTS pdb_chain_uniprot (.tsv ou .tsv.gz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kopis_data/sifts/mapping_pdb_uniprot.tsv"),
        help="Table de sortie avec colonne uniprot_ids",
    )
    args = parser.parse_args()

    join_astral_sifts(args.astral, args.sifts, args.output)