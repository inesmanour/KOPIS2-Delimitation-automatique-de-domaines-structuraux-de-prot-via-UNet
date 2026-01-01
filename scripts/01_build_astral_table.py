#!/usr/bin/env python
import argparse
import csv
from pathlib import Path

def parse_astral_header(header: str):
    """
    Exemple de header ASTRAL :
    >d1dlwa1 a.1.1.0 (A:) Protozoan/bacterial hemoglobin ...

    On veut extraire :
      - astral_id  = d1dlwa1
      - pdb_id     = 1dlw
      - chain_id   = A
      - domain_id  = 1
      - scop_class = a.1.1.0
    """
    header = header.strip()
    if header.startswith(">"):
        header = header[1:]  # enlever '>'

    parts = header.split()
    astral_id = parts[0]              # d1dlwa1
    scop_class = parts[1] if len(parts) > 1 else ""

    # d1dlwa1 : d + 1dlw + a + 1
    pdb_id = astral_id[1:5]           # 1dlw
    chain_id = astral_id[5] if len(astral_id) > 5 else ""
    domain_id = astral_id[6:] if len(astral_id) > 6 else ""

    return astral_id, pdb_id, chain_id.upper(), domain_id, scop_class

def build_astral_table(fasta_path: Path, out_tsv: Path):
    with fasta_path.open() as f, out_tsv.open("w", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["astral_id", "pdb_id", "chain_id", "domain_id", "scop_class"])

        for line in f:
            # on ne s'intéresse qu'aux lignes qui commencent par '>'
            if not line.startswith(">"):
                continue
            astral_id, pdb_id, chain_id, domain_id, scop_class = parse_astral_header(line)
            writer.writerow([astral_id, pdb_id, chain_id, domain_id, scop_class])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        
        default=Path("kopis_data/astral/astral_raw.fa"),
        help="Fichier FASTA ASTRAL brut",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kopis_data/astral/astral_cleaned.tsv"),
        help="Table TSV de sortie",
    )
    args = parser.parse_args()

    build_astral_table(args.input, args.output)
    print(f"Table ASTRAL écrite dans {args.output}")