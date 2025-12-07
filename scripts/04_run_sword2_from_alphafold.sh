#!/usr/bin/env bash
set -u              # on enlève -e pour ne pas quitter à la 1ère erreur
set -o pipefail

# Dossier où sont les mmCIF AlphaFold
AF_DIR="kopis_data/alphafold"
# Dossier de sortie pour SWORD2
SWORD_DIR="kopis_data/sword"

# Chemin vers SWORD2.py (à adapter si besoin)
SWORD2_SCRIPT="$HOME/SWORD2/SWORD2.py"

mkdir -p "$SWORD_DIR"

# On boucle sur tous les .cif
for cif_path in "${AF_DIR}"/*.cif; do
  # si aucun fichier ne matche, bash laisse le motif brut
  if [ ! -f "$cif_path" ]; then
    echo "Aucun .cif trouvé dans ${AF_DIR}, vérifie le dossier."
    break
  fi

  cif_file=$(basename "$cif_path")

  # Cas 1 : fichiers de la forme AF-A0A0F6CKR5-F1-model_v6.cif
  if [[ "$cif_file" == AF-* ]]; then
    tmp=${cif_file#AF-}      # enlève "AF-"
    tmp=${tmp%%.*}           # enlève tout après le premier "." (F1-model_v6.cif)
    uniprot_id=${tmp%%-*}    # garde tout avant le premier "-"
  else
    # Cas 2 : fichiers de la forme A0A017T5A5.cif
    uniprot_id=${cif_file%.cif}   # enlève juste .cif
  fi

  out_dir="${SWORD_DIR}/${uniprot_id}"

  # 🔴 SI LA PROT A DÉJÀ DES RÉSULTATS, ON SAUTE
  if [ -f "${out_dir}/SWORD2_summary.json" ]; then
    echo "[${uniprot_id}] déjà traité (SWORD2_summary.json trouvé), on saute."
    continue
  fi

  echo ">>> Traitement ${uniprot_id} avec SWORD2..."
  mkdir -p "$out_dir"

  # On exécute SWORD2 *dans* le dossier de la protéine
  (
    cd "$out_dir"
    python "$SWORD2_SCRIPT" \
      -i "../../alphafold/${cif_file}" \
      -o "." \
      --disable-energies \
      --disable-plots
  )
  status=$?

  if [ $status -ne 0 ]; then
    echo "[${uniprot_id}] ATTENTION : SWORD2 a renvoyé le code ${status}"
  fi

done