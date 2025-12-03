# KOPIS2-D-limitation-automatique-de-domaines-structuraux-de-prot-ines-via-U-Net-


Ce dépôt contient le code du projet **KOPIS2**, encadré par le Pr. Jean-Christophe Gelly (INSERM UMR_S1134 / Université Paris Cité).

L’objectif est de développer une approche d’apprentissage profond (U-Net) pour **segmenter les domaines structuraux de protéines** à partir de leurs **cartes de contacts**, en utilisant comme vérité terrain les domaines définis par **SWORD2**.


---

##  Objectif du projet

- Utiliser des structures 3D (PDB / modèles AlphaFold) pour générer ou récupérer des **cartes de contacts** entre résidus.
- Faire tourner **SWORD2** afin d’obtenir :
  - la **segmentation en domaines** (éventuellement discontinus),
  - les **matrices de contacts** (distances / probabilités).
- Construire, pour chaque protéine, des paires :
  - **entrée U-Net** : matrice de contacts N×N,
  - **label U-Net** : matrice N×N binaire indiquant si deux résidus sont dans le même domaine.
- Entraîner un modèle de type **U-Net (CNN 2D)** pour **imiter SWORD2** à partir des cartes de contacts.
- Évaluer la qualité des prédictions par rapport à SWORD2 (qualité des domaines, frontières, etc.).

---

##  Organisation du dépôt

### Code

```text
.
├── README.md                      # ce fichier
├── environment.yml                # définition de l'environnement (conda/pixi)
├── .gitignore                     # fichiers à exclure de Git
│
├── scripts/                       # scripts "pipeline" (préparation des données)
│   ├── 01_build_astral_table.py   # ASTRAL -> table propre (pdb_id, chain_id, etc.)
│   ├── 02_join_astral_sifts.py    # jointure ASTRAL + SIFTS -> ajout uniprot_id
│   ├── 03_download_alphafold.py   # téléchargement des modèles AlphaFold (mmCIF)
│   ├── 04_run_sword2_batch.sh     # lancement de SWORD2 sur toutes les protéines
│   └── 05_make_unet_data.py       # préparation des données pour l'U-Net
│
├── src/
│   ├── models/
│   │   └── unet.py                # définition du modèle U-Net (segmentation 2D)
│   │
│   ├── training/
│   │   └── train_unet.py          # boucle d'entraînement U-Net
│   │
│   └── utils/
│       ├── io.py                  # fonctions d'entrées/sorties
│       └── sword_parsing.py       # parsing des sorties SWORD2 (contacts + domaines)
│
├── notebooks/
│   ├── 00_exploration.ipynb       # exploration des données ASTRAL / SIFTS
│   └── 01_visualiser_contact_maps.ipynb
│
└── results/
    └── unet/                      # résultats, logs, modèles U-Net
```
--

## Arborescence des données (hors dépôt)

Les données sont stockées en dehors du dépôt (ou dans un dossier gitignoré), par exemple :

```text
kopis_data/
  astral/
    astral_raw.fa                  # fichier ASTRAL brut (séquences/domaines)
    astral_cleaned.tsv             # table propre : astral_id, pdb_id, chain_id, ...
  sifts/
    pdb_chain_uniprot.tsv.gz       # mapping officiel PDB–UniProt (SIFTS)
    mapping_pdb_uniprot.tsv        # jointure ASTRAL + SIFTS
  alphafold/
    Q9XXXXX.cif                    # modèles AlphaFold (mmCIF)
    Q8YYYYY.cif
    ...
  sword/
    Q9XXXXX/
      contacts.txt                 # matrice N×N (distance ou contact)
      contacts_prob.txt            # matrice N×N (probabilité de contact)
      domains.txt                  # domaines SWORD2 (bornes, PU, etc.)
    ...
  ml/
    unet/
      Q9XXXXX_contacts.npy         # matrice N×N entrée U-Net
      Q9XXXXX_domain_map.npy       # matrice N×N label U-Net (même domaine / pas même domaine)
      ...
```


Ce répertoire contient :
	•	les fichiers ASTRAL/SCOPe nettoyés,
	•	les mappings SIFTS (PDB ↔ UniProt),
	•	les structures AlphaFold au format .cif,
	•	les sorties SWORD2 (cartes de contacts, domaines),
	•	les données prêtes pour l’apprentissage de l’U-Net.

## Installation de l’environnement

### Exemple avec conda  :

```conda env create -f environment.yml
conda activate kopis2
```

L’environnement doit inclure typiquement :
	•	numpy, pandas
	•	pytorch, torchvision
	•	éventuellement biopython (parsing PDB/mmCIF)
	•	toute dépendance nécessaire pour lancer SWORD2 (en dehors de ce dépôt).



## Pipeline (résumé, côté scripts)
	1.	Préparation des listes de domaines (ASTRAL)
01_build_astral_table.py
    → astral_raw.fa → astral_cleaned.tsv
    (table contenant astral_id, pdb_id, chain_id, etc.)
	
    2.	Mapping PDB → UniProt (SIFTS)
02_join_astral_sifts.py
    → astral_cleaned.tsv + pdb_chain_uniprot.tsv.gz
    → mapping_pdb_uniprot.tsv (ajout de uniprot_id)
	
    3.	Téléchargement des structures AlphaFold
03_download_alphafold.py
    → pour chaque uniprot_id, téléchargement du modèle .cif dans kopis_data/alphafold/.
	
    4.	Lancement de SWORD2
04_run_sword2_batch.sh
    → boucle sur les .cif d’AlphaFold
    → pour chaque protéine : kopis_data/sword/UNIPROTID/ avec :
	•	cartes de contacts (distance / proba),
	•	fichiers de domaines SWORD2.
	
    5.	Préparation des données U-Net
05_make_unet_data.py
    → lit les matrices de contacts SWORD2 + domaines SWORD2
    → génère pour chaque protéine :
	•	UNIPROTID_contacts.npy (entrée U-Net),
	•	UNIPROTID_domain_map.npy (label U-Net)
stockés dans kopis_data/ml/unet/.
	
    
    6.	Entraînement de l’U-Net
src/training/train_unet.py
	•	charge les .npy dans kopis_data/ml/unet/,
	•	sépare les protéines en train/validation/test,
	•	entraîne le modèle défini dans src/models/unet.py.


## Définition de la tâche U-Net

Pour une protéine de longueur N :
	
    •	Entrée du modèle (X) :
une carte de contacts contact_map de taille N×N, vue comme une image 2D
(par ex. tensor de forme [1, N, N]).
	
    •	Label (Y) :
une domain map domain_map de taille N×N telle que :
	
        -	domain_map[i, j] = 1 si les résidus i et j appartiennent au même domaine (selon SWORD2),
	    -	0 sinon.
	
    •	Tâche :
apprentissage supervisé de type segmentation binaire 2D :
	
    -	le U-Net doit prédire, pour chaque paire (i, j), si elle est dans le même domaine.








Projet de Master 2 Bioinformatique (Université Paris Cité)
Encadrement : Pr. Jean-Christophe Gelly, Dr. Yasser Mohseni
