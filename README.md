#  KOPIS2 Délimitation automatique de domaines structuraux de protéines via U-Net


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
│   ├── 04_run_sword2_from_alphafold.sh     # lancement de SWORD2 sur toutes les protéines
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

## Arborescence des données 

 À noter que les données complètes sont stockées dans le .gitignore au vu du volume :

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
      AF-Q9XXXXX_A/
        PDBs_Clean/
          Peeling/
            Peeling.log
          file_ca_coo.pdb
          file_matrix_pu_contact.mtx
          file_proba_contact.mat
          file_pu_declineation.mtx
          XXXX.dsp
          XXXX.num
          XXXX.s2d
        PDBs_Stand/ XXXX
        sword.err
        sword.txt
        SWORD2_summary.json
        SWORD2_summary.txt
        sword2.log
        XXXX
      
      .fasta
      A.pdb
    ...
  ml/
    unet/
        contacts/ 
        Q9XXXXX.npy        
        domains/
        Q9XXXXX.npy       
      ...
```


Ce répertoire contient :
	•	les fichiers ASTRAL/SCOPe nettoyés,
	•	les mappings SIFTS (PDB ↔ UniProt),
	•	les structures AlphaFold au format .cif,
	•	les sorties SWORD2 (cartes de contacts, domaines),
	•	les données prêtes pour l’apprentissage de l’U-Net.


## Arborescence final :

```
~/KOPIS2-Delimitation-automatique-de-domaines-structuraux-de-prot-via-UNet/
  README.md
  environment.yml
  .gitignore

  scripts/
    01_build_astral_table.py
    02_join_astral_sifts.py
    03_download_alphafold.py
    04_run_sword2_from_alphafold.sh
    05_make_unet_data.py

  src/
    models/
      unet.py
    training/
      train_unet.py
      dataset.py

  notebooks/
    affichage_contact&domain.ipynb # visualisation contact_map + domain_map
    03_eval_unet.ipynb # évaluation du modèle U-Net (métriques + figures)

  results/
    unet/
      best_model.pt
      training_log.csv

  kopis_data/
    astral/
    sifts/
    alphafold/
    sword/
    ml/
    
```


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

###### En résumé : ASTRAL → SIFTS → AlphaFold (mmCIF) → SWORD2 (contacts + domaines) → préparation des matrices → entraînement + évaluation du U-Net.


	1.	Préparation des listes de domaines (ASTRAL)
scripts/01_build_astral_table.py
→ astral_raw.fa → astral_cleaned.tsv
(table contenant astral_id, pdb_id, chain_id, etc.)
	
    2.	Mapping PDB → UniProt (SIFTS)
scripts/02_join_astral_sifts.py
→ astral_cleaned.tsv + pdb_chain_uniprot.tsv.gz
→ mapping_pdb_uniprot.tsv (ajout de uniprot_id).
	
    3.	Téléchargement des structures AlphaFold
scripts/03_download_alphafold.py
→ pour chaque uniprot_id, téléchargement du modèle .cif dans kopis_data/alphafold/.
	
    4.	Lancement de SWORD2
scripts/04_run_sword2_from_alphafold.sh
→ boucle sur les .cif d’AlphaFold
→ pour chaque protéine : kopis_data/sword/UNIPROTID/ avec par exemple :
	•	matrices de contacts résidu–résidu (probabilités),
	•	fichiers de délimitation de domaines SWORD2.
	
    5.	Préparation des données U-Net
scripts/05_make_unet_data.py
→ lit, pour chaque protéine, la matrice de probas de contact + les domaines SWORD2 (Peeling.log)
→ génère pour chaque UNIPROTID :
	•	kopis_data/ml/unet/contacts/UNIPROTID.npy (entrée U-Net, matrice N×N de probas),
	•	kopis_data/ml/unet/domains/UNIPROTID.npy (label U-Net, matrice N×N binaire “même domaine / pas même domaine”).
	
    
    6.	Entraînement du U-Net
src/training/train_unet.py
	•	lit la liste des protéines disponibles dans kopis_data/ml/unet/contacts/ et domains/,
	•	crée un split train / validation / test (fichiers dans kopis_data/ml/unet/splits/),
	•	instancie le modèle défini dans src/models/unet.py,
	•	utilise le dataset ContactDomainDataset de src/training/dataset.py pour charger les paires (contact_map, domain_map),
	•	entraîne le U-Net (boucle d’epochs, loss, optimiseur, callbacks),
	•	sauvegarde le meilleur modèle dans results/unet/best_model.pt + les logs d’apprentissage.




## Composants du modèle U-Net

•	src/models/unet.py
Contient l’implémentation du modèle U-Net 2D (UNet2D).
	
  •	Entrée : un tenseur [B, 1, N, N] correspondant à une carte de contacts.
	
  •	Sortie : un tenseur [B, 1, N, N] de logits, interprétés comme la probabilité que chaque paire (i, j) soit dans le même domaine.
L’architecture encode le contexte global (taille des domaines, interactions à longue distance) puis reconstruit une “image” de mêmes dimensions via un décodeur avec skip connections.
	
  

•	src/training/dataset.py
Définit la classe ContactDomainDataset, qui fait le lien entre les fichiers .npy et PyTorch :
	
  •	lit les matrices dans
kopis_data/ml/unet/contacts/UNIPROTID.npy (X)
kopis_data/ml/unet/domains/UNIPROTID.npy (Y),
	
  •	renvoie pour chaque index un couple (contact_map, domain_map) déjà au bon format tensoriel,
	
  •	est utilisée à la fois par train_unet.py et par les notebooks d’évaluation.
	

•	src/training/train_unet.py
Script principal d’entraînement :
	
  •	crée les DataLoader (train_loader, val_loader) à partir de ContactDomainDataset,
	
  •	configure l’optimiseur, la loss (par ex. BCEWithLogitsLoss) et les callbacks (early stopping, sauvegarde du meilleur modèle),
	
  •	exécute la boucle d’apprentissage et suit les métriques,
	
  • produit les artefacts dans results/unet/ :
  - best_model.pt (meilleur modèle sur la validation),
  - training_log.csv (log d’apprentissage : loss, acc, F1 par epoch, etc.),
  - les courbes (loss / F1 train-val) sont tracées dans le notebook 03_eval_unet.ipynb.

## Notebook d’évaluation

•	notebooks/03_eval_unet.ipynb
Notebook dédié à l’analyse des résultats :
	
  •	recharge le modèle entraîné (results/unet/best_model.pt),
	
  •	utilise ContactDomainDataset pour parcourir le test set,
	
  •	calcule des métriques  (accuracy, F1) en comparant les prédictions du U-Net aux cartes de domaines SWORD2,
	
  •	visualise pour quelques protéines :
	
  •	la contact map (entrée),
	
  •	la domain_map SWORD2 (vérité terrain),
	
  •	la domain_map prédite par le U-Net (après sigmoid + threshold).

  •	trace ROC+AUC , matrice de confusion , courbe train/val 
  




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
Encadrement : Pr. Jean-Christophe Gelly, Dr. Yasser Mohseni , Gabriel Cretin 
