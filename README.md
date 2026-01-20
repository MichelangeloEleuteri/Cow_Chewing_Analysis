# Cow_Chewing_Analysis
## Quick demo (come vedere i risultati rapidamente)
1. Genera il CSV delle distanze dalle label YOLO-Pose:
python scripts/compute_d_from_labels.py --labels_dir <PERCORSO_LABELS> --out results/dists_all.csv --width 1080 --height 1920 --fps 30 --plot results/dists_all.png
2. Apri `notebooks/Chew_detection_with_nose_mouth_distance.ipynb` e lancia la cella “Quick Demo” (prima cella) per vedere il grafico della traccia distanza e le metriche principali.
3. Se non hai i dati, vedi `data/README.md` per istruzioni su come ottenerli.
