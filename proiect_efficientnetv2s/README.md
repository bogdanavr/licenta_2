# Proiect nou - EfficientNetV2-S pentru 4 emotii

Acest proiect reproduce structura generala folosita in proiectul de licenta din folderul `licenta`, dar este adaptat pentru un nou dataset aflat in `licenta2/archive`.

In proiectul de licenta initial modelul a fost antrenat pe `FER2013`. In aceasta varianta noua, modelul este antrenat pe datasetul din `licenta2`, filtrat la 4 clase:

- `happy`
- `neutral`
- `sad`
- `surprise`

Modelul folosit este `EfficientNetV2-S`, cu greutati pre-antrenate pe ImageNet si un cap de clasificare modificat pentru cele 4 emotii. Intrarea este redimensionata la `384x384`, in linie cu preprocess-ul oficial torchvision pentru aceasta arhitectura.

## Structura proiectului

```text
proiect_efficientnetv2s/
├── data/
│   └── processed_4classes/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── meta.json
├── export/
├── logs/
├── train/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── prepare_dataset.py
│   ├── test.py
│   ├── test_model.py
│   └── train.py
└── requirements.txt
```

## Ce face proiectul

1. Citeste `combined_labels.csv` din `licenta2/archive`.
2. Filtreaza doar etichetele `happy`, `neutral`, `sad`, `surprise`.
3. Construieste automat split-urile `train`, `val`, `test` in mod stratificat.
4. Detecteaza fata in fiecare imagine si salveaza crop-ul facial in `data/processed_4classes/images`.
5. Daca nu detecteaza o fata, aplica un `center square crop` ca fallback.
6. Antreneaza un model `EfficientNetV2-S` pentru clasificarea celor 4 emotii.
7. Evalueaza modelul pe setul de test si salveaza metrici si matricea de confuzie.

## De ce folosim `combined_labels.csv`

Datasetul din `licenta2/archive` contine atat imagini originale, cat si imagini augmentate, iar fisierul `combined_labels.csv` reprezinta referinta autoritara pentru etichete si localizarea imaginilor. Conform descrierii datasetului, structura originala a folderelor contine inconsistenta, deci pregatirea subsetului foloseste CSV-ul pentru `filepath`, `label` si `source`, nu doar numele folderelor.

Dupa preprocesare nu mai este nevoie de manifeste CSV pentru incarcare. Datasetul de antrenare este citit direct din folderele finale `train/val/test/<eticheta>`, iar `meta.json` ramane doar pentru statistici si trasabilitate.

Implicit sunt folosite atat exemplele `original`, cat si cele `augmented`, deoarece setul rezultat este echilibrat pentru cele 4 clase.

## Pasi de rulare

Din folderul proiectului:

```bash
python -m train.prepare_dataset
python -m train.train
python -m train.test
```

Pentru testarea unei imagini individuale:

```bash
python -m train.test_model --image /cale/catre/imagine.jpg
```

## Fisiere rezultate

- `data/processed_4classes/meta.json` - rezumatul datasetului procesat
- `export/best_model.pth` - cel mai bun checkpoint salvat dupa validare
- `logs/train_history.json` - istoricul antrenarii
- `logs/train_log.csv` - metrici pe epoca
- `logs/classification_report.json` - raportul pe setul de test
- `logs/confusion_matrix.png` - matricea de confuzie

## Observatie pentru documentatia de licenta

Formulare utila pentru a descrie diferenta fata de proiectul anterior:

> In proiectul initial de licenta, modelul a fost antrenat pe datasetul FER2013. In aceasta extensie am construit un proiect separat, in folderul `licenta2`, unde am antrenat un model EfficientNetV2-S pe un nou dataset de expresii faciale, disponibil local in `licenta2/archive`, restrans la clasele happy, neutral, sad si surprise. Pentru preprocesare am folosit `combined_labels.csv` ca referinta autoritara si am salvat imagini deja decupate pe fata in split-urile finale de antrenare, validare si testare.
