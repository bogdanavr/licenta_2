# licenta2 - clasificarea a 4 emotii faciale cu EfficientNet-B0 si export pentru Hailo

Acest proiect implementeaza un pipeline complet pentru clasificarea expresiilor faciale in 4 clase: `happy`, `neutral`, `sad` si `surprise`, pornind de la datasetul local `AffectNet Augmented` aflat in `licenta2/archive`.

Repository-ul este gandit ca un proiect end-to-end:
- pregatirea datasetului dintr-o sursa bruta cu etichete centralizate in CSV
- preprocesarea focalizata pe fata
- organizarea datasetului in split-uri `train`, `val`, `test`
- antrenarea unui model `EfficientNet-B0`
- evaluarea modelului cu mai multe metrici relevante
- inferenta pe imagini individuale
- export si compilare `ONNX -> HEF` pentru acceleratoare Hailo folosite in ecosistemul Raspberry Pi AI / NPU

README-ul este scris cu mult context, astfel incat proiectul sa poata fi folosit si ca documentatie tehnica sau context pentru un prompt ChatGPT.

## Obiectivul proiectului

Scopul principal al proiectului este construirea unui clasificator robust de expresii faciale pentru patru emotii fundamentale, pe baza unor imagini faciale provenite dintr-un dataset mai mare si mai eterogen.

Accentul proiectului cade pe:
- consistenta etichetelor
- reducerea zgomotului de fundal prin crop facial
- un pipeline reproductibil
- un model suficient de eficient pentru antrenare pe hardware modest
- compatibilitate ulterioara cu fluxul de deployment pe Hailo

## Dataset folosit

Datasetul sursa este `AffectNet Augmented`, disponibil local in:

```text
licenta2/archive/
```

Acesta contine:
- imagini originale
- imagini augmentate
- fisierul `combined_labels.csv`

In acest proiect, `combined_labels.csv` este tratat ca sursa autoritara pentru etichete si pentru localizarea imaginilor. Aceasta alegere este esentiala deoarece structura folderelor brute poate contine inconsistente intre folderul in care se afla imaginea si eticheta ei reala.

Cu alte cuvinte, pipeline-ul nu presupune ca numele directorului este suficient pentru a decide clasa unei imagini. In schimb, foloseste explicit informatia din coloanele `filepath`, `label` si `source` din CSV.

## Clasele tinta

Modelul este antrenat doar pentru urmatoarele 4 clase:
- `happy`
- `neutral`
- `sad`
- `surprise`

Toate celelalte etichete disponibile in `AffectNet Augmented` sunt ignorate in etapa de pregatire a subsetului final.

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
│   ├── best_model.pth
│   ├── emotion_model.onnx
│   └── compile_hailo.py
├── logs/
├── train/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── model.py
│   ├── prepare_dataset.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── test.py
│   └── test_model.py
├── compile_hailo.py
└── requirements.txt
```

## Rezumatul pipeline-ului

Fluxul complet este urmatorul:

1. Se citesc metadatele din `combined_labels.csv`.
2. Se selecteaza doar imaginile pentru cele 4 emotii tinta.
3. Se filtreaza optional dupa `source` (`original`, `augmented` sau ambele).
4. Se construieste un split stratificat `train / val / test`.
5. Pentru fiecare imagine se incearca detectarea fetei principale.
6. Daca fata este detectata, se face crop pe fata cu o mica marja in jurul ei.
7. Daca detectorul nu gaseste fata, se aplica `center square crop`.
8. Crop-ul rezultat este adus la forma patrata prin padding, pentru a evita deformarea fetei.
9. Imaginea procesata se salveaza offline la `256x256`.
10. In timpul antrenarii, modelul primeste crop-uri finale `224x224`, compatibile cu `EfficientNet-B0`.
11. Modelul este antrenat cu transfer learning, augmentari moderate si selectie a checkpoint-ului pe baza `macro F1`.
12. Modelul este evaluat pe test si poate fi folosit pentru inferenta individuala.
13. Optional, modelul exportat in ONNX poate fi compilat in format `.hef` pentru Hailo.

## De ce `EfficientNet-B0`

Modelul principal al proiectului este `EfficientNet-B0` din `torchvision`.

Alegerea a fost facuta din urmatoarele motive:
- este mai usor si mai rapid decat variantele mai mari precum `EfficientNetV2-S`
- ramane suficient de puternic pentru un task de clasificare faciala pe 4 clase
- are suport bun in `torchvision` si greutati pre-antrenate ImageNet
- foloseste natural input de `224x224`
- este mai potrivit pentru GPU-uri cu memorie limitata
- este o alegere mai realista pentru iteratie rapida si fine-tuning repetat

Pentru acest task, `EfficientNet-B0` ofera un echilibru bun intre cost computațional, viteza si potential de acuratete.

## Dimensiunile imaginilor: de ce `256 -> 224`

Un detaliu important al proiectului este separarea dintre:
- dimensiunea la care este salvata imaginea procesata: `256x256`
- dimensiunea efectiva care intra in model: `224x224`

Aceasta alegere este intentionata.

De ce nu se salveaza direct la `224x224`:
- salvarea la `256x256` lasa putin context in jurul fetei
- la train se poate aplica `RandomCrop(224)` fara a pierde complet flexibilitatea de incadrare
- la evaluare se poate aplica `CenterCrop(224)` intr-un mod stabil si reproductibil
- se evita un framing prea rigid, fixat prea devreme

Pe scurt, `256x256` este dimensiunea de stocare a imaginii preprocesate, iar `224x224` este dimensiunea reala de intrare in `EfficientNet-B0`.

## Detalii pe fisiere

### `train/config.py`

Fisier: [config.py](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/train/config.py)

Acest fisier centralizeaza:
- caile importante din proiect
- configuratia datasetului
- lista claselor tinta
- mapping-ul `label -> index`
- dimensiunile `SAVED_IMG_SIZE = 256` si `IMG_SIZE = 224`
- media si deviatia standard pentru normalizare
- identificatorul modelului si numele de afisare

Este punctul central de configurare pentru restul codului.

### `train/preprocessing.py`

Fisier: [preprocessing.py](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/train/preprocessing.py)

Acest modul contine logica reutilizabila de preprocesare faciala:
- extinderea bounding box-ului facial
- fallback `center square crop`
- transformarea crop-ului intr-o imagine patrata prin `border replicate padding`
- resize-ul final la dimensiunea dorita
- suport atat pentru imagini OpenCV (`BGR`), cat si Pillow (`RGB`)

Acest modul este folosit atat la pregatirea datasetului, cat si la inferenta pe o imagine singulara, pentru a pastra consistenta intre train si inference.

### `train/prepare_dataset.py`

Fisier: [prepare_dataset.py](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/train/prepare_dataset.py)

Acesta este scriptul care transforma datele brute in datasetul final folosit de model.

Ce face concret:
- citeste `combined_labels.csv`
- pastreaza doar clasele `happy`, `neutral`, `sad`, `surprise`
- filtreaza dupa `source`
- construieste split-uri stratificate
- citeste fiecare imagine cu OpenCV
- detecteaza fata principala cu `haarcascade_frontalface_default`
- face crop facial
- aplica fallback daca nu detecteaza fata
- completeaza crop-ul prin padding pentru a evita deformarea
- salveaza imaginile finale la `256x256`
- scrie un `meta.json` cu statistici despre split-uri, surse si rezultate ale detectiei faciale

Acest pas muta mare parte din costul de preprocesare inainte de antrenare si simplifica semnificativ pipeline-ul de intrare al modelului.

### `train/dataset.py`

Fisier: [dataset.py](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/train/dataset.py)

Defineste clasa `EmotionDataset`, care incarca imaginile direct din structura finala:

```text
data/processed_4classes/images/<split>/<label>/...
```

Dupa preprocesare, proiectul nu mai depinde de manifeste CSV pentru antrenare. Structura pe directoare este suficienta pentru incarcare si etichetare.

### `train/model.py`

Fisier: [model.py](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/train/model.py)

Contine functia `build_model()`, care:
- incarca `EfficientNet-B0`
- foloseste greutati ImageNet atunci cand `pretrained=True`
- inlocuieste capul final de clasificare cu unul pentru 4 clase
- aplica un `Dropout` inainte de stratul liniar final

Aceasta structura permite transfer learning direct si fine-tuning complet.

### `train/train.py`

Fisier: [train.py](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/train/train.py)

Este scriptul principal de antrenare.

Include:
- seed pentru reproductibilitate
- `DataLoader` pentru `train` si `val`
- augmentari moderate potrivite pentru expresii faciale
- `CrossEntropyLoss` cu `label_smoothing`
- ponderi pe clase doar daca apare dezechilibru semnificativ
- `AdamW`
- rate de invatare diferite pentru backbone si classifier
- `CosineAnnealingLR`
- mixed precision cu `torch.cuda.amp`
- early stopping
- selectie a checkpoint-ului pe baza `val_macro_f1`

Transformari pentru train:
- `Resize(256, 256)`
- `RandomHorizontalFlip`
- `ColorJitter` moderat
- `RandomAffine` usor
- `RandomCrop(224)`
- `Normalize`

Transformari pentru evaluare:
- `Resize(256, 256)`
- `CenterCrop(224)`
- `Normalize`

Alegerea `macro F1` drept criteriu principal de selectie este mai robusta decat simpla acuratete, pentru ca trateaza clasele mai echilibrat.

### `train/metrics.py`

Fisier: [metrics.py](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/train/metrics.py)

Acest modul centralizeaza calculul metricilor de clasificare si este folosit atat in `train.py`, cat si in `test.py`.

Metricile calculate includ:
- `accuracy`
- `balanced_accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `weighted_f1`
- `top2_accuracy`
- `macro_auc_ovr` atunci cand este posibil

Acest set de metrici ofera o imagine mai completa asupra calitatii modelului decat simpla `accuracy`.

### `train/test.py`

Fisier: [test.py](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/train/test.py)

Acest script incarca checkpoint-ul cel mai bun si il evalueaza pe split-ul `test`.

La final produce:
- `logs/test_metrics.json`
- `logs/classification_report.json`
- `logs/confusion_matrix.png`

Acest pas este separat de validarea din timpul antrenarii si este folosit pentru evaluarea finala a modelului.

### `train/test_model.py`

Fisier: [test_model.py](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/train/test_model.py)

Scriptul este folosit pentru inferenta pe o imagine individuala.

Ce face:
- incarca modelul salvat
- citeste imaginea data prin `--image`
- aplica acelasi pipeline de crop facial ca la preprocesare
- aplica `Resize + CenterCrop + Normalize`
- ruleaza inferenta
- afiseaza top predictiile si daca fata a fost detectata

Acest lucru este important pentru a pastra distributia de intrari cat mai apropiata de cea vazuta la antrenare.

## Metrici si artefacte generate

### `data/processed_4classes/meta.json`

Fisier: [meta.json](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/data/processed_4classes/meta.json)

Contine:
- sursa datasetului
- etichetele tinta
- proportiile split-urilor
- numarul de imagini pe split si pe clasa
- detalii despre preprocesare
- cate imagini au fost procesate prin detectie faciala si cate prin fallback

### `export/best_model.pth`

Fisier: [best_model.pth](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/export/best_model.pth)

Checkpoint-ul principal salveaza:
- greutatile modelului
- tipul modelului
- epoca cea mai buna
- metrica folosita pentru selectie
- valoarea celei mai bune metrici
- dimensiunile relevante ale inputului
- etichetele claselor

### `logs/`

Folderul [logs](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/logs) contine in mod normal:
- `train_log.csv`
- `train_history.json`
- `test_metrics.json`
- `classification_report.json`
- `confusion_matrix.png`

## Export si compilare pentru Hailo

Proiectul include si un flux separat pentru conversia modelului exportat in ONNX catre formatul `.hef` folosit de toolchain-ul Hailo.

Fisiere relevante:
- [compile_hailo.py](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/compile_hailo.py)
- [compile_hailo.py](/home/bogdan_avr/licenta2/proiect_efficientnetv2s/export/compile_hailo.py)

Scriptul de compilare:
- incarca modelul ONNX
- incarca imagini de calibrare din datasetul procesat
- aplica model script-ul Hailo pentru normalizare si optimizarea `global_avgpool_reduction`
- optimizeaza modelul pentru cuantizare
- compileaza fisierul `.hef`

Implicit, fluxul foloseste imaginile din:

```text
data/processed_4classes/images/train
```

Acest lucru este util pentru deployment pe dispozitive care folosesc acceleratoare Hailo.

## Medii virtuale folosite

Pentru rulare sunt utile doua medii separate:
- `torch_train` pentru preprocesare, antrenare, test si inferenta PyTorch
- `hailo_compile` pentru compilarea `ONNX -> HEF`

Exemple:

```bash
source /home/bogdan_avr/venvs/torch_train/bin/activate
cd /home/bogdan_avr/licenta2/proiect_efficientnetv2s
python -m train.prepare_dataset
python -m train.train
python -m train.test
python -m train.test_model --image /cale/catre/imagine.jpg
```

Pentru Hailo:

```bash
source /home/bogdan_avr/venvs/hailo_compile/bin/activate
cd /home/bogdan_avr/licenta2/proiect_efficientnetv2s
python -m export.compile_hailo --onnx export/emotion_model.onnx --calib-dir data/processed_4classes/images/train --hef-out export/emotion_model.hef --hw-arch hailo8l --division 2 2
```

## Ordinea recomandata de lucru

Ordinea recomandata pentru folosirea proiectului este:

1. rulezi `python -m train.prepare_dataset`
2. rulezi `python -m train.train`
3. rulezi `python -m train.test`
4. rulezi inferenta individuala cu `python -m train.test_model --image ...`
5. daca ai export ONNX, rulezi compilarea Hailo

## Observatii importante

- Daca modifici strategia de preprocesare, trebuie sa regenerezi datasetul procesat.
- Daca modifici dimensiunile din `config.py`, trebuie sa pastrezi consistenta intre `prepare_dataset.py`, `train.py`, `test.py` si `test_model.py`.
- Inferenta pe imagini individuale foloseste acelasi crop facial ca preprocesarea, tocmai pentru a evita discrepante intre train si deployment.
- Checkpoint-ul principal este ales dupa `macro F1`, nu doar dupa `accuracy`.

## Rezultatul final urmarit

La final, proiectul produce:
- un dataset facial preprocesat si organizat pe split-uri
- un model `EfficientNet-B0` antrenat pentru 4 emotii
- un set de metrici si vizualizari pentru evaluare
- posibilitatea de inferenta pe imagini individuale
- un flux de compilare catre `.hef` pentru Hailo
