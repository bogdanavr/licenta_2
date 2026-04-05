# Robot bazat pe Raspberry Pi pentru recunoașterea emoțiilor faciale și reacție multimodală
## README tehnic detaliat pentru `proiect_efficientnetb0`

## 1. Contextul proiectului

Acest repository implementează componenta de viziune și clasificare emoțională a unui sistem de tip **social robot** construit în jurul unui **Raspberry Pi 5**, a unei camere și a unui accelerator AI din ecosistemul **Raspberry Pi AI Kit / AI HAT (Hailo)**. Scopul practic este recunoașterea expresiilor faciale în timp real și declanșarea unei reacții multimodale simple, în implementarea actuală sub forma unui **feedback audio cu buzzer**.

Din punct de vedere software, proiectul este împărțit în două zone majore:

1. **zona de dezvoltare offline**, unde se pregătește datasetul, se antrenează modelul, se evaluează și se exportă;
2. **zona de deployment pe Raspberry Pi**, unde modelul rulează fie:
   - pe **CPU**, prin `onnxruntime`, sau
   - pe **NPU / AI HAT**, prin model compilat în format **HEF**.

Repository-ul nu este doar un folder cu scripturi separate, ci un flux coerent end-to-end:

- extragere și filtrare date dintr-un dataset brut;
- preprocesare focalizată pe față;
- organizare stratificată în `train / val / test`;
- antrenare cu transfer learning folosind `EfficientNet-B0`;
- evaluare cu metrici potrivite pentru clasificare multi-clasă;
- inferență pe imagini individuale;
- export `PyTorch -> ONNX`;
- compilare `ONNX -> HEF` pentru Hailo;
- integrare într-un pipeline live pe Raspberry Pi cu detecție de față, inferență și reacție audio.

---

## 2. Tema și rolul acestui repository în proiectul de licență

Tema generală este:

**Robot bazat pe Raspberry Pi pentru recunoașterea emoțiilor faciale și reacție multimodală**.

În această lucrare, repository-ul `proiect_efficientnetb0` acoperă în principal partea de:

- construire a clasificatorului de emoții;
- pregătire și standardizare a datelor;
- evaluare experimentală a modelului;
- pregătire a modelului pentru deployment embedded;
- implementare a rulării pe Raspberry Pi 5, atât pe CPU, cât și pe acceleratorul Hailo.

Prin urmare, acest folder poate fi privit ca **nucleul de machine learning și deployment embedded** al întregului proiect.

---

## 3. Obiectiv tehnic

Obiectivul tehnic al implementării este obținerea unui sistem suficient de robust și suficient de ușor pentru a putea rula pe dispozitive edge, fără a sacrifica foarte mult calitatea clasificării.

Mai concret, proiectul urmărește:

- clasificarea unei fețe în una dintre cele 4 emoții: `happy`, `neutral`, `sad`, `surprise`;
- menținerea consistenței dintre train și inferență prin același tip de preprocesare facială;
- separarea clară între pipeline-ul de antrenare și cel de deployment;
- posibilitatea de a compara rularea pe CPU vs AI accelerator;
- limitarea declanșărilor false printr-un mecanism de **stabilizare temporală** bazat pe istoric recent de predicții.

---

## 4. Structura repository-ului

```text
proiect_efficientnetb0/
├── data/
│   └── processed_4classes/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── meta.json
├── export/
│   ├── __init__.py
│   ├── compile_hailo.py
│   ├── emotion_model.hef
│   ├── emotion_model.metadata.json
│   └── export_onnx.py
├── logs/
├── pi/
│   ├── emotions_utils.py
│   ├── models/
│   │   ├── deploy.prototxt.txt
│   │   ├── emotion_model.hef
│   │   ├── hailort.log
│   │   └── res10_300x300_ssd_iter_140000.caffemodel
│   └── ssd_detect/
│       ├── detect_cpu_buzz.py
│       ├── detect_npu_buzz.py
│       ├── emotions_utils.py
│       ├── hailort.log
│       └── test_buzzer.py
├── train/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── model.py
│   ├── prepare_dataset.py
│   ├── preprocessing.py
│   ├── test.py
│   ├── test_model.py
│   └── train.py
├── README.md
└── requirements.txt
```

Observație importantă: în repository apar și artefacte deja generate, cum ar fi fișierele exportate sau unele fișiere utile direct pe Raspberry Pi. Cu alte cuvinte, proiectul conține atât **codul sursă**, cât și o parte din **rezultatele intermediare de deployment**.

---

## 5. Datasetul și filosofia de lucru cu datele

### 5.1. Sursa datelor

Scripturile din proiect pornesc de la un dataset local denumit în cod `AffectNet Augmented`, aflat în directorul `archive`. Configurația centrală definește explicit:

- `ARCHIVE_DIR`
- `COMBINED_LABELS_CSV`
- `PROCESSED_DIR`
- `IMAGES_DIR`
- `META_PATH`

în `train/config.py`.

### 5.2. De ce este important `combined_labels.csv`

O decizie foarte bună din proiect este faptul că **fișierul CSV este tratat ca sursă de adevăr** pentru etichete și pentru localizarea imaginilor, nu structura brută de directoare.

Asta înseamnă că pipeline-ul nu presupune că o imagine aparține clasei indicate de numele folderului în care se află. În schimb, scriptul:

- citește `filepath`, `label` și `source` din CSV;
- verifică dacă imaginea există fizic în arhivă;
- filtrează doar clasele țintă;
- ignoră extensiile nesuportate.

Această alegere este extrem de utilă în proiectele reale, deoarece dataseturile agregate sau augmentate manual pot conține inconsistențe în structura de directoare, iar folosirea unui manifest explicit reduce riscul de etichete greșite.

### 5.3. Clasele țintă

Modelul lucrează cu exact 4 clase:

- `happy`
- `neutral`
- `sad`
- `surprise`

Aceste clase sunt definite în `TARGET_LABELS`, iar mapările `CLASS_TO_IDX` și `IDX2LABEL` sunt construite automat în `train/config.py`.

### 5.4. Filtrarea pe sursă

Scriptul `train/prepare_dataset.py` permite argumentul:

```bash
--sources original augmented
```

Deci pot fi folosite:

- numai date originale,
- numai date augmentate,
- sau ambele.

Acest lucru este util experimental, pentru că permite compararea contribuției augmentării la performanța finală.

---

## 6. Preprocesarea imaginilor

### 6.1. Ideea generală

Pentru facial expression recognition, fundalul și elementele din afara feței pot introduce zgomot. În consecință, proiectul mută accentul pe **regiunea facială**, nu pe întreaga imagine.

Modulul `train/preprocessing.py` implementează exact această logică.

### 6.2. Detecția feței

În etapa offline de pregătire a datasetului, detecția facială este făcută cu:

- `haarcascade_frontalface_default.xml`
- prin `cv2.CascadeClassifier(...)`

și se lucrează pe imagine grayscale cu:

- conversie în grayscale;
- `equalizeHist`;
- `detectMultiScale(scaleFactor=1.1, minNeighbors=5, minSize=(48,48))`.

Dacă sunt detectate mai multe fețe, se alege **cea mai mare** (presupusă fața principală din imagine).

### 6.3. Extinderea bounding box-ului

După detectare, bounding box-ul este extins cu o marjă de aproximativ `20%` prin funcția `expand_bbox(...)`. Motivul este bun: expresia facială nu este codificată doar în zona strictă a ochilor și gurii, ci și în contextul imediat al feței, iar un crop prea strâns poate tăia informație relevantă.

### 6.4. Fallback-ul când nu este găsită fața

Dacă detectorul nu găsește nicio față, pipeline-ul nu abandonează imaginea. În loc de asta, aplică un:

- `center_square_crop`

Această decizie este pragmatică:

- evită pierderea totală a unor eșantioane;
- păstrează un comportament determinist;
- permite procesarea datasetului fără întrerupere.

### 6.5. Transformarea într-un pătrat fără deformare

După crop, imaginea este adusă la format pătrat cu `cv2.copyMakeBorder(..., borderType=cv2.BORDER_REPLICATE)`.

Asta este o alegere mai bună decât redimensionarea directă a unui dreptunghi la pătrat, pentru că:

- evită deformarea feței;
- păstrează proporțiile trăsăturilor faciale;
- produce intrări mai naturale pentru clasificator.

### 6.6. De ce `256x256` pentru salvare și `224x224` pentru model

Configurația definește:

- `SAVED_IMG_SIZE = 256`
- `IMG_SIZE = 224`

Strategia este foarte bine aleasă:

1. imaginea procesată se salvează offline la `256x256`;
2. în timpul antrenării sau evaluării, se aplică `RandomCrop(224)` sau `CenterCrop(224)`.

Avantaje:

- rămâne puțin context în jurul feței;
- există flexibilitate geometrică suplimentară la train;
- evaluarea rămâne standardizată;
- intrarea finală rămâne compatibilă cu `EfficientNet-B0`.

---

## 7. Pregătirea datasetului procesat

Scriptul-cheie este:

```bash
python -m train.prepare_dataset
```

### 7.1. Ce face concret

`train/prepare_dataset.py`:

- resetează directorul procesat dacă acesta există;
- citește CSV-ul centralizat;
- filtrează clasele țintă și sursele cerute;
- creează split-uri `train / val / test`;
- citește fiecare imagine cu OpenCV;
- aplică preprocesarea facială;
- salvează imaginea procesată în structura finală pe foldere;
- construiește un `meta.json` cu statistici utile.

### 7.2. Împărțirea în split-uri

În `config.py` sunt definite:

- `TRAIN_RATIO = 0.8`
- `VAL_RATIO = 0.1`
- `TEST_RATIO = 0.1`

Iar în `prepare_dataset.py` split-ul este **stratificat pe clasă**. Cu alte cuvinte, fiecare etichetă este împărțită separat și apoi elementele sunt amestecate în interiorul fiecărui split.

Aceasta este o alegere corectă, pentru că evită situațiile în care anumite clase ajung disproporționat într-un split.

### 7.3. Statistici generate

La final, `meta.json` conține inclusiv:

- câte imagini sunt în fiecare split;
- distribuția pe clase;
- distribuția pe surse;
- câte imagini au fost procesate cu față detectată;
- câte au intrat pe fallback cu center crop;
- dimensiunile de salvare și de input în model.

Acesta este un aspect foarte valoros pentru o lucrare de licență, deoarece face posibilă raportarea clară a metodologiei și a calității preprocesării.

---

## 8. Încărcarea datelor în PyTorch

`train/dataset.py` definește clasa `EmotionDataset`.

### 8.1. Cum funcționează

Datasetul nu mai lucrează cu CSV-ul brut după procesare. El citește imaginile direct din structura:

```text
data/processed_4classes/images/{split}/{label}/...
```

Avantajele sunt evidente:

- cod mai simplu la train/test;
- timp mai mic de încărcare logică;
- separare clară între pregătirea datelor și antrenare;
- mai puține locuri unde se pot introduce erori.

### 8.2. Validări utile

Clasa verifică:

- validitatea split-ului (`train`, `val`, `test`);
- existența directorului procesat;
- existența efectivă a imaginilor.

Dacă directorul lipseste, mesajul de eroare spune explicit să fie rulat mai întâi `python -m train.prepare_dataset`, ceea ce este foarte util pentru reproductibilitate.

---

## 9. Arhitectura modelului

### 9.1. Modelul de bază

`train/model.py` construiește modelul cu:

- `torchvision.models.efficientnet_b0`
- `EfficientNet_B0_Weights.DEFAULT` când `pretrained=True`

Deci este folosit **transfer learning** pornind de la greutăți pre-antrenate.

### 9.2. Capul de clasificare

Capul final este înlocuit cu:

```python
nn.Sequential(
    nn.Dropout(p=0.25, inplace=True),
    nn.Linear(in_features, num_classes),
)
```

Deci modelul final:

- păstrează backbone-ul EfficientNet-B0;
- înlocuiește clasificatorul final pentru cele 4 emoții;
- introduce un mic `Dropout` pentru regularizare.

### 9.3. De ce este o alegere bună `EfficientNet-B0`

Alegerea are sens tehnic și pentru licență, și pentru deployment embedded:

- este un model relativ compact;
- lucrează natural cu input `224x224`;
- are suport stabil în `torchvision`;
- este suficient de mic pentru iterații rapide;
- se potrivește mai bine decât modele mai mari într-un context edge.

În literatura recentă, arhitecturile **lightweight** sau compact-scaled rămân foarte atractive pentru recunoașterea emoțiilor atunci când se urmărește un compromis bun între cost și performanță, iar variante din familia EfficientNet apar frecvent în lucrări de FER și în sisteme video ușoare. În plus, în contexte competitive și benchmark-uri recente pentru afective computing continuă să fie raportate rezultate bune pentru modele ușoare pre-antrenate pe fețe sau pe seturi mari de imagini. Vezi secțiunea de bibliografie de la final.

---

## 10. Strategia de antrenare

Scriptul principal este:

```bash
python -m train.train
```

### 10.1. Hiperparametri definiți în cod

În `train/train.py` apar explicit:

- `BATCH_SIZE = 64`
- `EPOCHS = 30`
- `BACKBONE_LR = 1e-4`
- `CLASSIFIER_LR = 3e-4`
- `WEIGHT_DECAY = 1e-4`
- `NUM_WORKERS = 4`
- `PATIENCE = 7`
- `LABEL_SMOOTHING = 0.08`

Acești hiperparametri arată o strategie matură:

- învățare mai lentă pentru backbone;
- învățare mai rapidă pentru capul nou de clasificare;
- regularizare prin `weight decay`;
- oprire timpurie pentru a evita overfitting.

### 10.2. Reproductibilitate

Funcția `seed_everything(...)` setează seed pentru:

- `random`
- `numpy`
- `torch`
- `torch.cuda`

și activează `torch.backends.cudnn.benchmark = True` dacă există CUDA.

### 10.3. Augmentările de train

Transformările de train sunt moderate și bine alese pentru expresii faciale:

- `Resize(256, 256)`
- `RandomHorizontalFlip(p=0.5)`
- `RandomApply(ColorJitter(...), p=0.35)`
- `RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))`
- `RandomCrop(224)`
- `ToTensor()`
- `Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)`

Observația importantă este că augmentările sunt **moderate**, nu agresive. Asta este potrivit pentru FER, deoarece:

- expresiile faciale sunt sensibile la deformări;
- augmentări prea dure pot altera exact semnalul pe care vrem să îl clasificăm.

### 10.4. Transformările de evaluare

Pentru validare și test se folosesc transformări deterministe:

- `Resize(256, 256)`
- `CenterCrop(224)`
- `ToTensor()`
- `Normalize(...)`

Această separare train/eval este corectă și standard.

### 10.5. Tratarea dezechilibrului de clasă

Funcția `build_loss(...)` calculează distribuția claselor în train și verifică raportul de dezechilibru. Dacă `imbalance_ratio >= 1.25`, se construiesc ponderi inverse după frecvență și acestea sunt introduse în `CrossEntropyLoss`.

Deci ponderarea claselor nu este aplicată automat în orice condiție, ci doar când dezechilibrul este suficient de mare. Este o alegere echilibrată și evită introducerea inutilă a unei surse de instabilitate când setul este deja rezonabil distribuit.

### 10.6. Label smoothing

Se folosește:

- `CrossEntropyLoss(..., label_smoothing=0.08)`

Acest lucru este foarte potrivit pentru clasificare facială, unde etichetele pot conține ambiguitate sau zgomot. În recunoașterea expresiilor, granița dintre `neutral`, `sad` și uneori chiar `surprise` poate fi subtilă, mai ales în date augmentate sau etichetate semi-automat.

### 10.7. Optimizatorul

Scriptul folosește `AdamW`, cu grupuri de parametri separate:

- backbone cu decay;
- backbone fără decay;
- classifier cu decay;
- classifier fără decay.

Mai mult, backbone-ul și capul final au learning rate diferit. Aceasta este o decizie foarte bună în transfer learning, pentru că:

- backbone-ul pornește deja dintr-un spațiu de reprezentări utile;
- capul nou trebuie adaptat mai repede la cele 4 clase.

### 10.8. Scheduler

Se folosește:

- `CosineAnnealingLR(..., T_max=EPOCHS, eta_min=1e-6)`

Această alegere este adecvată pentru fine-tuning și oferă o scădere mai netedă a ratei de învățare.

### 10.9. Mixed precision

Dacă există GPU, scriptul folosește:

- `torch.autocast(...)`
- `torch.cuda.amp.GradScaler(...)`

Deci pipeline-ul este pregătit și pentru antrenare eficientă pe GPU, nu doar pe CPU.

### 10.10. Criteriul de selecție al checkpoint-ului

Checkpoint-ul cel mai bun este selectat după:

- `val_macro_f1`

nu după simpla `accuracy`.

Aceasta este una dintre cele mai bune decizii metodologice din proiect. Pentru un task cu 4 clase și posibilă distribuție inegală, `macro_f1` este mai informativ decât accuracy, deoarece tratează toate clasele mai echilibrat.

### 10.11. Early stopping

Dacă modelul nu se îmbunătățește timp de `PATIENCE = 7` epoci, antrenarea se oprește.

Asta ajută la:

- reducerea timpului de antrenare;
- evitarea overfitting-ului;
- păstrarea celui mai bun checkpoint.

---

## 11. Metrici urmărite

`train/metrics.py` calculează:

- `accuracy`
- `balanced_accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `weighted_f1`
- `top2_accuracy`
- `macro_auc_ovr` când probabilitățile permit calculul

Acest set este foarte bun pentru o lucrare aplicată, deoarece combină:

- metrici intuitive (`accuracy`),
- metrici robuste la dezechilibru (`balanced_accuracy`, `macro_f1`),
- metrici utile pentru analiză practică (`top2_accuracy`),
- metrică de separabilitate probabilistică (`macro_auc_ovr`).

`top2_accuracy` este interesantă în special pentru FER, deoarece uneori expresiile sunt apropiate semantic, iar a doua predicție poate fi încă relevantă pentru analiza calitativă.

---

## 12. Evaluarea finală

Scriptul:

```bash
python -m train.test
```

face evaluarea finală pe split-ul `test`.

### 12.1. Ce produce

- `logs/test_metrics.json`
- `logs/classification_report.json`
- `logs/confusion_matrix.png`

### 12.2. Ce este valoros aici

Acest pas este separat de validarea din timpul antrenării. Asta este corect metodologic:

- `val` este folosit pentru selecția modelului;
- `test` este folosit pentru raportarea finală.

Mai mult, matricea de confuzie normalizată este extrem de utilă pentru a arăta ce emoții se confundă cel mai des.

---

## 13. Testarea pe o imagine individuală

Scriptul:

```bash
python -m train.test_model --image /cale/catre/imagine.jpg
```

permite inferență punctuală.

### 13.1. Ce face

- încarcă modelul salvat;
- citește imaginea cu Pillow;
- reaplică preprocesarea facială prin `preprocess_face_rgb(...)`;
- aplică transformările de evaluare;
- calculează probabilitățile;
- afișează top-k predicții.

### 13.2. De ce este important

Acest script păstrează consistența cu pipeline-ul de train, deoarece nu face o inferență „brută” pe imagine, ci trece prin același tip de crop și standardizare. Este exact genul de detaliu care contează în deployment, pentru că micile nepotriviri între train și inference pot degrada performanța reală.

---

## 14. Exportul în ONNX

Scriptul este:

```bash
python -m export.export_onnx
```

### 14.1. Ce face

- încarcă checkpoint-ul `best_model.pth`;
- reconstruiește modelul cu `build_model(pretrained=False)`;
- exportă în `emotion_model.onnx`;
- opțional validează modelul ONNX cu `onnx.checker`;
- salvează metadate într-un fișier JSON.

### 14.2. Parametri importanți

Scriptul permite:

- alegerea `opset`-ului;
- batch fix sau batch dinamic;
- alegerea device-ului pentru export;
- dezactivarea validării post-export.

### 14.3. De ce ONNX este pasul corect

Pentru ecosistemul Hailo, modelul PyTorch nu este de regulă formatul final de deployment. Fluxul practic este:

```text
PyTorch checkpoint (.pth) -> ONNX -> HEF
```

Acesta este exact motivul pentru care proiectul include un pas explicit de export. În ecosistemul Raspberry Pi AI Kit, posibilitatea de a compila propriile modele pentru Hailo a fost extinsă prin Dataflow Compiler, iar formatul executabil HEF este artefactul folosit la rularea pe accelerator.

---

## 15. Compilarea pentru Hailo

Scriptul este:

```bash
python -m export.compile_hailo \
  --onnx export/emotion_model.onnx \
  --calib-dir data/processed_4classes/images/train \
  --hef-out export/emotion_model.hef \
  --hw-arch hailo8
```

### 15.1. Ce face concret

`export/compile_hailo.py`:

- încarcă modelul ONNX;
- pregătește setul de imagini pentru calibrare;
- construiește un `model script` pentru Hailo;
- aplică normalizarea în scriptul Hailo;
- încearcă optimizarea `global_avgpool_reduction` pentru stratul `avgpool`;
- rulează cuantizarea/optimizarea;
- compilează modelul final în format `.hef`.

### 15.2. Calibrarea

Pentru cuantizare, scriptul selectează imagini din:

```text
data/processed_4classes/images/train
```

Imaginile sunt:

- convertite la RGB;
- redimensionate;
- centrate la dimensiunea de intrare;
- transformate în `uint8`.

Scriptul poate limita numărul de imagini de calibrare, dar recomandă `>= 1024`, ceea ce este rezonabil pentru cuantizare mai stabilă.

### 15.3. Normalizarea în toolchain-ul Hailo

Un aspect foarte bun este că normalizarea nu este „uitată” la compilare. Scriptul convertește `IMAGE_MEAN` și `IMAGE_STD` în scala 0..255 și le introduce explicit în model script-ul Hailo.

Asta este important pentru consistența dintre:

- modelul antrenat în PyTorch;
- modelul exportat;
- modelul rulat pe accelerator.

### 15.4. Ajustarea pentru avgpool

Scriptul încearcă să detecteze automat stratul final de `GlobalAveragePool` și poate aplica o optimizare de tip `global_avgpool_reduction`. Acesta este un detaliu foarte util practic, pentru că modelele exportate în ONNX nu se mapează întotdeauna ideal direct pe toolchain-ul hardware.

### 15.5. De ce acest pas este foarte important pentru licență

Această secțiune ridică proiectul de la nivelul unui simplu experiment ML la un proiect de **deployment embedded real**. În mod concret, nu este suficient să existe un `.pth` bun; modelul trebuie:

- exportat într-un format interoperabil;
- calibrat;
- cuantizat;
- compilat pentru acceleratorul țintă.

Asta susține foarte bine contribuția originală privind comparația CPU vs AI HAT.

---

## 16. Artefactele principale generate

### 16.1. `data/processed_4classes/meta.json`

Conține:

- sursa datasetului;
- clasele țintă;
- split ratio;
- setările de preprocesare;
- totaluri pe split;
- distribuții pe clase;
- câte imagini au avut fața detectată;
- câte au folosit fallback.

### 16.2. `export/best_model.pth`

Checkpoint-ul salvează:

- `model_state_dict`;
- numele modelului;
- numele de afișare;
- epoca cea mai bună;
- metrica după care a fost selectat;
- valoarea celei mai bune metrici;
- dimensiunile de intrare;
- etichetele.

### 16.3. `export/emotion_model.onnx`

Versiunea interoperabilă a modelului, potrivită pentru deployment sau compilare ulterioară.

### 16.4. `export/emotion_model.hef`

Modelul executabil pentru Hailo.

### 16.5. `logs/`

În mod normal include:

- `train_log.csv`
- `train_history.json`
- `test_metrics.json`
- `classification_report.json`
- `confusion_matrix.png`

Aceste fișiere sunt foarte utile pentru partea experimentală a licenței.

---

## 17. Componenta de Raspberry Pi: rulare live în timp real

În folderul `pi/ssd_detect/` se află componenta live pentru Raspberry Pi.

Aici pipeline-ul este diferit de cel offline de pregătire a datasetului: pentru rulare live nu se mai folosește detectorul Haar, ci un detector **SSD bazat pe OpenCV DNN**, cu modelul clasic `res10_300x300_ssd_iter_140000.caffemodel` și fișierul `deploy.prototxt.txt`.

Aceasta este o decizie foarte bună practic, fiindcă în streaming video un detector DNN de față este în general mai robust decât Haar cascade-ul folosit offline pentru pregătirea rapidă a datasetului.

### 17.1. Scriptul CPU

`pi/ssd_detect/detect_cpu_buzz.py`:

- citește cadre din `Picamera2`;
- detectează fețe cu SSD Caffe prin OpenCV DNN;
- decupează regiunea facială;
- normalizează imaginea în stilul folosit la antrenare (`mean/std` ImageNet);
- rulează inferența cu `onnxruntime` pe `CPUExecutionProvider`;
- face smoothing temporal al etichetelor;
- declanșează buzzerul când emoția stabilă se schimbă;
- afișează pe ecran bounding box, FPS și latență.

### 17.2. Scriptul NPU

`pi/ssd_detect/detect_npu_buzz.py`:

- citește cadre din cameră;
- detectează fețe tot cu SSD OpenCV;
- pregătește crop-ul facial ca `uint8`, `RGB`, `NHWC`;
- încarcă modelul `.hef` prin `hailo_platform`;
- rulează inferența pe accelerator prin `InferVStreams`;
- face smoothing temporal;
- comandă buzzerul la schimbarea emoției stabile;
- afișează FPS și latență.

### 17.3. De ce este foarte bună separarea CPU / NPU

Această separare permite benchmark comparativ real între:

- inferența clasică pe CPU;
- inferența accelerată cu Hailo.

Pentru partea experimentală a licenței, acesta este un punct foarte puternic, deoarece face posibilă raportarea:

- latenței pe cadru;
- FPS-ului;
- eventual consumului energetic;
- diferențelor de preprocesare și format de intrare.

---

## 18. Detecția de față live cu SSD

În scripturile live sunt definiți câțiva parametri practici:

- `CONF_THR = 0.6`
- `SCALE = 0.75`
- `PADDING = 0.15`

### 18.1. Rolul lor

- `CONF_THR` controlează filtrarea detecțiilor slabe;
- `SCALE` reduce rezoluția de procesare pentru detecție, deci poate crește viteza;
- `PADDING` lărgește bounding box-ul facial pentru a nu tăia prea agresiv expresia.

### 18.2. De ce acest design are sens

Într-un sistem embedded real, detecția feței trebuie să fie:

- suficient de robustă;
- suficient de rapidă;
- configurabilă ușor.

Aceste trei constante fac posibil tocmai acest compromis.

---

## 19. Stabilizarea temporală a predicțiilor

Atât în `pi/emotions_utils.py`, cât și în `pi/ssd_detect/emotions_utils.py`, apare clasa `EmotionSystem`.

### 19.1. Ce face

Clasa păstrează ultimele `window_size` predicții într-un `deque` și returnează emoția dominantă prin `Counter(...).most_common(1)`.

Mai calculează și FPS-ul pe baza numărului de cadre procesate într-o fereastră scurtă.

### 19.2. De ce este o idee foarte bună

În practică, predicțiile frame-by-frame sunt zgomotoase. O singură clipire, o mișcare de cap sau un blur scurt poate produce o etichetă greșită. Dacă sistemul ar reacționa instant la fiecare cadru, s-ar obține multe declanșări false.

Prin folosirea unei ferestre temporale și a emoției dominante:

- reacția devine mai stabilă;
- scad fluctuațiile vizuale și sonore;
- crește robustețea percepută a robotului.

În literatura de video emotion recognition și în sisteme afective multimodale, agregarea temporală sau smoothing-ul pe secvențe este o idee recurentă tocmai pentru că emoția observată în video are natură temporală, nu strict instantanee.

---

## 20. Reacția multimodală: buzzer

`pi/ssd_detect/test_buzzer.py` implementează răspunsul audio simplu folosind:

- `gpiozero.TonalBuzzer`
- pinul `GPIO 17`

Fiecare emoție are o secvență proprie de note și durate:

- `HAPPY`
- `NEUTRAL`
- `SAD`
- `SURPRISE`

Această parte este importantă deoarece transformă clasificarea într-un comportament observabil. Cu alte cuvinte, sistemul nu doar estimează o emoție, ci **acționează** în funcție de aceasta.

Pentru un proiect de licență cu temă de social robot, această trecere de la percepție la reacție este esențială.

---

## 21. Diferențe între pipeline-ul offline și pipeline-ul live

Este util de subliniat că proiectul folosește două strategii de detecție facială, fiecare cu motivul ei:

### Offline, la pregătirea datasetului

- Haar cascade (`haarcascade_frontalface_default`)
- simplu, ușor, suficient pentru procesare în lot

### Online, în rulare live pe Raspberry Pi

- SSD face detector OpenCV DNN (`res10_300x300_ssd_iter_140000`)
- mai robust pentru video și condiții reale

Aceasta nu este o inconsistență, ci o alegere practică:

- pentru dataset processing contează să fie ușor și rapid de rulat pe mii de imagini;
- pentru live demo contează mai mult robustețea detecției în flux video.

---

## 22. Dependențe

În `requirements.txt` apar:

- `torch`
- `torchvision`
- `numpy`
- `pillow`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `opencv-python`

Pe lângă acestea, pentru anumite părți din proiect mai sunt necesare în practică:

- `onnx`
- `onnxruntime`
- `picamera2`
- `gpiozero`
- pachetele Hailo (`hailo_sdk_client`, `hailo_platform`) în mediile dedicate

Deci, realist, proiectul cere **cel puțin două medii separate**:

1. mediu pentru train/eval/export ONNX;
2. mediu pentru toolchain-ul Hailo și/sau pentru rularea pe Raspberry Pi.

---

## 23. Ordinea recomandată de lucru

### 23.1. Pentru dezvoltare ML

```bash
python -m train.prepare_dataset
python -m train.train
python -m train.test
python -m train.test_model --image /cale/catre/imagine.jpg
```

### 23.2. Pentru export și deployment

```bash
python -m export.export_onnx
python -m export.compile_hailo --onnx export/emotion_model.onnx --calib-dir data/processed_4classes/images/train --hef-out export/emotion_model.hef --hw-arch hailo8
```

### 23.3. Pentru rulare live pe Raspberry Pi

CPU:

```bash
python detect_cpu_buzz.py
```

NPU:

```bash
python detect_npu_buzz.py
```

---

## 24. Puncte forte ale implementării

### 24.1. Pipeline complet

Proiectul acoperă tot lanțul:

- date -> preprocesare -> train -> test -> export -> compilare -> deployment live.

### 24.2. Consistență bună între train și inferență

Normalizarea, dimensiunile și logica de crop sunt gândite coerent.

### 24.3. Alegere potrivită de metrici

Nu se bazează doar pe accuracy, ci și pe macro F1, balanced accuracy și altele.

### 24.4. Compatibilitate edge

Alegerea `EfficientNet-B0` și fluxul ONNX/HEF susțin clar obiectivul embedded.

### 24.5. Gândire experimentală sănătoasă

Există infrastructură pentru:

- evaluare separată pe test;
- matrice de confuzie;
- statistici despre date;
- comparație CPU vs NPU.

### 24.6. Mecanism de stabilizare temporală

Foarte important pentru reducerea reacțiilor false într-un sistem live.

---

## 25. Limitări actuale

Deși proiectul este bine construit, există și câteva limitări naturale:

1. **Numărul de clase este redus la 4**, ceea ce simplifică problema, dar nu acoperă întregul spectru emoțional.
2. **Detecția facială offline folosește Haar cascade**, care nu este cel mai robust detector în condiții dificile.
3. **Stabilizarea temporală este bazată pe majoritate simplă**, nu pe un model temporal mai sofisticat.
4. **Reacția multimodală este momentan simplă**, bazată pe buzzer, nu pe TTS, LED-uri sau comportament robotic mai complex.
5. **Benchmark-ul energetic** nu este implementat explicit în codul vizibil din acest folder, deși proiectul este pregătit conceptual pentru astfel de comparații.

Aceste limitări nu diminuează valoarea implementării; din contră, oferă direcții clare de extindere pentru partea de discuții și viitor work.

---

## 26. Posibile direcții de îmbunătățire

1. Înlocuirea detectorului Haar offline cu un detector mai robust și mai modern.
2. Introducerea unui smoothing temporal mai avansat:
   - medie exponențială pe probabilități;
   - prag de persistență;
   - HMM / TCN / voting ponderat.
3. Extinderea la mai multe emoții.
4. Logare automată a latenței și FPS-ului în fișiere CSV în rularea live.
5. Integrare cu:
   - sintetizator vocal,
   - LED-uri,
   - servo-uri,
   - reacții comportamentale diferențiate.
6. Evaluare sistematică în condiții controlate:
   - lumină;
   - distanță;
   - unghi;
   - ochelari;
   - ocluzii parțiale.

---

## 27. De ce alegerile din proiect au sens și din perspectiva literaturii

Fără a transforma README-ul într-un review de literatură, merită menționate câteva justificări solide:

- **Modelele compacte** sunt foarte potrivite pentru recunoaștere de expresii în sisteme embedded și video, unde latența contează, nu doar acuratețea.
- **Pre-antrenarea și fine-tuning-ul** rămân strategii standard și eficiente atunci când setul final este mai îngust decât problema generală de clasificare de imagini.
- **Preprocesarea centrată pe față** este firească în FER, deoarece reduce zgomotul de fundal și concentrează informația relevantă.
- **Agregarea temporală** sau smoothing-ul pe secvențe este o idee bine justificată pentru scenarii video, unde predicțiile instantanee pot fluctua.
- **Fluxul ONNX -> HEF** este compatibil cu practica actuală din ecosistemul Hailo / Raspberry Pi AI, unde modelul trebuie compilat către format executabil pentru accelerator.

Pe scurt, alegerile din acest repository nu sunt arbitrare; ele sunt aliniate atât cu nevoile practice ale deployment-ului edge, cât și cu tendințe uzuale din literatura FER modernă.

---

## 28. Concluzie

`proiect_efficientnetb0` este un repository bine structurat pentru o lucrare de licență aplicată, deoarece nu se oprește la antrenarea unui model, ci acoperă întregul traseu necesar unui sistem embedded real:

- pregătirea datelor;
- modelare și antrenare;
- evaluare corectă;
- export și compilare pentru accelerator;
- integrare live pe Raspberry Pi;
- reacție multimodală;
- mecanism de stabilizare temporală.

Cea mai importantă calitate a proiectului este faptul că leagă coerent **partea academică** de **partea practică de deployment**. Din această cauză, el oferă o bază foarte bună pentru redactarea descrierii ample a proiectului, a contribuțiilor originale și a capitolului experimental din lucrarea de licență.

---

## 29. Bibliografie orientativă și surse utile pentru justificarea alegerilor

### Despre EfficientNet și modele ușoare pentru FER

1. Mingxing Tan, Quoc V. Le, **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**, ICML 2019.
2. Andrey V. Savchenko, **Facial expression and attributes recognition based on multi-task learning of lightweight neural networks**, arXiv:2103.17107.
3. Andrey V. Savchenko et al., **HSEmotion Team at the 6th ABAW Competition**, arXiv:2403.11590.
4. **Facial Emotion Recognition on FER-2013 using an EfficientNet-based compact detector**, arXiv:2601.18228.

### Despre smoothing / consistență temporală în video emotion recognition

5. Valentin Vielzeuf et al., **Temporal Multimodal Fusion for Video Emotion Classification in the Wild**, ACM ICMI 2017.

### Despre ecosistemul Raspberry Pi AI / Hailo

6. Raspberry Pi, **Raspberry Pi AI Kit available now at $70**.
7. Raspberry Pi, **Raspberry Pi AI Kit update: Dataflow Compiler now available**.
8. Hailo, **Open Model Zoos: Advancing TensorFlow & ONNX AI**.
9. Hailo Community, discuții despre formatul **HEF** și fluxul de compilare `ONNX -> HEF`.

### Despre detecția facială cu OpenCV DNN

10. OpenCV Documentation, materiale despre DNN-based face detection / face analysis pipeline.

---

## 30. Rezumat foarte scurt pentru folosire ulterioară în descrierea proiectului

Acest proiect implementează un sistem complet de clasificare a emoțiilor faciale pentru 4 clase (`happy`, `neutral`, `sad`, `surprise`) folosind `EfficientNet-B0`, cu un pipeline care pornește dintr-un dataset brut cu etichete centralizate în CSV, aplică preprocesare facială, realizează antrenare și evaluare în PyTorch, exportă modelul în ONNX și îl compilează în HEF pentru rulare accelerată pe Hailo. Pe Raspberry Pi 5, sistemul rulează live împreună cu o cameră și un detector SSD de fețe, atât pe CPU, cât și pe NPU, iar decizia emoțională este stabilizată temporal și transformată într-o reacție audio prin buzzer.
