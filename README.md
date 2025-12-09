# Federated Learning: Aggregation Strategies, Heterogeneity & Robustness

Questo repository contiene il codice sorgente e il framework di simulazione sviluppato per la Tesi di Laurea Magistrale in Informatica: **"Tecniche di aggregazione nel Federated Learning: Eterogeneità e Robustezza"**.

[cite_start]Il progetto è basato su **[Flower](https://flower.ai/)** (Flwr) e **PyTorch**, ed è progettato per analizzare l'impatto dell'eterogeneità dei dati (Non-IID) e la resilienza contro attacchi avversari (Adversarial Attacks) in scenari Federated Learning Cross-Silo[cite: 861, 887, 1279].

## 🌟 Caratteristiche Principali

**Simulazione Scalabile**: Orchestrazione automatizzata di centinaia di run sperimentali tramite `run_experiments.py`.
**Gestione Eterogeneità (Non-IID)**: Partizionamento dei dati basato su Distribuzione di Dirichlet ($\alpha$) per simulare sbilanciamento nelle label e nella quantità di dati.
**Strategie di Aggregazione Avanzate**: Implementazione custom di diverse strategie che estendono le classi base di Flower con logging avanzato e metriche:
    * *Baselines*: FedAvg, FedProx, FedAvgM (Momentum).
    * *Ottimizzatori Adattivi*: FedAdam, FedYogi[cite: 1152].
    * *Strategie Robuste*: FedMedian [cite: 1199][cite_start], FedTrimmedAvg [cite: 1213][cite_start], MultiKrum[cite: 1230].
* **Attacchi Avversari (Adversarial Attacks)**: Implementazione lato client di attacchi per testare la robustezza:
    * *Model Poisoning*: Gaussian Updates (Attacco Bizantino).
    * *Data Poisoning*: Label Flipping (mirato o random).
    * *Backdoor Attack*: Iniezione di trigger pattern nelle immagini.
* **Personalizzazione (FedPer)**: Implementazione dell'approccio *Federated Learning with Personalization Layers* per gestire l'eterogeneità[cite: 1184, 1530].
* **Metriche Rigorose**: Aggregazione statisticamente corretta delle metriche (es. somma delle matrici di confusione per il calcolo dell'F1-Score globale) e salvataggio automatico in JSON[cite: 1311].

## 📂 Struttura del Progetto

* `run_experiments.py`: Script principale. Legge le configurazioni, gestisce i seed per la riproducibilità e lancia le simulazioni Flower.
* `run_configuration.py`: Definisce i dizionari di configurazione per i diversi scenari (IID, Non-IID, Attacchi, ecc.).
* `server_app.py`: Logica del Server Flower. Gestisce l'inizializzazione della strategia, la configurazione globale e la valutazione centralizzata.
* `client_app.py`: Logica del Client Flower. Gestisce il training locale, l'applicazione degli attacchi (se il client è malevolo) e la personalizzazione del modello.
* `strategy.py`: Implementazione delle classi `Custom...` per le strategie di aggregazione (ereditano da `StrategyMixin`).
* `strategy_mixin.py`: Mixin che aggiunge funzionalità di logging JSON e aggregazione custom delle metriche alle strategie standard.
* `task.py`: Definizione del modello (CNN), funzioni di training/test e caricamento dati (Dataset e Partitioners).
* `analysis.py`: Script di post-processing che legge i risultati JSON generati e produce grafici comparativi in PDF.

## 🚀 Installazione

Il progetto richiede Python (>= 3.8). Le dipendenze sono gestite nel file `pyproject.toml`.

1.  **Clona il repository:**
    ```bash
    git clone [https://github.com/tuo-username/fl-tesi.git](https://github.com/tuo-username/fl-tesi.git)
    cd fl-tesi
    ```

2.  **Installa le dipendenze:**
    Puoi installare il pacchetto in modalità editabile o installare direttamente le librerie richieste:
    ```bash
    pip install -e .
    ```
    *Oppure manualmente:*
    ```bash
    pip install "flwr[simulation]>=1.22.0" "flwr-datasets[vision]>=0.5.0" torch torchvision scikit-learn numpy pandas matplotlib
    ```

## 💻 Utilizzo

### 1. Esecuzione degli Esperimenti
Il file `run_experiments.py` è l'entry point. Esegue sequenzialmente tutti gli scenari definiti in `run_configuration.py`.

```bash
python run_experiments.py
```
Nota: Lo script eseguirà un numero configurabile di ripetizioni per ogni scenario cambiando il seed per garantire la significatività statistica. I log e i risultati verranno salvati nella cartella outputs/YYYY-MM-DD_HH-MM-SS

### 2. Configurazione scenari
Per modificare o aggiungere esperimenti, modifica la lista scenarios in `run_configuration.py`.

Esempio di configurazione per un attacco Backdoor in scenario Non-IID:
```json
 {
    "scenario_name": "Backdoor_Attack_NonIID",
    "strategy-name": "multikrum",     # Strategia di difesa
    "partitioner-name": "dirichlet",  # Tipo di partizionamento
    "dirichlet-alpha": 0.1,           # Grado di eterogeneità
    "malicious-clients-ids": [3, 5],  # ID dei client attaccanti
    "attack_name": "backdoor",        # Tipo di attacco
    "attack_target_class": 1,         # Classe selezionata dall'attaccante
    "num-malicious-nodes": 2,         # Parametro per Krum
}
```

### 3. Analisi dei Risultati

Dopo aver completato gli esperimenti, utilizza lo script di analisi per generare i plot:

```bash
python analysis.py
```

Lo script cercherà i file `results.json` nella cartella `outputs/` e genererà i grafici nella cartella `plots/` (sia standard che client-only).

---

## 🛡️ Dettagli sugli Attacchi Implementati

La logica degli attacchi è incapsulata in `client_app.py`:

* **Label Flipping**: Inverte le etichette di un subset di dati di training (es. tutte le label 'X' diventano 'Y') prima del training locale.
* **Backdoor**: Inserisce un **trigger** (pattern di pixel) nelle immagini e forza la label target. L'efficacia è misurata tramite l'**ASR** (Attack Success Rate) durante la validazione.
* **Byzantine (Gaussian)**: Il client salta il training e restituisce al server i pesi del modello globale corrotti con rumore gaussiano.

## 📊 Dataset Supportati

Il progetto è configurato per utilizzare principalmente:

* **Fashion MNIST**: Per task di Computer Vision con CNN.
* **Adult Census Income**: (Supportato via configurazione XGBoost separata) per dati tabulari.

---

## 📝 Autore

* **Samuele Lolli**
* Alma Mater Studiorum - Università di Bologna
* Anno Accademico 2024/2025
