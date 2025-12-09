# Tecniche di aggregazione nel Federated Learning: eterogeneità e robustezza

Questo repository contiene il codice sorgente e il framework di simulazione sviluppato per la Tesi di Laurea Magistrale in Informatica: **"Tecniche di aggregazione nel Federated Learning: Eterogeneità e Robustezza"**.

Il progetto è basato su **[Flower](https://flower.ai/)** e **PyTorch**, ed è progettato per analizzare l'impatto dell'eterogeneità dei dati (Non-IID) e la resilienza contro attacchi avversari in scenari Federated Learning cross-silo.

## Caratteristiche Principali

* **Simulazione scalabile**: Orchestrazione automatizzata di centinaia di run sperimentali tramite `run_experiments.py`. Con la giusta potenza computazionale è possibile supportare le simulazioni su un numero molto alto di nodi.

* **Gestione eterogeneità (Non-IID)**: Partizionamento dei dati basato su Distribuzione di Dirichlet ($\alpha$) per simulare sbilanciamento nelle label e nella quantità di dati.

* **Strategie di aggregazione**: Analisi sperimentale di di diverse strategie che estendono le classi base di Flower con logging avanzato e metriche:
   
   *Baselines*: FedAvg, FedProx, FedAvgM (Momentum).
   
   *Ottimizzatori Adattivi*: FedAdam, FedYogi.
   
   *Strategie Robuste*: FedMedian, FedTrimmedAvg, MultiKrum.
   
* **Attacchi avversari**: Implementazione lato client di attacchi per testare la robustezza:
    * *Model Poisoning*: Gaussian Updates.
    * *Data Poisoning*: Label Flipping.
    * *Backdoor Attack*: Iniezione di trigger volto a introdurre una vulnerabilità nel modello.
    
* **Personalizzazione (FedPer)**: Implementazione dell'approccio *FedPer* per gestire l'eterogeneità.

* **Metriche**: Aggregazione statisticamente corretta delle metriche (es. somma delle matrici di confusione per il calcolo dell'F1-Score globale) e salvataggio automatico in JSON.

## Struttura del Progetto

* `run_experiments.py`: Script principale. Legge le configurazioni, gestisce i seed per la riproducibilità e lancia le simulazioni Flower.
* `run_configuration.py`: Definisce i dizionari di configurazione per i diversi scenari (IID, Non-IID, Attacchi, ecc.).
* `server_app.py`: Logica del Server Flower. Gestisce l'inizializzazione della strategia, la configurazione globale e la valutazione centralizzata.
* `client_app.py`: Logica del Client Flower. Gestisce il training locale, l'applicazione degli attacchi (se il client è malevolo) e la personalizzazione del modello.
* `strategy.py`: Implementazione delle classi `Custom...` per le strategie di aggregazione (ereditano da `StrategyMixin`).
* `strategy_mixin.py`: Mixin che aggiunge funzionalità di logging JSON e aggregazione custom delle metriche alle strategie standard.
* `task.py`: Definizione del modello (CNN), funzioni di training/test e caricamento dati (Dataset e Partitioners).
* `analysis.py`: Script di post-processing che legge i risultati JSON generati e produce grafici comparativi in PDF.

## Installazione

Il progetto richiede Python (>= 3.8). Le dipendenze sono gestite nel file `pyproject.toml`.

1.  **Clona il repository:**
    ```bash
    git clone https://github.com/samuele-lolli/Aggregation-Federated-Learning-Analysis
    cd Aggregation-Federated-Learning-Analysis
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

## Utilizzo

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
Lo script `analysis.py` automatizza l'analisi post-simulazione e la generazione di grafici.

1.  **Scansione Dati:** Lo script cerca tutti i file **`results.json`** nella cartella `outputs/`.
2.  **Filtraggio:** Utilizza la lista interna **`PLOTS_DEFINITIONS`** per selezionare e raggruppare i risultati degli esperimenti da confrontare (es. "FedAvg vs FedProx").
3.  **Generazione Grafici:**
    * Aggrega i dati su **15 ripetizioni** (run).
    * Calcola e rappresenta la **deviazione standard** come area ombreggiata (shadow area) per indicare la variabilità.
    * Produce i grafici in formato **`.pdf`**. 

4.  **Output:** I grafici vengono salvati in:
    * **`plots/standard/`** (metriche Server-Side + Client-Side)
    * **`plots/client_only/`** (solo metriche lato client)

## Dettagli sugli Attacchi Implementati

La logica degli attacchi è incapsulata in `client_app.py`:

* **Label Flipping**: Inverte le etichette di un subset di dati di training (es. tutte le label 'X' diventano 'Y') prima del training locale.
* **Backdoor**: Inserisce un **trigger** (pattern di pixel) nelle immagini e forza la label target. L'efficacia è misurata tramite l'**ASR** (Attack Success Rate) durante la validazione.
* **Byzantine (Gaussian)**: Il client salta il training e restituisce al server i pesi del modello globale corrotti con rumore gaussiano.

## Dataset Supportati

Il progetto è configurato per utilizzare principalmente:

* **Fashion MNIST**
* **MNIST**

Tuttavia può essere facilmente esteso per utilizzare altri dataset e modelli. 

## Esperimenti con XGBoost
Su [questo repository](https://github.com/samuele-lolli/XGBoost-Federated-Learning-Analysis) è possibile trovare il codice basato su Flower e sulla stessa logica utilizzata in questo progetto per il modello XGBoost. 

## Autori
* Lolli Samuele
* Prof. Ferretti Stefano
