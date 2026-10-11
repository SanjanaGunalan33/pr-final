# RCAEval Bayesian Network — Root Cause Assistant

A beginner-friendly Python/Streamlit project for probabilistic root-cause analysis using RCAEval incident metadata and metric features. The main model is a discrete Bayesian Network; posterior inference is performed with `pgmpy` Variable Elimination.

## What the app does

- Provides a **single-page** incident workflow.
- Lets users select one or more symptom categories.
- Allows an optional application selection; leave it unknown for a network-wide prediction.
- For each symptom, asks which service appears most affected, with options for unknown/no clear anomaly/another service.
- Returns the top five root-cause candidates with posterior probabilities and a CSV download.
- Shows the actual fitted graph and model metadata in an expandable section.

**Important scope note:** this is a research prototype learned from RCAEval, not a universal diagnosis engine. The model's root-cause candidates and service vocabulary come from the training dataset. An unknown application is treated as unobserved context; it does not magically generalize to arbitrary unseen service names or fault classes. Accuracy measured with known application context must not be represented as symptom-only accuracy.

## Dataset

Put these two original files in `data/`:

- `rcaeval_cases.csv`
- `rcaeval_features.csv`

The loader checks required columns, duplicate IDs, label consistency, and join integrity. Metadata incidents without metric records are preserved in the inventory but cannot contribute measured symptom features; missing metrics are not fabricated. The training script writes the verified inventory and split details to `outputs/`.

## Windows / VS Code setup

Use Python 3.11 or 3.12. From the project folder in a VS Code terminal:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
pytest -q
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Alternatively, double-click `run_website.bat` after installing dependencies and training the model. Streamlit prints the local URL, usually `http://localhost:8501`.

## Main files

- `app.py` — single-page Streamlit interface.
- `preprocessing.py` — data validation, feature extraction, and train-fitted discretization.
- `rca_model.py` — Bayesian Network, BDeu parameter estimation, Variable Elimination, CPT export, and graph drawing.
- `train_model.py` — grouped split, training-only representation selection, model fitting, artifact export, and evaluation.
- `evaluate_model.py` — ranking metrics and benchmark evaluation.
- `tests/test_model.py` — model/data pipeline tests.
- `models/rca_bn.joblib` — saved fitted model used by the app.
- `outputs/bayesian_network.png` — graph rendered from the fitted model object.
- `outputs/network_edges.csv` / `network_structure.csv` — actual graph structure.
- `outputs/conditional_probability_tables.csv` — learned CPTs in long form.
- `outputs/run_summary.json`, `evaluation_metrics.csv`, `split_info.json` — evaluation and split records.

## Model and evaluation

The Bayesian Network is trained from the training partition using Bayesian estimation with a BDeu prior (equivalent sample size 10). Continuous metric changes are converted to discrete states using thresholds fitted on training data. The split groups repeated runs by application, root-cause service, and fault type to reduce scenario leakage. Cross-validation for representation selection is performed within training data.

The project may include an Extra Trees benchmark. Its scores are **not** Bayesian Network scores. Report Top-1, Top-3, balanced accuracy, and Macro-F1 with the model name and split protocol. A 70% target is an objective, not a promised outcome.

## Rebuild artifacts

```powershell
python train_model.py
```

Training overwrites fitted artifacts and evaluation outputs. The Streamlit app loads the saved model and does not train on each page refresh.

## Limitations

- Predictions are limited to root-cause classes and symptom states represented by RCAEval.
- Service localization is part of the current model's symptom representation; a user must indicate the service that appears most affected for a selected symptom when known.
- Results for unknown application context require separate evaluation before being described as general symptom-only accuracy.
- Dataset size and class imbalance limit generalization; posterior values are model estimates, not guarantees.


## Final full-variable build (required for the agreed node design)

The checked-in trained model is a starter artifact. To rebuild the model to the agreed full-variable design, run:

```bash
python train_model.py --full-individual
python -m streamlit run app.py
```

`--full-individual` disables the top-K metric limit and retains every usable standardized metric column that has enough non-missing values and at least three distinct values as its own Low/Medium/High/Unknown node. It also creates a binary indicator node for each root-cause class, while retaining the mutually exclusive `RootCause` class node as the tractable inference target. `FaultType` is represented as an intermediate descriptor node. The exact final node count is calculated and saved from the fitted graph in `outputs/run_summary.json` and `outputs/network_structure.csv`; it is data-dependent and should not be guessed in advance.

The full-variable training run is heavier than the starter model. Use Python 3.11 or 3.12 and install `requirements.txt` first. The website intentionally has no Incident Context/application selector and no search icon.
