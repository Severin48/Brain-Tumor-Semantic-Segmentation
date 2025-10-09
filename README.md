# Brain-Tumor MRI Semantic Segmentation

An AI Lab project at HKA for semantic segmentation of brain tumors in MRI scans, based on the [LGG MRI Segmentation dataset on Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

This project uses modern Python tooling, including `uv` for high-speed environment and package management.

---

## Setup & Installation

Follow these steps to set up the project environment on your local machine.

#### Prerequisites
-   **Git:** To clone the repository.
-   **uv:** This project uses `uv` for environment management. If you don't have it, install it by following the [official `uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/).

#### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Severin48/Brain-Tumor-Semantic-Segmentation.git
    cd Brain-Tumor-Semantic-Segmentation
    ```

2.  **Create a virtual environment (will be inside .venv by default):**
    ```bash
    uv venv
    ```

3.  **Activate the environment:**
    * On **Linux or macOS** (bash/zsh):
        ```bash
        source .venv/bin/activate
        ```
    * On **Windows** (PowerShell):
        ```powershell
        .venv\Scripts\Activate.ps1
        ```

4.  **Install dependencies:**
    This command installs the project in editable mode (`-e`) along with all required development packages from `pyproject.toml`.
    ```bash
    uv pip install -e .[dev]
    ```
    You are now ready to run the scripts and notebooks!

---

## How to use
1. Open JupyterLab via ```bash jupyter lab``` and you are ready to execute the Cells within notebooks such as Baseline.ipynb
3. Run the file 'hyperparameter_search' via: python hyperparameter_search.py [n_trials:int] [task:str] [model_type:str] [augment:True|False]
4. The json with the corresponding parameters will be written in best_params.json
5. Load hyperparameters from json before training a model with those parameters (see TrainWithLoadedParams.ipynb)

## File explanations
1. data.py: Create data loaders
2. eval.py: Evaluation methods
3. train_generalized_earlystopping.py: Training methods
4. TrainWithLoadedParams.ipynb: Compare different models with loaded hyperparameters
5. hyperparameter_search.py: Optuna-based hyperparameter search (use with cmd command)
6. Baseline_Classification.ipynb: Definition, Training and Evaluation of Classification Model (VGG16)
7. BaselineUNet.py: Baseline Model
8. ImprovedUNet.py: Improved UNet Model
9. best_params.json: Stores best hyperparameters after running hyperparameter_search.py
10. DataExploration.ipynb: Outputs of data exploration
11. Baseline.ipynb: Outputs of training baseline model without mask filtering (full dataset)
12. OnlyWithMasks.ipynb: Outputs of training baseline model after doing mask filtering

---

## 1 – Dataset

| Detail                | Value                                                              |
| ----------------------| -------------------------------------------------------------------|
| Kaggle slug           | `kaggle/input/lgg-mri-segmentation/`                               |
| File type             | TIF                                                                |
| Image count           | Total 7858; for each scan 1 mask = 3929                            |
| Sub-Folders           | 110; 1 per patient                                                 |
| Image size            | 256 × 256 px                                                       |
| Info                  | Tumors are annotated by pixel-based segmentation masks             |


Download via the Kaggle CLI or web UI:

```bash
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
```


---

## 2 – Project Goal

1. **Establish a minimally viable baseline.**  
2. **Swap in a standard U‑Net** show clear improvement on the same split.  
3. **Introduce an advanced U‑Net (optional)** to improve results.
4. **Document every step** so that results can be reproduced.  

---

## 3 – Timeline & Milestones

| Week / Due Date         | Main Focus                                                        | Key Deliverables                                           | Done |
| ----------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------- | :--: |
| **≤ 09 May**            | **Research** – dataset inspection, planning                       | Timeline, Architecture choice, repo, define Metrics        | ✅   |
| **09 → 16 May**         | **Project & Data Setup → Baseline**                               | Training notebook/script, first evaluation, short Report   | ✅   |
| **16 → 23 May**         | **Classic U‑Net** – Implement vanilla U‑Net                       | Working Model, first segmentation examples                 | ✅   |
| **23 → 30 May**         | **Classic U‑Net** – Implement and test on valdata + Binary Model  | Metrics table, qualitative segmentation examples           | ✅   |
| **30 May → 06 Jun**     | **U‑Net tuning and eval explo** –  Regularization, tuning         | Ideas on how to finetune, make model  more complex         | ✅   |
| **06 → 20 Jun**         | **Advanced U‑Net** – Implement, tune, test                        | This is optional, depends on progress                      | ✅    |
| **20 → 27 Jun**         | **Docu and Slides** – Prepare Presenation                         | Final report, slides                                       | ✅    |