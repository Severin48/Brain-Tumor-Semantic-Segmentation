## How to use
1. Run the file 'hyperparameter_search' via: python hyperparameter_search.py [n_trials:int] [task:str] [model_type:str] [augment:True|False]
2. The json with the corresponding parameters will be written in best_params.json
3. Load hyperparameters from json before training a model with those parameters (see TrainWithLoadedParams.ipynb)

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


# Brain‑Tumor MRI Semantic Segmentation

AI LAB HKA Project: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Brain MRI images together with manual FLAIR abnormality segmentation masks

**Brain MRI segmentation** by Mateusz Buda on Kaggle.

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
We use 06 -> 20 Jun as a buffer, depending on vanilla U-Net progress.

| Week / Due Date         | Main Focus                                                        | Key Deliverables                                           | Done |
| ----------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------- | :--: |
| **≤ 09 May**            | **Research** – dataset inspection, planning                       | Timeline, Architecture choice, repo, define Metrics        | ✅   |
| **09 → 16 May**         | **Project & Data Setup → Baseline**                               | Training notebook/script, first evaluation, short Report   | ✅   |
| **16 → 23 May**         | **Classic U‑Net** – Implement vanilla U‑Net                       | Working Model, first segmentation examples                 | ✅   |
| **23 → 30 May**         | **Classic U‑Net** – Implement and test on valdata + Binary Model  | Metrics table, qualitative segmentation examples           | ✅   |
| **30 May → 06 Jun**     | **U‑Net tuning and eval explo** –  Regularization, tuning         | Ideas on how to finetune, make model  more complex         | ✅   |
| **06 → 20 Jun**         | **Advanced U‑Net** – Implement, tune, test                        | This is optional, depends on progress                      | ✅    |
| **20 → 27 Jun**         | **Docu and Slides** – Prepare Presenation                         | Final report, slides                                       | ✅    |