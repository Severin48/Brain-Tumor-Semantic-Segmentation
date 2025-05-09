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
| Image count           | 7858                                                               |
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
3. **Introduce an advanced U‑Net flavor (optional)** (U‑Net++, Attention U‑Net, or similar) to push metrics further.  
4. **Document every step** so that results can be reproduced.  

---

## 3 – Timeline & Milestones
We use 06 -> 20 Jun as a buffer, depending on vanilla U-Net progress.

| Week / Due Date         | Main Focus                                                        | Key Deliverables                                           | Done |
| ----------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------- | :--: |
| **≤ 09 May**            | **Research** – dataset inspection, planning                       | Timeline, Architecture choice, repo, define Metrics        | ☐    |
| **09 → 16 May**         | **Project & Data Setup → Baseline**                               | Training notebook/script, first evaluation, short Report   | ☐    |
| **16 → 23 May**         | **Classic U‑Net** – implement vanilla U‑Net                       | Working Model, first segmentation examples                 | ☐    |
| **23 → 30 May**         | **Classic U‑Net** – implement and test on valdata                 | Metrics table, qualitative segmentation examples           | ☐    |
| **30 May → 06 Jun**     | **U‑Net tuning and eval** –  regularization, pre/post‑processing? | Full evaluation on testdata, extended Report               | ☐    |
| **06 → 20 Jun**         | **Advanced U‑Net** – implement, tune, test                        | This is optional, depends on progress                      | ☐    |
| **20 → 27 Jun**         | **Docu and Slides** – prepare Presenation                         | Final report, slides                                       | ☐    |


---

## 4 – Metrics

* **Primary:** accuracy
* **Secondary:** 

_All metrics are logged per‑epoch and summarized on the validation dataset; Hold‑out Test‑Set will only be used in the final report._  

---
## 5 – Upcoming Questions for Meetup

**9 May**: Timeline realistic? Implementing CNN first a good idea or go with u-net right away? is u-net the model to go, alternatives? advanced u-net realsitic? Dataset has bounding boxes, but we want segmentation how we match data?
    **Notes**: Two options => i. just do classification task without segmentation ii. change dataset to actual segmentation dataset => consider different model maybe ResNet. Send Decision for approvment by Monday 12.05[]
