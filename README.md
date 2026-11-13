# AttentiveMultiBranch-EEG-AD-FTD

Code and results for the paper:

**An Attentive Multi-Branch Deep Learning Framework for EEG-Based Classification of Alzheimer's Disease, Frontotemporal Dementia, and Cognitively Normal Subjects with Subject-Independent Validation**

## 1. Dataset

The EEG data are public and are **not** included in this repository.

- OpenNeuro ds004504, version 1.0.6: https://doi.org/10.18112/openneuro.ds004504.v1.0.6
- 88 subjects (36 AD, 23 FTD, 29 CN), resting-state eyes-closed EEG, 19 channels.
- All notebooks use the preprocessed recordings in `derivatives/` (band-pass 0.5–45 Hz, A1–A2 reference, ASR, ICA with ICLabel) and `participants.tsv`.
- The original binary notebooks download a Kaggle mirror of the dataset (`thngdngvn/openneuro-ds004504`) and need a personal `kaggle.json` API file, which is not included. The dataset can also be downloaded from OpenNeuro; set `DATASET_ROOT` in the first cells.
- The subject-independent notebooks run in Google Colab and accept an uploaded dataset ZIP or a local copy of the dataset folder.

## 2. Repository contents

```
notebooks/
  subject_independent/        5-fold subject-grouped cross-validation (main results of the revised paper)
  original_segment_level/
    AD_CN/                    10-fold segment-level cross-validation, AD vs CN
    FTD_CN/                   10-fold segment-level cross-validation, FTD vs CN
    three_class/              70/10/20 locked test, calibration, Grad-CAM, t-SNE
folds/                        subject IDs in each fold (train / validation / test)
results/                      result files reported in the paper
requirements.txt
```

### 2.1 Subject-independent experiments (`notebooks/subject_independent/`)

| Notebook | What it does | Paper |
|---|---|---|
| `Notebook_1_ThreeClass_Subjectwise.ipynb` | Three-class model (temporal + Welch PSD + 10 spectral biomarkers); 5-fold StratifiedGroupKFold over subjects; inner validation subjects for checkpoint selection; normalization from training subjects only; segment- and subject-level predictions | Section 4.1 |
| `Binary_Subjectwise_Windows.ipynb` | AD–CN and FTD–CN with 5-fold subject-grouped cross-validation for each window length | Section 4.2 |
| `Notebook_2_ThreeClass_Subjectwise_Ablation_Statistics_Colab.ipynb` | Eight-variant ablation on the same subject folds; exact McNemar tests with Holm correction; class-stratified subject bootstrap | Section 4.3 |
| `Notebook_3_Subjectwise_Calibration_Bootstrap_Colab.ipynb` | Brier, ECE, MCE and NLL on held-out subjects; reliability bins with counts; 95% CIs from 1,500 subject bootstrap resamples | Section 4.4 |
| `Notebook_4_Binary_Confounders_Heldout_Saliency_Colab.ipynb` | Audit of the original binary fold files; recording duration, age and MMSE checks; input×gradient saliency on held-out subjects with permutation tests and FDR correction | Sections 4.5–4.6 |

Notebooks 2–4 read the results ZIP written by Notebook 1, so run Notebook 1 first.

### 2.2 Original segment-level experiments (`notebooks/original_segment_level/`)

These reproduce the results of the first submission (Section 5 of the revised paper). Segments from the same subject can appear in both training and test sets, so these results are subject-overlapping.

**`AD_CN/`**

| Notebook | Experiment |
|---|---|
| `a__final_windows_gpu_ready_notebook.ipynb` | 5 s windows |
| `b__notebook_10s_version.ipynb` | 10 s windows |
| `c__15s_version.ipynb` | 15 s windows |
| `d__20s_version.ipynb` | 20 s windows |
| `e__25s_version.ipynb` | 25 s windows |
| `f__30s_version.ipynb` | 30 s windows |
| `g_final_windows_30s_15s_overlap.ipynb` | 30 s windows, 15 s overlap |
| `time_domain_branch_only.ipynb` | Ablation, 5 s: time branch only |
| `frequency_domain_branch_only.ipynb` | Ablation, 5 s: frequency branch only |
| `dual_branch_without_se_attention.ipynb` | Ablation, 5 s: dual branch without SE |

**`FTD_CN/`**

| Notebook | Experiment |
|---|---|
| `ftd_cn_5s_paper10fold.ipynb` | 5 s windows |
| `ftd_cn_30s_paper10fold.ipynb` | 30 s windows |
| `ftd_cn_30s_15s_overlap_paper10fold.ipynb` | 30 s windows, 15 s overlap |
| `ftd_cn_time_domain_branch_only.ipynb` | Ablation, 5 s: time branch only |
| `ftd_cn_frequency_domain_branch_only.ipynb` | Ablation, 5 s: frequency branch only |
| `ftd_cn_dual_branch_without_se_attention.ipynb` | Ablation, 5 s: dual branch without SE |

**`three_class/`**

| Notebook | Experiment |
|---|---|
| `Last_Gradcam_etc_notebook.ipynb` | Three-class 70/10/20 locked test, 10-fold out-of-fold calibration, Grad-CAM, t-SNE |

Binary protocol: 10-fold `StratifiedKFold` over segments (shuffled, seed 42); 10% of the training segments held out for validation (`StratifiedShuffleSplit`, seed 42 + fold). "Tuned" results use the decision threshold, chosen from 181 values between 0.05 and 0.95, that gives the highest validation accuracy. Batch size, epochs and learning rate for each run are listed in Table 4 of the paper.

## 3. Preprocessing and segmentation

- 19 channels (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2), resampled from 500 Hz to 256 Hz, converted to µV.
- Three-class: non-overlapping 5 s windows; windows within 0.5 s of a boundary annotation are removed.
- Binary: non-overlapping windows of 5, 10, 15, 20, 25 and 30 s, and 30 s windows with 15 s overlap.

## 4. Main results (subject-independent)

| Task | Result |
|---|---|
| Three-class (AD / FTD / CN) | 69 of 88 subjects correct (78.41%), macro AUC 0.880 |
| AD–CN, 5 s | 84.62%, AUC 0.900 |
| FTD–CN, 5 s | 80.77%, AUC 0.860 |

The numbers in the paper can be checked against the files in `results/` and the saved notebook outputs.

## 5. Environment

| Notebooks | Platform | Versions |
|---|---|---|
| Original binary (AD–CN, FTD–CN) | Local GPU | Python 3.10.18, TensorFlow 2.10.0, MNE 1.11.0 |
| Original three-class | Kaggle, 2 × Tesla T4 | Python 3.12.13, TensorFlow 2.19.0 |
| Subject-independent | Google Colab, GPU | see `requirements.txt` |

Seeds: 42 for the original binary runs; 2026 (outer folds) and 2027 (inner folds) for the subject-independent splits; 2026 for the saliency permutation tests.

## License

MIT License
