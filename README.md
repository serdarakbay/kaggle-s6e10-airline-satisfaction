# Kaggle Playground S6E10: Predicting Airline Satisfaction

My solution for [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10). Binary classification on synthetic tabular data generated from a real airline survey dataset (129k rows). Metric is ROC AUC.

## Results

| Stage | CV AUC | Public LB |
|---|---|---|
| Baseline LightGBM | 0.95838 | 0.95800 |
| Feature engineering + tuned LightGBM | 0.96059 | |
| Feature engineering + tuned XGBoost | 0.96062 | |
| Feature engineering + CatBoost | 0.96035 | |
| Hill climbing (own GBDTs + TabPFN OOF) | 0.96147 | |
| LR stack, own 14 models (`all14`) | 0.96158 | 0.96114 |
| LR stack, 138 OOFs incl. public ones (`lib138`) | 0.96209 | 0.96175 |
| LR stack, 164 OOFs incl. public ones (`lib164`) | **0.96218** | **0.96176** |

Public LB went from ~1823rd to ~529th out of ~2500 teams.

Private LB: _will add after the competition ends._

## What I did

**Features**
- Digit features from numeric columns (Age, Flight Distance, delays)
- A LightGBM trained on the original dataset, its prediction used as a feature
- Target rates from the original data grouped by Flight Distance (plus Class, Type of Travel, Customer Type, Age)
- Frequency encoding
- Route profile: rows with the same Flight Distance treated as the same route, mean of each column per route and the row's difference from it

**Models** 

LightGBM, XGBoost and CatBoost, 5-fold stratified CV, OOF predictions saved for each.

**Ensembling**

Started with hill climbing on OOFs. The final version is a logistic regression stacker on logits of OOF predictions, scored with nested CV so the stacker weights don't leak into the CV score. Adding public OOFs from other Kagglers gave the biggest jump.

**Final picks**

I picked two stacks that are built differently instead of the two best scores: `lib164` (uses public OOFs) and `all14` (only my own models). `lib138` scored almost the same as `lib164` but it's basically the same file, so if one fails on private LB the other fails too.

## What didn't work
These didn't improve CV, so I dropped them:
- Pairwise interaction features
- Summary stats over the service rating columns
- Target encoding (code is still in notebook 01, not used)
- LightGBM as the meta model instead of LR

A feature that helps a single model can add nothing once there's a strong stack on top.

## What I learned
- Trust CV, not public LB. Public LB is 20% of the test set.
- Before reacting to an LB change, check rank correlation between the two submissions.
- Stacking gains come from different models, not more of the same model.
- If the stacker is fit on OOFs, you need nested CV to get an honest score.

## Files
```
notebooks/
  01_base_models.ipynb   feature engineering, LightGBM / XGBoost / CatBoost, blends
  02_stacking.ipynb      LR stacking with nested CV (all14, lib138, lib164)
```
Notebook 02 reads the OOF files that notebook 01 saves.
Competition data isn't included (Kaggle rules), download it from the competition page. Notebooks were run on Kaggle with GPU.

## Credits
- Chris Deotte: TabPFN and other tabular foundation model OOFs, public OOF library
- abdullahsafwan333: `aviator-oof-dataset`
- koumeimaki: `s6e10-route-tfm-nn-oof`
