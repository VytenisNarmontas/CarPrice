# CarPrice: step-by-step build guide

## Goal and boundaries

Build a Python project that predicts advertised BMW asking prices from a historical UK dataset. It should include data inspection, a simple baseline, a trained model, honest evaluation and a small local demonstration.

Version 1 uses model, registration year, mileage, fuel, transmission and engine size. Prices remain in GBP; mileage remains in miles. It does not claim to estimate today's Lithuanian prices. Keep the original year feature rather than deriving age from today's date, since the cars were listed in the past.

Python, VS Code and Git were found on your computer. The implementation folder suggested below is C:\dev\CarPrice. This guide is stored separately under C:\Stuff\Life\Projects\CarPrice. Only this guide has been created; the application and environment have not been installed or built for you.

## Working method

Work in one-hour sessions. Each session should produce one result you can show or explain. If the result is unfinished when the hour ends, write the next exact action and continue next time.

AI can explain documentation, propose small changes, review code and help debug. Before moving on, understand what your change does, run it yourself and be able to explain its inputs and outputs. You may use generated code; do not accept an entire unfamiliar application in one step.

## 1. Create and open the project

Open PowerShell and run:

```powershell
New-Item -ItemType Directory -Path C:\dev\CarPrice
Set-Location C:\dev\CarPrice
code .
```

If the folder already exists, open it and inspect its contents before creating anything with the same name.

In VS Code, choose Terminal > New Terminal. Make sure it opens in C:\dev\CarPrice. If needed:

```powershell
Set-Location C:\dev\CarPrice
```

Create a virtual environment and install the first libraries:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pandas scikit-learn matplotlib joblib
```

The virtual environment holds this project's packages. The commands below call its Python explicitly, so activation is optional. You do not need to change PowerShell's execution policy.

In VS Code: Ctrl+Shift+P > Python: Select Interpreter > select C:\dev\CarPrice\.venv\Scripts\python.exe. If that command is missing, install Microsoft's Python extension from the Extensions panel.

Done when: the package installation succeeds and VS Code uses the project's interpreter.

## 2. Create a small folder structure

Use the VS Code Explorer to create these folders and files as needed. Start with inspect_data.py; the other scripts come later.

```text
C:\dev\CarPrice\
  .venv\
  data\
    raw\
    processed\
  models\
  reports\
  inspect_data.py
  train.py
  evaluate.py
  app.py
  README.md
  DATA_SOURCE.md
  .gitignore
  requirements.txt
```

Do not write code inside .venv. Keep the original CSV unchanged in data\raw.

## 3. Download the dataset

Open [100,000 UK Used Car Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes). Download the dataset, signing in if requested, then extract the archive. Copy only bmw.csv into:

```text
C:\dev\CarPrice\data\raw\bmw.csv
```

The publisher labels this dataset CC0. Record the publisher, source URL, displayed licence, dataset version if available, download date and units in DATA_SOURCE.md. The dataset is historical and contains advertised listings. A downloaded file does not establish today's market values.

Done when: the CSV is in the expected folder and you have recorded its source.

## 4. Inspect the data

Put this starter code in inspect_data.py:

```python
import pandas as pd

cars = pd.read_csv("data/raw/bmw.csv")

print(cars.head())
print(cars.shape)
print(cars.columns)
print(cars.dtypes)
print(cars.isna().sum())
print(cars.duplicated().sum())
print(cars["model"].value_counts())
print(cars[["year", "price", "mileage", "engineSize"]].describe())
```

Run it from the project root:

```powershell
.\.venv\Scripts\python.exe inspect_data.py
```

Research these pandas operations: read_csv, head, shape, columns, dtypes, isna, duplicated, value_counts and describe. Ask for an explanation of any line you cannot read.

Answer in reports\data_notes.md:

- How many rows and columns exist?
- Which BMW models appear most often?
- Are values missing? Are there exact duplicate rows?
- What are the minimum/maximum prices, years and mileages?
- Are model names padded with spaces or inconsistent?
- Which columns can a user realistically supply when requesting a prediction?

Then make a price histogram and a mileage-versus-price scatter plot with matplotlib. Save them under reports.

Done when: you can describe the dataset and show two plots. This is a useful stopping point for the first session.

## 5. Define a simple cleaning policy

Write down the rules before training:

- Strip unnecessary spaces from column names and category labels.
- Remove exact duplicate rows before splitting.
- Remove rows without a usable target price.
- Investigate impossible values such as negative mileage or non-positive prices.
- Do not delete expensive or unusual cars merely because they are inconvenient to predict.
- Keep a count of rows removed by each rule.

A data-cleaning rule based on physical validity differs from estimating a statistic. Medians, scaling parameters, category encodings and other learned transformations must be fitted on the training partition only, through the model pipeline.

The BMW file may not identify individual vehicles. Exact deduplication cannot guarantee that a relisted car is absent from another partition. Document this limitation. If a future dataset has a reliable vehicle/listing grouping identifier, keep each vehicle's records together during splitting.

Done when: cleaning is repeatable, the original file is untouched and you can justify each exclusion.

## 6. Choose inputs and split the data

Start with:

```text
Inputs (X): model, year, transmission, mileage, fuelType, engineSize
Target (y): price
```

Confirm those names against the actual CSV. The file may have additional fields such as tax or mpg; leave those out of the first version to simplify the form.

Make a reproducible 70% training / 15% validation / 15% test split, for example by first setting aside 30% and then dividing that held-out portion equally. Use random_state=42 where applicable. Save the partitions under data\processed so train.py and evaluate.py use the same records.

- Training data: the model learns from it.
- Validation data: compare models and make development decisions.
- Test data: use after selecting the final approach to report an estimate of performance on unseen data.

Research sklearn.model_selection.train_test_split. Do not fit the model, imputer, scaler or encoder on validation/test rows. Never include price among the inputs.

Done when: partitions are reproducible and you can explain what each is for.

## 7. Build the baseline

In train.py, first use sklearn.dummy.DummyRegressor with strategy="median". Fit it on the training rows and calculate its validation MAE. This predicts the training-set median price for every car.

Write a short result in reports\model_comparison.csv:

```text
model,validation_mae_gbp
median_baseline,<your measured value>
```

Do not copy an example score. Your baseline establishes how well a very simple approach performs on your data split.

Optional later: a stronger baseline using the training-set median for each BMW model, with a fallback for unseen categories.

Done when: you have an actual validation baseline and understand why it exists.

## 8. Build one preprocessing pipeline

Use scikit-learn's ColumnTransformer and Pipeline:

- Text categories: model, transmission, fuelType.
- Numeric inputs: year, mileage, engineSize.
- Missing categories: SimpleImputer with an appropriate fixed/mode strategy.
- Category conversion: OneHotEncoder with handle_unknown="ignore".
- Missing numeric values: SimpleImputer using training-set medians.
- For a linear model, consider StandardScaler for numeric features.

One-hot encoding turns labels into numeric indicator columns. A pipeline keeps all preprocessing attached to the estimator so training and later predictions use the same transformations.

Research each piece and build the pipeline in small steps. Fit it only on the training partition. Do not save a bare estimator that requires you to reconstruct preprocessing manually in the demo.

Done when: the pipeline can fit on the training data and predict the validation rows, including a category not encountered during training without crashing.

## 9. Compare a linear model and a random forest

Try Ridge regression as the linear model, then RandomForestRegressor. Start with modest settings rather than an enormous parameter search. Use random_state where the estimator supports it.

Keep the same partitions and inputs for a fair comparison. Record each model's validation MAE and approximate training/prediction time. If useful, investigate just a few forest settings, such as the number of trees and minimum samples per leaf.

Explain:

- What does each model learn, at a high level?
- Did it beat the baseline?
- Why might it perform better or worse?
- Is training error much lower than validation error? What could that mean?

If the model performs badly, first inspect data handling and the task definition. Do not keep adding libraries without understanding the result.

Done when: you have measured comparisons and can justify your selected approach.

## 10. Evaluate the chosen approach

Freeze your choice of inputs, preprocessing and settings. You may then refit the selected pipeline on training plus validation data before evaluating once on the reserved test partition. Refit the baseline on that same training-plus-validation data too, so the final comparison uses the same available training records.

In evaluate.py:

- Calculate test MAE in GBP and report it alongside the baseline evaluated on the same test rows.
- Inspect absolute errors by BMW model, year range and price band. Include group sizes so tiny groups do not look conclusive.
- Save actual-versus-predicted and error plots.
- Inspect examples with large errors and document plausible limitations.

MAE is average absolute error; it is not a guaranteed plus/minus bound for an individual prediction. Avoid a made-up confidence percentage or price range. Calibrated prediction intervals can be a later feature.

Looking at test failures is useful for reporting. If you use them to tune the model further, that set is now influencing development and should not be described as an untouched final test.

Done when: reports explain performance, limitations and a few failures with actual measured values.

## 11. Save and reload the complete pipeline

Save your selected fitted pipeline with joblib under models\carprice.joblib. Save the expected input columns and relevant metadata, such as units, supported year/model ranges, dataset version and evaluation results.

In a new Python process, load your saved pipeline and predict one row containing exactly the required inputs. Confirm that this agrees with the in-memory prediction for that row.

Done when: inference works independently of the training script.

## 12. Build the local demonstration

Install Streamlit in your environment:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit
```

Create app.py with:

- A BMW model selector using the dataset's real labels.
- Registration year, mileage in miles and engine size in litres.
- Fuel and transmission selectors.
- A Predict button and a price displayed in GBP.
- A visible note that this is a historical UK asking-price model.
- A short display of the measured overall test error, labelled as an average.

Load the saved pipeline instead of retraining when the form changes. Make a one-row pandas DataFrame with the same feature names as the training inputs. Use input bounds/support checks so the app does not confidently present predictions for wildly unsupported years or engine sizes.

Run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Done when: the form predicts from the saved model and another person can understand the units and limitations.

Optional later: show similar cars or vary mileage while keeping other inputs fixed. Interpret this as model behaviour, not proof of a causal effect.

## 13. Add a few meaningful checks

Use small checks that protect real behaviour:

- A saved/reloaded pipeline produces the same prediction for the same input.
- The target column never enters the input features.
- Missing values and an unseen category follow the intended preprocessing behaviour.
- Re-running data preparation with the same inputs produces the same split.
- Inference returns one finite numeric prediction per input row.

If you add group identifiers, also check that groups do not cross partitions. Do not invent a test requiring every possible lower-mileage car to have a higher prediction: a learned model does not automatically enforce that rule.

## 14. Make it reproducible and suitable for GitHub

Create .gitignore before staging files:

```gitignore
.venv/
__pycache__/
.pytest_cache/
data/
models/
```

Keep download/rebuild instructions for data and models in README.md. You do not need to publish the raw dataset to demonstrate the project.

In the project environment, run:

```powershell
.\.venv\Scripts\python.exe -m pip freeze
```

Copy the resulting package/version list into requirements.txt using VS Code. Reproducibility matters more than guessing package versions in advance. Record your Python version too.

Initialize local Git:

```powershell
git init
git status
```

Inspect the files before staging and committing. Only push to GitHub when you are ready to publish the project.

The README should explain:

1. The problem and project scope.
2. Dataset source, licence, units and age.
3. Installation, download, training, evaluation and app commands.
4. Inputs, preprocessing, split design and baseline.
5. Actual results and important failure cases.
6. Limitations and your contribution, including relevant AI assistance.
7. A screenshot and a short demonstration video.

Done when: someone can follow your instructions to reproduce the project.

## 15. Put the result on your CV

Use only measured results. Template:

> Built a BMW asking-price prediction pipeline using Python, pandas and scikit-learn; improved held-out MAE by [measured percentage] against a median baseline and delivered a Streamlit demonstration with reproducible evaluation.

Percentage improvement is (baseline MAE - model MAE) / baseline MAE * 100, using the same evaluation rows. If it does not improve, say what you evaluated and learned rather than claiming an improvement.

Be able to answer: Why this dataset? Why this split? What is leakage? What is your baseline? Why this model? Where does it fail? What did you personally decide, implement or improve?

## Suggested one-hour sessions

1. Environment, dataset and first inspection.
2. Data notes, plots and cleaning policy.
3. Reproducible split and median baseline.
4. Preprocessing pipeline and linear model.
5. Random forest and validation comparison.
6. Final evaluation, error analysis and model persistence.
7. Streamlit demo and useful checks.
8. README, reproducibility check, demo video and CV wording.

These are session targets, not a promise that the whole project takes eight hours. Allow extra sessions to understand unfamiliar concepts and debug. Preserve the scope while learning.

## Primary documentation

- [Dataset publisher](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes).
- [Python virtual environments](https://docs.python.org/3/library/venv.html): environment creation and use.
- [pandas getting started](https://pandas.pydata.org/docs/getting_started/intro_tutorials/): reading and examining tables.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): leakage and consistent preprocessing.
- [scikit-learn MAE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html): meaning of the evaluation metric.
- [Streamlit installation](https://docs.streamlit.io/get-started/installation/command-line): local app setup and running.
