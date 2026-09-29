# Adult Income Prediction using Machine Learning and MLOps

## 📌 Project Overview

This project develops an end-to-end **Machine Learning classification system** to predict whether an individual's annual income is:

- `<=50K`
- `>50K`

The prediction is based on demographic, educational, occupational, and employment-related attributes.

The project demonstrates both a complete Machine Learning workflow and an MLOps workflow, including:

- Data loading and understanding
- Data cleaning
- Missing-value handling
- Exploratory Data Analysis (EDA)
- Feature and target separation
- Categorical encoding
- Numerical feature scaling
- Train-test splitting
- Machine Learning model training
- Model evaluation and comparison
- Hyperparameter tuning
- Final model selection
- Prediction on unseen data
- Data validation
- Production preprocessing pipelines
- MLflow experiment tracking
- Model registry
- Model versioning
- Model lifecycle management
- Automated pipeline execution

---

# 🎯 Problem Statement

The objective is to build a binary classification model that can predict an individual's income category using information such as:

- Age
- Workclass
- Education
- Education number
- Marital status
- Occupation
- Relationship
- Race
- Sex
- Capital gain
- Capital loss
- Hours worked per week
- Native country

The target variable is `income`.

---

# 📊 Dataset

The project uses the **Adult Income dataset** in CSV format.

## Target Variable

| Value | Meaning |
|---|---|
| `<=50K` | Annual income is less than or equal to $50,000 |
| `>50K` | Annual income is greater than $50,000 |

The target is converted into binary values for Machine Learning:

```text
<=50K → 0
>50K  → 1
````

## Dataset Size

```text
Samples: 32,561
Columns: 15
```

## Dataset Files

```text
adult.csv
adult_cleaned.csv
```

---

# 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine Learning and model evaluation
* **MLflow** – Experiment tracking and model registry
* **Pandera** – Data validation
* **Joblib** – Model/preprocessor serialization
* **Pytest** – Testing
* **Jupyter Notebook / VS Code** – Development environment
* **Git**
* **GitHub**

---

# 📂 Project Structure

```text
Adult-Income-Prediction/
│
├── configs/
│   └── config.py
│
├── data/
│   ├── raw/
│   │   └── adult.csv
│   │
│   └── processed/
│       ├── X_train_final.npy
│       ├── X_test_final.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── dataset_metadata.json
│       └── adult_cleaned.csv
│
├── models/
│   ├── preprocessor.pkl
│   └── random_forest_model.pkl
│
├── outputs/
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── predictions.csv
│
├── pipelines/
│   ├── run_lab5_pipeline.py
│   └── run_lab6_registry.py
│
├── reports/
│   ├── model_evaluation_report.md
│   └── production_model_report.json
│
├── src/
│   ├── income_prediction.py
│   ├── validate_data.py
│   ├── preprocess.py
│   ├── preprocess_pipeline.py
│   ├── train.py
│   ├── train_mlflow.py
│   ├── evaluate.py
│   ├── validate_outputs.py
│   ├── train_registry.py
│   ├── automate_lifecycle.py
│   └── generate_registry_report.py
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_evaluate.py
│
├── .gitignore
├── README.md
└── requirments.txt
```

> Local MLflow tracking files such as `mlruns/` and `mlflow.db` are intentionally excluded from GitHub.

---

# 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Missing & Unknown Value Handling
   ↓
Duplicate Removal
   ↓
Exploratory Data Analysis
   ↓
Feature & Target Separation
   ↓
Train-Test Split
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Final Model Evaluation
   ↓
Prediction on New Data
```

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed.

## 1. Unknown Value Handling

The Adult dataset contains unknown categorical values represented by `?`.

These values are converted into `NaN`:

```python
df = df.replace("?", np.nan)
```

## 2. Duplicate Removal

Duplicate records are identified and removed:

```python
df = df.drop_duplicates()
```

## 3. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

Stratification is used to maintain the target-class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

## 4. Numerical Feature Processing

Numerical features are processed using:

* Median imputation
* Standard scaling

```text
Missing Values
       ↓
Median Imputation
       ↓
StandardScaler
```

## 5. Categorical Feature Processing

Categorical features are processed using:

* Most-frequent imputation
* One-hot encoding

```text
Missing Values
       ↓
Most Frequent
       ↓
One-Hot Encoding
```

The preprocessing steps are integrated into a Scikit-learn pipeline to prevent information leakage from the test dataset.

---

# 📈 Exploratory Data Analysis

Several visualizations were created to understand the dataset and identify relationships with income.

## Visualizations Include

* Income distribution
* Age distribution
* Education distribution
* Workclass distribution
* Occupation distribution
* Working-hours distribution
* Education vs Income
* Age vs Income
* Working Hours vs Income
* Sex vs Income
* Correlation matrix
* Numerical feature boxplots

EDA helps understand the distribution of the data before model training.

---

# 🤖 Machine Learning Models

Four classification algorithms were initially trained and compared.

## 1. Logistic Regression

Used as a baseline classification model.

## 2. Decision Tree

Used to capture non-linear decision boundaries and provide an interpretable model.

## 3. Random Forest

An ensemble of decision trees designed to capture more complex relationships in tabular data.

## 4. Gradient Boosting

A sequential ensemble method where each new model attempts to improve previous errors.

---

# 📊 Model Evaluation

The following metrics were used:

## Accuracy

Measures the percentage of correctly classified observations.

## Precision

Measures how many observations predicted as `>50K` were actually `>50K`.

## Recall

Measures how many actual `>50K` observations were correctly identified.

## F1 Score

Provides a balance between precision and recall.

## ROC-AUC

Measures the model's ability to distinguish between the two income classes.

Because the target classes are imbalanced, **F1 Score** was given particular importance when selecting the model.

---

# 🏆 Model Comparison

The actual results generated by the project are:

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Gradient Boosting   | **0.8591** |     0.7724 |     0.5886 | **0.6681** | **0.9138** |
| Tuned Random Forest |     0.8573 | **0.7800** |     0.5676 |     0.6571 |     0.9077 |
| Logistic Regression |     0.8502 |     0.7344 |     0.5925 |     0.6558 |     0.9004 |
| Random Forest       |     0.8467 |     0.7149 | **0.6046** |     0.6551 |     0.8925 |
| Decision Tree       |     0.8099 |     0.6052 |     0.6071 |     0.6062 |     0.7407 |

## Best Initial Model

Based on the recorded comparison results, Gradient Boosting achieved:

```text
F1 Score = 0.6681
ROC-AUC  = 0.9138
```

These results are from the original Machine Learning model comparison.

---

# 🔧 Hyperparameter Tuning

Hyperparameter tuning was performed for the Random Forest model using `GridSearchCV`.

The parameters explored included:

```python
parameter_grid = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [10, 15],
    "classifier__min_samples_split": [2, 5]
}
```

Three-fold cross-validation was used with **F1 Score** as the optimization metric.

The tuned Random Forest achieved:

```text
Accuracy : 0.8573
Precision: 0.7800
Recall   : 0.5676
F1 Score : 0.6571
ROC-AUC  : 0.9077
```

---

# 🔍 Confusion Matrix

A confusion matrix was generated for the Random Forest model to analyze:

* True Negatives
* False Positives
* False Negatives
* True Positives

The target labels were:

```text
<=50K
>50K
```

A classification report was also generated to provide class-level precision, recall, and F1 Score.

---

# 🧪 Prediction on New Data

The trained model was also tested on a new unseen individual.

Example input:

```python
{
    "age": 35,
    "workclass": "Private",
    "fnlwgt": 180000,
    "education": "Bachelors",
    "education.num": 13,
    "marital.status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital.gain": 0,
    "capital.loss": 0,
    "hours.per.week": 45,
    "native.country": "United-States"
}
```

The model produces:

```text
Predicted income: <=50K or >50K
```

The prediction probability for both classes is also calculated.

---

# ⚙️ MLOps Implementation

The Machine Learning project was extended with an MLOps workflow covering **Lab 3, Lab 4, Lab 5, and Lab 6**.

```text
                 Adult Income Dataset
                         │
                         ▼
                Data Validation
                         │
                         ▼
                 Preprocessing
                         │
                         ▼
                  Model Training
                         │
                         ▼
                 Model Evaluation
                         │
                         ▼
                MLflow Tracking
                         │
                         ▼
             Production Data Pipeline
                         │
                         ▼
                  Model Registry
                         │
                         ▼
              Model Lifecycle
```

---

# 🧪 Lab 3 – Machine Learning Pipeline

Lab 3 implements the basic Machine Learning pipeline.

```text
Dataset
   ↓
Data Validation
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
```

## Main Files

```text
src/validate_data.py
src/preprocess.py
src/train.py
src/evaluate.py
```

The pipeline performs:

* Dataset loading
* Data validation
* Data cleaning
* Feature preprocessing
* Model training
* Model evaluation
* Output generation

---

# 📊 Lab 4 – MLflow Experiment Tracking

MLflow is used to track Machine Learning experiments.

The following information is tracked:

### Parameters

* Model parameters
* Random Forest configuration
* Dataset-related parameters

### Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Artifacts

* Trained models
* Preprocessing information
* Evaluation outputs
* Metadata

The MLflow tracking workflow provides experiment reproducibility and model artifact management.

---

# 🏭 Lab 5 – Production Data Pipeline

Lab 5 converts the preprocessing workflow into a production-oriented Scikit-learn pipeline.

## Production Pipeline

```text
Adult Income Dataset
        ↓
Data Validation
        ↓
Missing Value Handling
        ↓
Feature Separation
        ↓
Numerical Processing
        ↓
Categorical Processing
        ↓
One-Hot Encoding
        ↓
Feature Scaling
        ↓
Train/Test Split
        ↓
Processed Dataset
```

## Validation Result

The dataset validation produced:

```text
Dataset shape: (32561, 15)

All required columns are present.

Adult Income dataset validation passed.
```

## Production Preprocessing Result

```text
Training samples   : 26048
Testing samples    : 6513
Processed features : 104
```

## Generated Files

```text
data/processed/X_train_final.npy
data/processed/X_test_final.npy
data/processed/y_train.npy
data/processed/y_test.npy
data/processed/dataset_metadata.json

models/preprocessor.pkl
```

## Run Lab 5

```powershell
python pipelines\run_lab5_pipeline.py
```

Successful execution:

```text
[SUCCESS] Lab 5 Production Pipeline fully executed!
```

---

# 🗂️ Lab 6 – Model Registry & Lifecycle

Lab 6 implements Model Registry and Model Lifecycle Management using MLflow.

## Registered Model

```text
Adult_Income_Production_Model
```

## Model Version

```text
Version 1
```

## Model Run ID

```text
58570570e3564919a426699276ba7560
```

## Production Model Metrics

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.7901 |
| Precision | 0.5396 |
| Recall    | 0.8737 |
| F1 Score  | 0.6672 |
| ROC-AUC   | 0.9036 |

## Local Model

The production model is also saved locally as:

```text
models/random_forest_model.pkl
```

## Registry Report

The model registry report is generated at:

```text
reports/production_model_report.json
```

## Run Lab 6

```powershell
python pipelines\run_lab6_registry.py
```

Successful execution:

```text
[SUCCESS] Lab 6 Model Registry Pipeline fully executed!
```

---

# 📊 Production Model

The registered production model is a **Random Forest Classifier**.

The recorded production model evaluation was:

```text
Accuracy : 0.7901
Precision: 0.5396
Recall   : 0.8737
F1 Score : 0.6672
ROC-AUC  : 0.9036
```

These metrics belong to the registered production model run and are separate from the original model-comparison results documented earlier in this README.

---

# 📈 MLflow UI

MLflow can be used to visualize experiments, runs, metrics, parameters, artifacts, and registered models.

## Start MLflow

From the project directory:

```powershell
mlflow ui
```

MLflow runs locally at:

```text
http://127.0.0.1:5000
```

Open the address in a web browser.

## Registered Model

```text
Adult_Income_Production_Model
```

The MLflow UI provides access to:

* Experiments
* Runs
* Parameters
* Metrics
* Model artifacts
* Registered models
* Model versions

---

# 🧪 Production Output Validation

Lab 5 also validates the generated preprocessing outputs.

The following files are checked:

```text
[OK] data/processed/X_train_final.npy
[OK] data/processed/X_test_final.npy
[OK] data/processed/y_train.npy
[OK] data/processed/y_test.npy
[OK] data/processed/dataset_metadata.json
[OK] models/preprocessor.pkl
```

The validation confirms that all required preprocessing artifacts were generated successfully.

---

# 🧪 Testing

Unit tests are included for the main Machine Learning components.

```text
tests/
├── test_preprocess.py
├── test_train.py
└── test_evaluate.py
```

Run the tests using:

```powershell
pytest
```

---

# ▶️ How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Surya8522/Adult-Income-Prediction.git
```

## 2. Open the Project

```bash
cd Adult-Income-Prediction
```

## 3. Create Virtual Environment

```powershell
python -m venv .venv313
```

## 4. Activate Virtual Environment

```powershell
.venv313\Scripts\Activate.ps1
```

## 5. Install Dependencies

```powershell
pip install -r requirments.txt
```

## 6. Validate Dataset

```powershell
python src\validate_data.py
```

## 7. Run Preprocessing

```powershell
python src\preprocess.py
```

## 8. Train Model

```powershell
python src\train.py
```

## 9. Evaluate Model

```powershell
python src\evaluate.py
```

## 10. Run Lab 5

```powershell
python pipelines\run_lab5_pipeline.py
```

## 11. Run Lab 6

```powershell
python pipelines\run_lab6_registry.py
```

## 12. Start MLflow

```powershell
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

# 📌 Complete MLOps Workflow

```text
                         ┌─────────────────────┐
                         │  Adult Income Data  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Data Validation    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Preprocessing     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Model Training    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Model Evaluation   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ MLflow Experiment   │
                         │     Tracking        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Production Data     │
                         │     Pipeline        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Model Registry    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Model Lifecycle     │
                         │    Management       │
                         └─────────────────────┘
```

---

# 💾 Output Files

The project generates several output files.

## Cleaned Dataset

```text
adult_cleaned.csv
```

Contains the cleaned version of the dataset after data preparation.

## Model Comparison

```text
outputs/model_comparison.csv
```

Contains performance metrics of the trained Machine Learning models.

## Confusion Matrix

```text
outputs/confusion_matrix.png
```

Provides a visual representation of classification results.

## Classification Report

```text
outputs/classification_report.txt
```

Contains class-level evaluation metrics.

## Predictions

```text
outputs/predictions.csv
```

Contains predictions generated by the trained model.

## Evaluation Report

```text
reports/model_evaluation_report.md
```

Contains the model evaluation information.

## Production Model Report

```text
reports/production_model_report.json
```

Contains information related to the registered production model.

---

# 🔐 Git and GitHub

The project is version controlled using Git and hosted on GitHub.

Repository:

```text
https://github.com/Surya8522/Adult-Income-Prediction
```

The following local files are excluded from GitHub:

```text
.venv/
.venv313/
mlruns/
mlflow.db
__pycache__/
.pytest_cache/
.ipynb_checkpoints/
.vscode/
*.log
```

This keeps local environment files and MLflow tracking data outside the source repository.

---

# 📚 Key Learning Outcomes

Through this project, the following Machine Learning and MLOps concepts are demonstrated:

## Machine Learning

* Binary classification
* Data cleaning
* Missing-value handling
* Duplicate removal
* Exploratory Data Analysis
* Feature-target separation
* Train-test splitting
* Stratified sampling
* Numerical feature scaling
* Categorical feature encoding
* Scikit-learn pipelines
* ColumnTransformer
* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Model evaluation
* Confusion matrix
* Classification report
* ROC-AUC
* F1 Score
* Hyperparameter tuning
* GridSearchCV
* Cross-validation
* Prediction on unseen data

## MLOps

* Data validation
* Production preprocessing
* Pipeline automation
* MLflow experiment tracking
* Model artifact management
* Model registry
* Model versioning
* Model lifecycle management
* Production model reporting
* Unit testing
* Git version control
* GitHub repository management

---

# ⚠️ Limitations

* The model is trained on the available Adult Income dataset and may not generalize to different populations.
* Income prediction is influenced by many factors that may not be represented in the dataset.
* The project is intended for educational and Machine Learning demonstration purposes.
* The model should not be used as the sole basis for real-world financial or employment decisions.

---

# 🚀 Future Improvements

Possible improvements include:

* Feature importance analysis
* ROC curve visualization
* Precision-Recall curve
* Additional ensemble models
* Advanced hyperparameter optimization
* Cross-validation comparison
* Model explainability using SHAP
* Deployment using Flask or FastAPI
* Interactive deployment using Streamlit
* Creating a web interface for real-time predictions
* CI/CD integration
* Docker containerization
* Cloud deployment
* Model monitoring
* Data drift detection
* Automated model retraining

---

# 📌 Project Status

| Component                   | Status       |
| --------------------------- | ------------ |
| Dataset Validation          | ✅ Completed  |
| Data Preprocessing          | ✅ Completed  |
| Model Training              | ✅ Completed  |
| Model Evaluation            | ✅ Completed  |
| Lab 3 – ML Pipeline         | ✅ Completed  |
| Lab 4 – MLflow Tracking     | ✅ Completed  |
| Lab 5 – Production Pipeline | ✅ Completed  |
| Lab 6 – Model Registry      | ✅ Completed  |
| Production Model            | ✅ Registered |
| Model Version 1             | ✅ Registered |
| MLflow UI                   | ✅ Configured |
| Unit Tests                  | ✅ Included   |
| GitHub Repository           | ✅ Uploaded   |

---

# 📜 Conclusion

This project demonstrates a complete **end-to-end Machine Learning and MLOps workflow** for predicting annual income categories.

The Machine Learning workflow starts with raw data and progresses through cleaning, exploratory analysis, preprocessing, model development, evaluation, comparison, hyperparameter tuning, and prediction on unseen data.

The MLOps workflow extends the project with data validation, production preprocessing, MLflow experiment tracking, model artifact management, model registration, versioning, and lifecycle management.

The original model comparison recorded Gradient Boosting with an F1 Score of `0.6681` and ROC-AUC of `0.9138`.

The registered production Random Forest model recorded:

```text
Accuracy : 0.7901
Precision: 0.5396
Recall   : 0.8737
F1 Score : 0.6672
ROC-AUC  : 0.9036
```

Overall, the project provides practical experience in building, evaluating, tracking, registering, and managing Machine Learning models for structured/tabular data.

---

# 👨‍💻 Author

**SuryaPrakash Balusu**

Computer Science & Engineering – Artificial Intelligence & Machine Learning

**Technologies:**

```text
Python | Pandas | NumPy | Scikit-learn | Matplotlib |
Seaborn | MLflow | Pandera | Pytest | Git | GitHub
```
