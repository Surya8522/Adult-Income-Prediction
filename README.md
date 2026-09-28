# Adult Income Prediction using Machine Learning

## 📌 Project Overview

This project develops an end-to-end **Machine Learning classification system** to predict whether an individual's annual income is:

* `<=50K`
* `>50K`.

The prediction is based on demographic, educational, occupational, and employment-related attributes.

This project demonstrates a complete Machine Learning workflow, including:

* Data loading and understanding
* Data cleaning
* Missing-value handling
* Exploratory Data Analysis (EDA)
* Feature and target separation
* Categorical encoding
* Numerical feature scaling
* Train-test splitting
* Machine Learning model training
* Model evaluation and comparison
* Hyperparameter tuning
* Final model selection
* Prediction on unseen data 


---

## 🎯 Problem Statement

The objective is to build a binary classification model that can predict an individual's income category using information such as:

* Age
* Workclass
* Education
* Education number
* Marital status
* Occupation
* Relationship
* Race
* Sex
* Capital gain
* Capital loss
* Hours worked per week
* Native country

The target variable is `income`.

---

## 📊 Dataset

The project uses the **Adult Income dataset** in CSV format.

### Target Variable

| Value   | Meaning                                        |
| ------- | ---------------------------------------------- |
| `<=50K` | Annual income is less than or equal to $50,000 |
| `>50K`  | Annual income is greater than $50,000          |

The target is converted into binary values for Machine Learning:

```text
<=50K → 0
>50K  → 1
```

### Dataset Files

```text
adult.csv
adult_cleaned.csv
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine Learning and model evaluation
* **Jupyter Notebook / VS Code** – Development environment

---

## 📂 Project Structure
```text
Adult-Income-Prediction/
│
├── configs/
│   └── config.py
│
├── data/
│   ├── raw/
│   │   └── adult.csv
│   └── processed/
│       └── adult_cleaned.csv
│
├── logs/
│   └── training.log
│
├── models/
│   └── final_model.pkl
│
├── notebooks/
│   └── income_prediction.ipynb
│
├── outputs/
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── predictions.csv
│
├── reports/
│   └── model_evaluation_report.md
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_evaluate.py
│
├── README.md
└── requirements.txt
```

---

## 🔄 Machine Learning Workflow

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

## 🧹 Data Preprocessing

The following preprocessing steps were performed.

### 1. Unknown Value Handling

The Adult dataset contains unknown categorical values represented by `?`.

These values are converted into `NaN`:

```python
df = df.replace("?", np.nan)
```

### 2. Duplicate Removal

Duplicate records are identified and removed:

```python
df = df.drop_duplicates()
```

### 3. Train-Test Split

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

### 4. Numerical Feature Processing

Numerical features are processed using:

* Median imputation
* Standard scaling

```text
Missing Values → Median Imputation → StandardScaler
```

### 5. Categorical Feature Processing

Categorical features are processed using:

* Most-frequent imputation
* One-hot encoding

```text
Missing Values → Most Frequent → One-Hot Encoding
```

The preprocessing steps are integrated into a Scikit-learn pipeline to prevent information leakage from the test dataset.

---

## 📈 Exploratory Data Analysis

Several visualizations were created to understand the dataset and identify relationships with income.

### Visualizations Include

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

## 🤖 Machine Learning Models

Four classification algorithms were initially trained and compared.

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Decision Tree

Used to capture non-linear decision boundaries and provide an interpretable model.

### 3. Random Forest

An ensemble of decision trees designed to capture more complex relationships in tabular data.

### 4. Gradient Boosting

A sequential ensemble method where each new model attempts to improve previous errors.

---

## 📊 Model Evaluation

The following metrics were used:

### Accuracy

Measures the percentage of correctly classified observations.

### Precision

Measures how many observations predicted as `>50K` were actually `>50K`.

### Recall

Measures how many actual `>50K` observations were correctly identified.

### F1 Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two income classes.

Because the target classes are imbalanced, **F1 Score** was given particular importance when selecting the best model.

---

## 🏆 Model Comparison

The actual results generated by the project are:

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Gradient Boosting   | **0.8591** |     0.7724 |     0.5886 | **0.6681** | **0.9138** |
| Tuned Random Forest |     0.8573 | **0.7800** |     0.5676 |     0.6571 |     0.9077 |
| Logistic Regression |     0.8502 |     0.7344 |     0.5925 |     0.6558 |     0.9004 |
| Random Forest       |     0.8467 |     0.7149 | **0.6046** |     0.6551 |     0.8925 |
| Decision Tree       |     0.8099 |     0.6052 |     0.6071 |     0.6062 |     0.7407 |

### Best Initial Model

Based on the actual comparison results:

**Gradient Boosting** achieved the highest F1 Score:

```text
F1 Score = 0.6681
```

It also achieved the highest ROC-AUC:

```text
ROC-AUC = 0.9138
```

Therefore, Gradient Boosting showed the strongest overall performance among the models evaluated in the final comparison.

---

## 🔧 Hyperparameter Tuning

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

The tuning process demonstrates how model hyperparameters can be optimized using cross-validation.

---

## 🔍 Confusion Matrix

A confusion matrix was generated for the Random Forest model to analyze:

* True Negatives
* False Positives
* False Negatives
* True Positives

The target labels were displayed as:

```text
<=50K
>50K
```

A classification report was also generated to provide class-level precision, recall, and F1 Score.

---

## 🧪 Prediction on New Data

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

## 💾 Output Files

The project generates the following files:

### `adult_cleaned.csv`

Contains the cleaned version of the dataset after data preparation.

### `model_comparison.csv`

Contains the performance metrics of the trained models.

### `income_prediction.ipynb`

Contains the complete implementation, analysis, visualizations, model training, evaluation, tuning, and prediction workflow.

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Adult-Income-Prediction.git
```

### 2. Open the Project

```bash
cd Adult-Income-Prediction
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Run the Notebook

Using Jupyter:

```bash
jupyter notebook
```

Then open:

```text
income_prediction.ipynb
```

You can also open and run the notebook directly using **VS Code** with the Jupyter extension.

---

## 📌 Key Learning Outcomes

Through this project, the following Machine Learning concepts are demonstrated:

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

---

## ⚠️ Limitations

* The model is trained on the available Adult Income dataset and may not generalize to different populations.
* Income prediction is influenced by many factors that may not be represented in the dataset.
* The project is intended for educational and Machine Learning demonstration purposes.
* The model should not be used as the sole basis for real-world financial or employment decisions.

---

## 🚀 Future Improvements

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

---

## 📜 Conclusion

This project demonstrates a complete **end-to-end Machine Learning classification workflow** for predicting annual income categories.

The workflow starts with raw data and progresses through cleaning, exploratory analysis, preprocessing, model development, evaluation, comparison, hyperparameter tuning, and prediction on unseen data.

Among the evaluated models, **Gradient Boosting achieved the highest F1 Score of 0.6681 and ROC-AUC of 0.9138** in the recorded final comparison results.

Overall, the project provides practical experience in building and evaluating Machine Learning models for structured/tabular data.

---

## 👨‍💻 Author

**SuryaPrakash Balusu**

Machine Learning Project
Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn
