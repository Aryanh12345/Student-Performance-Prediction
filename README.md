# 🎓 Student Performance Prediction

A machine-learning project that predicts a student's expected test score from **age, study hours, and attendance percentage**.

## Live application

Deploy this project with Streamlit Community Cloud and use `app.py` as the application entry point.

## What the project does

- Loads and explores student-performance data
- Handles missing values
- Removes duplicate records
- Treats impossible study-hour and attendance values as missing
- Trains and evaluates regression models in the notebook
- Uses a Random Forest regression model for the deployed predictor
- Saves the trained model with Joblib
- Provides a Streamlit web interface for predictions

## Model

The deployed model is a Random Forest Regressor with 300 trees. On the cleaned dataset and a fixed 80/20 test split (`random_state=42`), the current training script produced:

- R²: about **0.72**
- MAE: about **9.11 points**
- RMSE: about **11.68 points**

These are test-set measurements for this dataset, not guarantees for new data.

## Grade mapping

The application converts the predicted score into the grade bands represented by the dataset:

| Score | Grade |
|---:|:---|
| 91–100 | A+ |
| 81–90 | A |
| 66–80 | B |
| 51–65 | C |
| 0–50 | F |

Grade is derived from the predicted score instead of training a classifier using `Test_Score` as an input. This avoids target leakage.

## Project structure

```text
Student-Performance-Prediction/
├── app.py
├── data/
│   └── student_performance.csv
├── models/
│   └── student_score_model.pkl
├── notebook/
│   └── student_performance.ipynb
├── screenshots/
├── src/
│   └── train_model.py
├── .gitignore
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Retrain the model

```bash
python src/train_model.py
```

## Dataset notes

The supplied dataset contains 1,020 rows and includes some missing values, duplicate records, negative study-hour values, and attendance values above 100%. The training script removes duplicate rows and treats impossible numeric values as missing before imputation.
