import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"

# Models directory
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Project directory:", BASE_DIR)
print("Dataset path:", DATA_PATH)


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n================ DATA ================\n")
print(df)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nShape of data:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nInformation of data:")
df.info()

print("\nData description:")
print(df.describe())


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n================ MISSING VALUES ================\n")

print("Number of null values:")
print(df.isnull().sum())

print("\nMissing value exists?")
print(df.isnull().values.any())


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\n================ CLEANING DATA ================\n")

# Numerical columns
df["Study_Hours"] = df["Study_Hours"].fillna(
    df["Study_Hours"].median()
)

df["Attendance(%)"] = df["Attendance(%)"].fillna(
    df["Attendance(%)"].median()
)

df["Test_Score"] = df["Test_Score"].fillna(
    df["Test_Score"].median()
)

# Categorical columns
df["Grade"] = df["Grade"].fillna(
    df["Grade"].mode()[0]
)

df["Gender"] = df["Gender"].fillna(
    df["Gender"].mode()[0]
)

print("Missing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

# Study hours cannot be negative
df = df[df["Study_Hours"] >= 0]

# Attendance must be between 0 and 100
df = df[
    (df["Attendance(%)"] >= 0)
    & (df["Attendance(%)"] <= 100)
]

# Test score should be between 0 and 100
df = df[
    (df["Test_Score"] >= 0)
    & (df["Test_Score"] <= 100)
]


# ============================================================
# DUPLICATES
# ============================================================

print("\n================ DUPLICATES ================\n")

print("Number of duplicated rows:")
print(df.duplicated().sum())

print("\nDuplicated rows:")
print(df[df.duplicated()])

df = df.drop_duplicates()

print("\nDuplicates after removal:")
print(df.duplicated().sum())


# ============================================================
# DATA TYPES
# ============================================================

print("\n================ DATA TYPES ================\n")
print(df.dtypes)


# ============================================================
# ENCODE CATEGORICAL DATA
# ============================================================

print("\n================ LABEL ENCODING ================\n")

gender_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(
    df["Gender"]
)

print("Encoded Gender:")
print(df["Gender"])


# ============================================================
# GRADE ENCODING
# ============================================================

grade_encoder = LabelEncoder()

df["Grade_Encoded"] = grade_encoder.fit_transform(
    df["Grade"]
)

print("\nGrade encoding:")
print(
    dict(
        zip(
            grade_encoder.classes_,
            grade_encoder.transform(grade_encoder.classes_)
        )
    )
)


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Study_Hours"],
    df["Test_Score"]
)

plt.xlabel("Study Hours")
plt.ylabel("Test Score")
plt.title("Study Hours vs Test Score")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

df["Grade"].value_counts().plot(kind="bar")

plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.title("Grade Distribution")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

plt.scatter(
    df["Attendance(%)"],
    df["Test_Score"]
)

plt.xlabel("Attendance (%)")
plt.ylabel("Test Score")
plt.title("Attendance vs Test Score")
plt.tight_layout()
plt.show()


# ============================================================
# SCORE PREDICTION
# ============================================================

print("\n================ SCORE PREDICTION ================\n")

# Features
X = df[
    [
        "Age",
        "Study_Hours",
        "Attendance(%)"
    ]
]

# Target
y = df["Test_Score"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ============================================================
# STANDARD SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_prediction = linear_model.predict(
    X_test_scaled
)


# ============================================================
# DECISION TREE
# ============================================================

tree_model = DecisionTreeRegressor(
    random_state=42
)

tree_model.fit(
    X_train_scaled,
    y_train
)

tree_prediction = tree_model.predict(
    X_test_scaled
)


# ============================================================
# RANDOM FOREST
# ============================================================

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(
    X_train_scaled,
    y_train
)

forest_prediction = forest_model.predict(
    X_test_scaled
)


# ============================================================
# MODEL COMPARISON
# ============================================================

result = []


result.append(
    [
        "Linear Regression",
        r2_score(y_test, linear_prediction),
        mean_absolute_error(y_test, linear_prediction),
        mean_squared_error(
            y_test,
            linear_prediction
        )
    ]
)


result.append(
    [
        "Decision Tree",
        r2_score(y_test, tree_prediction),
        mean_absolute_error(y_test, tree_prediction),
        mean_squared_error(
            y_test,
            tree_prediction
        )
    ]
)


result.append(
    [
        "Random Forest",
        r2_score(y_test, forest_prediction),
        mean_absolute_error(y_test, forest_prediction),
        mean_squared_error(
            y_test,
            forest_prediction
        )
    ]
)


comparison = pd.DataFrame(
    result,
    columns=[
        "Model",
        "R2",
        "MAE",
        "MSE"
    ]
)

print("\nModel comparison:")
print(comparison.round(3))


# ============================================================
# SAVE BEST SCORE MODEL
# ============================================================

joblib.dump(
    forest_model,
    MODEL_DIR / "student_score_model.pkl"
)

joblib.dump(
    scaler,
    MODEL_DIR / "student_scaler.pkl"
)

print("\nScore model saved successfully.")


# ============================================================
# GRADE PREDICTION
# ============================================================

print("\n================ GRADE CLASSIFICATION ================\n")

# IMPORTANT:
# Test_Score is NOT used as an input feature here.
# This avoids target leakage.

X_grade = df[
    [
        "Age",
        "Study_Hours",
        "Attendance(%)"
    ]
]

y_grade = df["Grade_Encoded"]


X_train_grade, X_test_grade, y_train_grade, y_test_grade = train_test_split(
    X_grade,
    y_grade,
    test_size=0.2,
    random_state=42,
    stratify=y_grade
)


# ============================================================
# GRADE MODEL
# ============================================================

grade_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

grade_model.fit(
    X_train_grade,
    y_train_grade
)

grade_prediction = grade_model.predict(
    X_test_grade
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test_grade,
    grade_prediction
)

precision = precision_score(
    y_test_grade,
    grade_prediction,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test_grade,
    grade_prediction,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test_grade,
    grade_prediction,
    average="weighted",
    zero_division=0
)

matrix = confusion_matrix(
    y_test_grade,
    grade_prediction
)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:")
print(matrix)


# ============================================================
# SAVE GRADE MODEL
# ============================================================

joblib.dump(
    grade_model,
    MODEL_DIR / "student_grade_model.pkl"
)

joblib.dump(
    grade_encoder,
    MODEL_DIR / "grade_encoder.pkl"
)

print("\nGrade model saved successfully.")


# ============================================================
# TEST PREDICTION
# ============================================================

def predict_marks():

    print("\n================ TEST PREDICTION ================\n")

    age = float(
        input("Enter age: ")
    )

    hours = float(
        input("Enter study hours per day: ")
    )

    attendance = float(
        input("Enter attendance percentage: ")
    )

    new_data = pd.DataFrame(
        {
            "Age": [age],
            "Study_Hours": [hours],
            "Attendance(%)": [attendance]
        }
    )

    new_data_scaled = scaler.transform(
        new_data
    )

    prediction = forest_model.predict(
        new_data_scaled
    )

    predicted_score = float(
        prediction[0]
    )

    print(
        f"\nPredicted test score: "
        f"{predicted_score:.2f}"
    )


# Uncomment this if you want terminal input
# predict_marks()


print("\n========================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("========================================")
