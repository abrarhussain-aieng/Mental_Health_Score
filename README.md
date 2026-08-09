# 🧠 Mental Health Score Prediction

A complete end-to-end **Machine Learning regression project** that predicts a student's mental health score using academic, lifestyle, social-media usage, and stress-related information.

The project covers the complete machine learning workflow:

**Data Analysis → Data Cleaning → Feature Engineering → Preprocessing → Model Training → Model Evaluation → Model Serialization → FastAPI API → Deployment**

## 🚀 Live Demo

**Deployed API:**
[Open the Mental Health Score Prediction API](https://mental-health-score-m1nf.onrender.com?utm_source=chatgpt.com)

### API Status

The root endpoint returns an API status message confirming that the service is running.

---

## 📌 Project Overview

Mental health can be influenced by several lifestyle, academic, and social-media-related factors. This project uses machine learning to estimate a student's **Mental Health Score** based on multiple input features.

The final model is integrated into a **FastAPI REST API**, allowing users or applications to send student information and receive a predicted mental health score.

The API loads the trained `Mental_Health_Model.pkl` model and exposes a `/predict` endpoint for predictions.

> **Note:** This project is intended for educational and predictive modeling purposes. The prediction should not be considered a medical diagnosis or professional mental-health assessment.

---

# 🎯 Objectives

The main objectives of this project are:

* Build a machine learning regression model for mental health score prediction.
* Analyze relationships between lifestyle, academic, and social-media factors.
* Clean and preprocess the dataset.
* Perform exploratory data analysis (EDA).
* Engineer useful features.
* Compare multiple regression models.
* Build a complete preprocessing and modeling pipeline.
* Serialize the final model using Joblib.
* Develop a REST API using FastAPI.
* Add input validation using Pydantic.
* Deploy the API so it can be accessed online.

---

# 📊 Dataset

The project uses a dataset containing information related to **student social media usage and mental health impact**.

The dataset contains:

* **5,000 records**
* **13 original columns**

The target variable is:

```text
Mental_Health_Score
```

### Dataset Features

| Feature                   | Description                                |
| ------------------------- | ------------------------------------------ |
| `Age`                     | Age of the student                         |
| `Gender`                  | Gender of the student                      |
| `Country`                 | Student's country                          |
| `Academic_Level`          | Academic level                             |
| `Most_Used_Platform`      | Most frequently used social-media platform |
| `Purpose_Of_Use`          | Main purpose of social-media usage         |
| `Avg_Daily_Usage_Hours`   | Average daily social-media usage           |
| `Daily_Unlocks`           | Number of daily device/app unlocks         |
| `Study_Hours`             | Daily study hours                          |
| `Physical_Activity_Hours` | Physical activity hours                    |
| `Sleep_Hours_Per_Night`   | Average sleep hours                        |
| `Stress_Level`            | Reported stress level                      |
| `Mental_Health_Score`     | Target variable                            |

An additional engineered feature called `Grouped_country` is created during preprocessing.

---

# 🔎 Exploratory Data Analysis

The notebook performs several EDA steps to understand the dataset and relationships between variables.

### Analysis includes:

* Dataset structure and information
* Statistical summary
* Missing-value analysis
* Duplicate-value analysis
* Target distribution
* Correlation analysis
* Stress Level vs Mental Health Score
* Daily social-media usage vs Mental Health Score
* Sleep hours vs Mental Health Score
* Most-used social-media platforms
* Numerical outlier analysis
* Skewness analysis

### Example relationships investigated

The project specifically analyzes:

```text
Stress Level → Mental Health Score
```

```text
Daily Usage Hours → Mental Health Score
```

```text
Sleep Hours → Mental Health Score
```

These analyses help understand which variables may contain useful predictive information.

---

# 🧹 Data Cleaning

The preprocessing workflow includes:

### Removing duplicate records

Duplicate rows are removed from the dataset.

### Handling physical activity values

Negative values in `Physical_Activity_Hours` are clipped to zero.

### Missing values

The dataset analysis checks all columns for missing values before modeling.

---

# ⚙️ Feature Engineering

The project creates a grouped country feature to reduce the number of country categories.

The top 10 countries are retained while other countries are grouped into:

```text
Other
```

The API uses the same country grouping approach when receiving new prediction requests.

### Top country categories

```text
Other
India
USA
Canada
Australia
UK
Germany
Mexico
Turkey
France
```

---

# 🧪 Train-Test Split

The dataset is divided into training and testing sets using:

```text
Training data: 70%
Testing data: 30%
```

with:

```python
random_state=42
```

This provides a reproducible train-test split.

---

# 🔧 Feature Preprocessing

A `ColumnTransformer` is used to apply different preprocessing techniques to different feature types.

## Numerical Features

Standard scaling is applied to numerical features such as:

* Age
* Daily Unlocks
* Average Daily Usage Hours
* Sleep Hours
* Physical Activity Hours

## Skewed Features

`Study_Hours` is treated as a skewed feature.

The pipeline applies:

```python
np.log1p()
```

followed by:

```python
StandardScaler()
```

## Ordinal Feature

`Stress_Level` has a natural order:

```text
Low → Medium → High → Very High
```

Therefore, it is processed using `OrdinalEncoder`.

## Categorical Features

The following categorical features are processed using `OneHotEncoder`:

* Most Used Platform
* Grouped Country
* Gender
* Academic Level
* Purpose of Use

Unknown categories are handled using:

```python
handle_unknown="ignore"
```

This makes the preprocessing pipeline more robust when new categorical values are encountered.

---

# 🤖 Machine Learning Models

Two regression models were evaluated.

## 1. Linear Regression

Linear Regression was used as the baseline model.

### Test Performance

| Metric |       Result |
| ------ | -----------: |
| R²     | **0.744675** |
| MAE    | **0.536427** |
| RMSE   | **0.669662** |

---

## 2. Random Forest Regression

Random Forest Regression was evaluated as a more flexible nonlinear model.

### Test Performance

| Metric |       Result |
| ------ | -----------: |
| R²     | **0.891084** |
| MAE    | **0.325262** |
| RMSE   | **0.437375** |

### Training Performance

```text
Training R²: 0.982627
Testing R²:  0.891084
```

Based on the notebook's evaluation, **Random Forest Regression performed better than Linear Regression** on the test set and was selected as the final model.

---

# 🏆 Model Comparison

| Model             |  Training R² |   Testing R² |          MAE |         RMSE |
| ----------------- | -----------: | -----------: | -----------: | -----------: |
| Linear Regression |     0.731899 |     0.744675 |     0.536427 |     0.669662 |
| **Random Forest** | **0.982627** | **0.891084** | **0.325262** | **0.437375** |

### Final Model

```text
Random Forest Regressor
```

The trained Random Forest pipeline was saved using Joblib as:

```text
Mental_Health_Model.pkl
```

---

# 💾 Model Serialization

The final trained pipeline is saved using:

```python
import joblib

joblib.dump(rf_pipeline, "Mental_Health_Model.pkl")
```

The saved file contains the trained machine learning pipeline used by the API.

---

# 🚀 FastAPI

The trained model is served using **FastAPI**.

The API loads the saved model:

```python
model = joblib.load("Mental_Health_Model.pkl")
```

and creates a FastAPI application.

The project also enables CORS so that external frontend applications can communicate with the API.

---

# 📡 API Endpoints

## GET `/`

Checks whether the API is running.

### Example response

```json
{
  "message": "Welcome to the Mental Health Score Prediction API",
  "status": "API is running"
}
```

The root endpoint is implemented directly in the FastAPI application.

---

## POST `/predict`

This endpoint accepts student information and returns the predicted mental health score.

### Request fields

| Field                     | Type    | Validation / Values                           |
| ------------------------- | ------- | --------------------------------------------- |
| `age`                     | Integer | 10–100                                        |
| `gender`                  | String  | Male / Female                                 |
| `country`                 | String  | Country name                                  |
| `academic_level`          | String  | Undergraduate / Graduate / High School        |
| `most_used_platform`      | String  | Supported social-media platforms              |
| `purpose_of_use`          | String  | Networking / Education / Entertainment / News |
| `avg_daily_usage_hours`   | Float   | 0–24                                          |
| `daily_unlocks`           | Integer | ≥ 0                                           |
| `study_hours`             | Float   | 0–24                                          |
| `physical_activity_hours` | Float   | 0–24                                          |
| `sleep_hours_per_night`   | Float   | 0–24                                          |
| `stress_level`            | String  | Low / Medium / High / Very High               |

These validation constraints are implemented using Pydantic's `BaseModel`, `Field`, and `Literal`.

---

# 📝 Example API Request

```json
{
  "age": 22,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 5.5,
  "daily_unlocks": 80,
  "study_hours": 4,
  "physical_activity_hours": 1,
  "sleep_hours_per_night": 7,
  "stress_level": "Medium"
}
```

---

# 📤 Example API Response

```json
{
  "predicted_mental_health_score": 6.77
}
```

The API rounds the returned prediction to two decimal places.

---

# 🔄 End-to-End Architecture

```text
                    ┌──────────────────────┐
                    │      User Input      │
                    │  Student Information  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI         │
                    │   POST /predict      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Pydantic Validation│
                    │   Input Validation   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ Grouped Country      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Preprocessing Pipeline│
                    │ Scaling + Encoding   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Random Forest Model  │
                    │   Regression Model   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Mental Health Score  │
                    │      Prediction      │
                    └──────────────────────┘
```

---

# 📁 Project Structure

A recommended GitHub repository structure is:

```text
Mental-Health-Score-Prediction/
│
├── main.py
├── Mental_Health_Model.pkl
├── Predicting_Mental_Health_Score.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File                                   | Purpose                                               |
| -------------------------------------- | ----------------------------------------------------- |
| `main.py`                              | FastAPI application and prediction endpoint           |
| `Mental_Health_Model.pkl`              | Serialized trained machine learning pipeline          |
| `Predicting_Mental_Health_Score.ipynb` | Data analysis, preprocessing, training and evaluation |
| `requirements.txt`                     | Python dependencies                                   |
| `README.md`                            | Project documentation                                 |
| `.gitignore`                           | Files excluded from Git                               |

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Science

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Random Forest Regression
* Linear Regression
* Pipelines
* ColumnTransformer
* StandardScaler
* OneHotEncoder
* OrdinalEncoder

### Model Persistence

* Joblib

### API Development

* FastAPI
* Pydantic
* Uvicorn

### Deployment

* Render

---

# 💻 Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

```bash
cd YOUR-REPOSITORY
```

Replace `YOUR-USERNAME/YOUR-REPOSITORY` with your actual GitHub repository.

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the API Locally

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

The API should become available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive API documentation can then be accessed through:

```text
http://127.0.0.1:8000/docs
```

The `/docs` interface can be used to test the `/predict` endpoint directly from the browser.

---

# 🌐 Deployment

The API is deployed using **Render**.

### Live Deployment

[https://mental-health-score-m1nf.onrender.com](https://mental-health-score-m1nf.onrender.com?utm_source=chatgpt.com)

The deployed service can be used as the backend API for an external frontend or application.

---

# 🔐 Input Validation

The API uses Pydantic validation to prevent invalid values from reaching the machine learning model.

For example:

```python
age: int = Field(..., ge=10, le=100)
```

ensures that the age is between 10 and 100.

Similarly, numeric lifestyle variables have appropriate ranges, while categorical fields use `Literal` to restrict accepted values.

This provides an additional layer of reliability before prediction.

---

# 🔄 Prediction Pipeline

When a request is sent to `/predict`, the following process takes place:

```text
1. Receive JSON request
        ↓
2. Validate request with Pydantic
        ↓
3. Group country into supported categories
        ↓
4. Create Pandas DataFrame
        ↓
5. Apply trained preprocessing pipeline
        ↓
6. Pass processed features to Random Forest
        ↓
7. Generate prediction
        ↓
8. Round prediction to 2 decimals
        ↓
9. Return JSON response
```

The API constructs the model input using the same feature names used during training.

---

# 📈 Key Results

The final Random Forest model achieved:

```text
Test R²   : 0.891084
MAE       : 0.325262
RMSE      : 0.437375
```

Compared with Linear Regression, Random Forest provided substantially better test performance.

This makes Random Forest the selected model for the deployed prediction API.

---

# ⚠️ Limitations

This project has several limitations that should be considered:

* The model is trained on a specific student/social-media dataset.
* Predictions depend on the quality and distribution of the training data.
* Machine learning predictions do not establish medical or psychological diagnoses.
* The model may not generalize equally well to every population or demographic group.
* The deployed API should be treated as an educational/data-science application rather than a clinical tool.
* The current API enables CORS broadly, which is convenient for development and frontend integration but should be restricted to trusted origins for a production system.

---

# 🔮 Future Improvements

Possible future improvements include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Cross-validation for more robust model evaluation.
* Additional regression algorithms such as Gradient Boosting, XGBoost, or HistGradientBoosting.
* Explainable AI using SHAP.
* Better monitoring of model performance after deployment.
* More comprehensive input validation.
* Restricting CORS to trusted frontend domains.
* Adding automated API tests.
* Adding CI/CD using GitHub Actions.
* Containerizing the application with Docker.
* Adding authentication and rate limiting for a production API.
* Building a dedicated frontend for easier interaction with the prediction service.

---

# 🧪 Model Evaluation Metrics

The project uses three primary regression metrics.

### R² Score

Measures how well the model explains the variance in the target variable.

Higher is generally better.

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

Lower is better.

### Root Mean Squared Error — RMSE

Measures prediction error while giving greater weight to larger errors.

Lower is better.

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Handling categorical variables
* Ordinal encoding
* One-hot encoding
* Feature scaling
* Log transformation
* ColumnTransformer
* Scikit-learn Pipelines
* Regression modeling
* Model comparison
* Model evaluation
* Model serialization
* REST API development
* Pydantic validation
* FastAPI
* CORS
* Model deployment
* GitHub project documentation

---

# 👨‍💻 Author

**Abrar Hussain**

This project was developed as an end-to-end machine learning project demonstrating how a trained regression model can be transformed into a deployable API.

---

# ⭐ If You Found This Project Useful

If you find this project interesting or useful:

* ⭐ Star the repository
* 🍴 Fork the project
* 🐛 Open an issue for bugs or suggestions
* 💡 Submit a pull request with improvements

---

## 📄 License

This project is provided for educational and learning purposes.
