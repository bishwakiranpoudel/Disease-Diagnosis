# Disease Prediction API

This project uses a machine learning model to predict diseases based on input symptoms. The API provides an interface to interact with the model and retrieve detailed information about the predicted diseases, including description, precautions, medications, diet plans, and workouts.

## Features

- Predict diseases based on a list of symptoms.
- Retrieve detailed information about diseases, including precautions, medications, diet, and workout recommendations.
- Fast and easy-to-use API built with FastAPI.

---

## 1. Dataset Preparation

The training dataset consists of symptoms mapped to various diseases. Each row in the dataset represents a set of symptoms (as binary indicators) and their corresponding disease.

### Example Structure:

| Symptom 1 | Symptom 2 | ... | Symptom N | Disease          |
| --------- | --------- | --- | --------- | ---------------- |
| 1         | 0         | ... | 0         | Fungal Infection |
| 0         | 1         | ... | 1         | GERD             |

### Steps:

1. **Preprocessing:**

   - Converted symptoms into numerical binary values (1 for presence, 0 for absence).
   - Encoded disease labels using integer encoding.

2. **Splitting:**

   - Split the dataset into training and testing sets (80% training, 20% testing).

3. **Feature Selection:**
   - Mapped symptoms to a dictionary for easy indexing during prediction.

---

## 2. Model Training

The Random Forest Classifier was used as the primary model for prediction.

### Steps:

1. Imported the required libraries:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score
   ```
2. Loaded the preprocessed dataset.
3. Split the data into features (`X`) and target labels (`y`).
4. Split the dataset into training and test sets:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
5. Trained the model using:
   ```python
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   ```
6. Evaluated the model on the test set:
   ```python
   y_pred = model.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))
   ```
7. Saved the trained model using Pickle:
   ```python
   import pickle
   with open('randomforest.pkl', 'wb') as file:
       pickle.dump(model, file)
   ```

---

## 3. How to Run the Project

### Prerequisites

1. Python 3.7+
2. Required Python libraries:
   - `fastapi`
   - `uvicorn`
   - `pandas`
   - `numpy`
   - `scikit-learn`

Install dependencies using:

```bash
pip install -r requirements.txt
```

### File Structure

```
.
|-- api.py                    # FastAPI application
|-- randomforest.pkl          # Trained Random Forest model
|-- All_Diseases_Details.csv  # Disease details dataset
|-- requirements.txt          # Python dependencies
```

### Steps to Run

1. **Start the API Server:**
   Run the following command in the terminal:

   ```bash
   uvicorn api:app --reload
   ```

2. **Access the API:**
   Open your browser or API testing tool and navigate to:

   - Root Endpoint: `http://127.0.0.1:8000`
   - Swagger UI Documentation: `http://127.0.0.1:8000/docs`

3. **Test the Prediction Endpoint:**
   Use the `/predict_disease/` endpoint with a JSON payload like:
   ```json
   {
     "symptoms": ["itching", "skin_rash"]
   }
   ```

---

## 4. How It Works

- **Input Symptoms:** Users provide a list of symptoms.
- **Prediction:**
  - Symptoms are converted into a binary vector.
  - The trained Random Forest model predicts the disease index.
- **Details Retrieval:**
  - The predicted disease is mapped to detailed information from `All_Diseases_Details.csv`.

---

## 5. Example Response

### Request:

```json
{
  "symptoms": ["itching", "skin_rash"]
}
```

### Response:

```json
{
  "predicted_disease": "Fungal infection",
  "details": {
    "description": "A common skin infection caused by fungi.",
    "precautions": ["Keep the affected area dry.", "Use antifungal creams."],
    "medications": ["Clotrimazole", "Miconazole"],
    "diet": ["Avoid sugary foods.", "Include probiotics."],
    "workout": ["Yoga", "Stretching"]
  }
}
```

---

## 6. Troubleshooting

- **File Not Found Errors:** Ensure `randomforest.pkl` and `All_Diseases_Details.csv` are in the correct paths.
- **Dependency Issues:** Ensure all required libraries are installed using `pip install -r requirements.txt`.
- **Model Version Mismatch:** Use the same `scikit-learn` version for training and running the model.

---

## 7. Future Improvements

- Support for additional symptoms and diseases.
- Incorporation of advanced models (e.g., Gradient Boosting, Neural Networks).
- Deployment to cloud platforms (e.g., AWS, Heroku).

---

## Author

Bishwa Kiran Poudel - Disease Prediction API
