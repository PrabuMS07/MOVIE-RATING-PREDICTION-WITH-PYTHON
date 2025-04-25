
# 🎬 IMDb Movie Rating Prediction with Python (Jupyter Notebook) ⭐

This project predicts the IMDb rating for movies using data from the included "IMDb Movies India.csv" dataset. The analysis and modeling are performed within a Jupyter Notebook (`Movie Rating Prediction with Python.ipynb`) using Python and libraries like Pandas, Scikit-learn, Matplotlib, and Seaborn.

## 📝 Project Overview

The Jupyter Notebook (`Movie Rating Prediction with Python.ipynb`) demonstrates a workflow for predicting movie ratings:

1.  **Data Loading & Exploration:** Loads the dataset (`IMDb Movies India.csv` with ISO-8859-1 encoding) and performs initial exploration using `info()`, `describe()`, and checks for missing values.
2.  **Data Preprocessing:**
    *   Handles missing values by dropping specific columns with high null counts (`Duration`, `Year`, `Votes`, `Director`, `Actor 1`, `Actor 2`, `Actor 3`) and then dropping rows with any remaining missing values.
    *   Processes the `Genre` column by splitting comma-separated genres and exploding them into separate rows for analysis and feature creation.
3.  **Feature Engineering:** Uses `TfidfVectorizer` on the processed `Genre` column to create numerical features representing genre information.
4.  **Model Training:**
    *   Splits the data into training and testing sets.
    *   Trains a `LinearRegression` model using the TF-IDF genre features to predict the `Rating`.
5.  **Model Evaluation:** Evaluates the model's performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) score.
6.  **Visualization:** Includes visualizations like:
    *   Histograms and distribution plots for the `Rating`.
    *   Bar plots showing the top genres (based on the processed data).
    *   *(Note: Visualizations for top directors/actors are present in the code but might rely on columns dropped earlier in the typical notebook flow).*
7.  **Prediction Function:** Provides a function (`predict_movie_rating`) that takes a movie name as input, retrieves its genre(s), transforms them using the trained TF-IDF vectorizer, and predicts the rating using the trained Linear Regression model.

## 💾 Dataset

*   **File:** `IMDb Movies India.csv` (Included in the repository)
*   **Source:** Likely sourced from Kaggle or similar platforms.
*   **Encoding:** Loaded using `encoding='ISO-8859-1'`.
*   **Content:** Contains information about Indian movies, including Name, Year, Duration, Genre, Rating, Votes, Director, and Actors.

## ✨ Features & Target

*   **Input Features (Derived):** TF-IDF vectors generated from the movie `Genre` column (after cleaning and processing).
*   **Target Variable:** `Rating` (Numerical IMDb rating).

## ⚙️ Technologies & Libraries

*   Python 3.x
*   Jupyter Notebook
*   Pandas
*   NumPy
*   Scikit-learn (`train_test_split`, `LinearRegression`, `TfidfVectorizer`, `mean_absolute_error`, `mean_squared_error`, `r2_score`)
*   Matplotlib
*   Seaborn

## 🛠️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PrabuMS07/MOVIE-RATING-PREDICTION-WITH-PYTHON.git
    cd MOVIE-RATING-PREDICTION-WITH-PYTHON
    ```
2.  **Set up a Python environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install required libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab notebook
    ```
    

## ▶️ Usage

1.  Ensure you have completed the Setup steps and activated your virtual environment.
2.  Start Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    ```
    or
    ```bash
    jupyter notebook
    ```
3.  Your browser should open the Jupyter interface. Navigate to and open the `Movie Rating Prediction with Python.ipynb` file.
4.  Run the cells in the notebook sequentially (e.g., using `Shift + Enter` or the "Run" button) to execute the code, see the analysis, visualizations, model training, evaluation, and predictions.

## 📊 Evaluation Metrics

The model's performance is evaluated using:

*   **Mean Absolute Error (MAE):** Average absolute difference between actual and predicted ratings.
*   **Mean Squared Error (MSE):** Average squared difference between actual and predicted ratings.
*   **R-squared (R2) Score:** Proportion of variance in the rating explained by the genre features.

## 💡 Potential Considerations

*   **Feature Scope:** The current model primarily relies on TF-IDF vectors derived from the `Genre`. The notebook initially drops Director and Actor columns due to missing values, limiting the features used for the final prediction model compared to some other approaches.
*   **Missing Value Strategy:** Dropping columns and rows with missing data simplifies the process but significantly reduces the dataset size and potentially removes valuable information. Alternative imputation strategies could be explored.
*   **Model Complexity:** Linear Regression is used. Exploring more complex models (e.g., Random Forest, Gradient Boosting) might yield different results, especially if more features were incorporated.

---
