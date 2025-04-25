

# 🎬 IMDb Movie Rating Prediction with Python 🐍

This project predicts the IMDb rating for movies listed in the "IMDb Movies India.csv" dataset using features like Genre, Director, and Actors. It employs a Linear Regression model built with Python and Scikit-learn, intended to be run as a standard Python script.

## 📝 Project Description

This Python script (`your_script_name.py` - *rename as needed*) performs the following steps when executed:

1.  **Loads Data:** Reads the `IMDb Movies India.csv` dataset using Pandas (with `ISO-8859-1` encoding).
2.  **Initial Display:** Prints the first few rows (`head()`) and a summary of missing values (`isnull().sum()`) to the terminal.
3.  **Preprocessing:**
    *   Handles missing values by dropping rows with any nulls (`dropna()`). **Caution:** This might significantly reduce dataset size.
    *   Selects specific features: `Genre`, `Director`, `Actor 1`, `Actor 2`, `Actor 3`.
    *   Applies **One-Hot Encoding** to these categorical features using `pd.get_dummies()`.
4.  **Data Splitting:** Splits the encoded data and target variable (`Rating`) into training and testing sets.
5.  **Feature Scaling:** Applies `StandardScaler` to scale the encoded features.
6.  **Model Training:** Trains a `LinearRegression` model on the scaled training data.
7.  **Prediction & Evaluation:**
    *   Makes predictions on the scaled test set.
    *   Calculates and **prints** the Mean Squared Error (MSE) and R-squared (R2) score to the terminal.
8.  **Example Prediction:** Includes an example demonstrating how to prepare data for and predict the rating of a hypothetical new movie, printing the result to the terminal. *(Note: Requires manually constructing the one-hot encoded input for the new movie)*.

## 💾 Dataset

*   **File:** `IMDb Movies India.csv` (Needs to be in the same directory as the script or provide the full path).
*   **Encoding:** The script uses `encoding='ISO-8859-1'`.
*   **Preprocessing Note:** Rows containing *any* missing values are dropped.

## ✨ Features Used

*   **Input Features:** `Genre`, `Director`, `Actor 1`, `Actor 2`, `Actor 3` (transformed via One-Hot Encoding).
*   **Target Variable:** `Rating` (IMDb Rating).

## ⚙️ Technologies & Libraries

*   Python 3.x
*   Pandas
*   NumPy
*   Scikit-learn (`train_test_split`, `OneHotEncoder` *(implicitly via get_dummies)*, `StandardScaler`, `LinearRegression`, `mean_squared_error`, `r2_score`)

## 🛠️ Setup & Installation (using VS Code)

1.  **Clone/Download:** Get the project files (Python script and `IMDb Movies India.csv`).
2.  **Open Folder:** Open the project folder in Visual Studio Code (`File` > `Open Folder...`).
3.  **Python Interpreter:** Ensure you have a Python 3 interpreter selected in VS Code (Check the bottom status bar). If not, install Python and select it.
4.  **Terminal:** Open the integrated terminal in VS Code (`View` > `Terminal` or `Ctrl + \``).
5.  **(Optional but Recommended) Create Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows PowerShell: .\venv\Scripts\Activate.ps1 or cmd: venv\Scripts\activate.bat
    ```
6.  **Install Dependencies:** Create a `requirements.txt` file with the content below:
    ```txt
    pandas
    numpy
    scikit-learn
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```
7.  **Dataset:** Make sure the `IMDb Movies India.csv` file is located in the same folder as your Python script.

## ▶️ Usage (within VS Code)

1.  Ensure you have completed the Setup steps (and activated the virtual environment if you created one).
2.  Open your Python script file (e.g., `movie_rating_predictor.py`) in the VS Code editor.
3.  Run the script using one of these methods:
    *   **From the Terminal:** Type `python your_script_name.py` and press Enter.
    *   **Using VS Code's Run Button:** Click the "Run Python File" button (usually a green triangle) in the top-right corner of the editor (requires the Python extension for VS Code).
    *   **Right-Click:** Right-click within the editor and select "Run Python File in Terminal".

4.  **Output:** Observe the output printed directly **in the VS Code Terminal window**. This will include:
    *   The `head()` of the DataFrame.
    *   The count of null values per column.
    *   The calculated `Mean Squared Error`.
    *   The calculated `R-squared` score.
    *   The `Predicted Rating` for the example new movie.

## 📊 Evaluation Metrics

*   **Mean Squared Error (MSE):** Printed to the terminal. Lower is better.
*   **R-squared (R2) Score:** Printed to the terminal. Closer to 1 indicates a better fit by the model based on the selected features.

## 💡 Potential Future Improvements (Modify the Script)

*   **Add Visualizations:** Use `matplotlib` or `seaborn` to create plots (histograms, scatter plots). You'll need to add `import matplotlib.pyplot as plt` and use `plt.show()` at the end of the script to display plots in separate windows, or `plt.savefig()` to save them as image files. Remember to add `matplotlib` and `seaborn` to your `requirements.txt` if you use them.
*   **Improve Missing Value Handling:** Implement imputation instead of `dropna()`.
*   **Alternative Encoding:** For features like Director/Actors with many unique values, explore Target Encoding or HashingVectorizer instead of One-Hot Encoding to manage dimensionality.
*   **Different Models:** Experiment with other Scikit-learn regressors.
*   **Refine Prediction Input:** Create a function that takes raw movie details (genre, director, actors) and handles the encoding/scaling internally, making prediction easier.

---
