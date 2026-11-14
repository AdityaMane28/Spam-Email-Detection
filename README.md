# 📧 Spam Email Detection using Machine Learning

A simple and beginner-friendly machine learning project that classifies messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** techniques and a **Multinomial Naive Bayes** classifier.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Example Prediction](#example-prediction)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Spam messages waste time, clutter inboxes, and can be used for phishing and fraud. This project builds a text classifier that learns from labelled messages and predicts whether a new message is spam or not.

The complete workflow, from loading data to predicting a brand-new email, is implemented in a single Jupyter Notebook.

## Features

- Loads and explores a labelled spam/ham dataset
- Converts text into numerical features using **Bag-of-Words (CountVectorizer)**
- Trains a **Multinomial Naive Bayes** model
- Evaluates using **accuracy, confusion matrix, and classification report**
- Predicts whether a custom email/message is spam

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Data Handling | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Environment | Jupyter Notebook |

## Project Structure

```
Spam-Email-Detection/
│
├── Spam_Email_Detection.ipynb   # Main notebook (full workflow)
├── spam1.csv                    # Dataset (add this file yourself)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Dataset

The notebook expects a CSV file named `spam1.csv` (encoding: `latin-1`) with two columns:

| Column | Description |
|--------|-------------|
| `label` | `ham` (not spam) or `spam` |
| `message` | The text of the email/SMS |

A commonly used dataset in this format is the **SMS Spam Collection Dataset** (available on Kaggle / UCI Machine Learning Repository). Place the CSV in the same folder as the notebook.

## How It Works

1. **Load data** – read `spam1.csv` with Pandas
2. **Explore data** – inspect shape, types, and unique values
3. **Preprocess** – rename columns and map labels (`ham → 0`, `spam → 1`)
4. **Split data** – 80% training / 20% testing (`random_state=42`)
5. **Vectorize text** – convert messages to word-count vectors with `CountVectorizer`
6. **Train model** – fit `MultinomialNB` on the training vectors
7. **Evaluate** – accuracy, confusion matrix, precision/recall/F1
8. **Predict** – classify a new, unseen message

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Spam-Email-Detection.git
cd Spam-Email-Detection

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

1. Make sure `spam1.csv` is in the project folder.
2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `Spam_Email_Detection.ipynb` and run all cells (**Kernel → Restart & Run All**).

## Example Prediction

```python
new_email = ["Congratulations! You won a free iPhone. Click now!"]
new_email_count = vectorizer.transform(new_email)
prediction = model.predict(new_email_count)

print("Spam Mail" if prediction[0] == 1 else "Not Spam Mail")
```

**Output:**
```
Spam Mail
```

## Evaluation Metrics

The model is evaluated on the 20% held-out test set using:

- **Accuracy** – overall share of correct predictions
- **Confusion Matrix** – true/false positives and negatives
- **Classification Report** – precision, recall, and F1-score for each class

> 💡 Add your own results here after running the notebook, e.g. `Accuracy: 98%`.

## Future Improvements

- Use **TF-IDF** instead of plain word counts
- Add text cleaning (lowercasing, stop-word removal, stemming/lemmatization)
- Compare other models (Logistic Regression, SVM, Random Forest)
- Handle class imbalance (spam is usually the minority class)
- Wrap the model in a **Scikit-learn Pipeline** and save it with `joblib`
- Build a web app with **Streamlit** or **Flask** for live predictions

## Contributing

Contributions are welcome! Fork the repo, create a feature branch, and open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

⭐ If you found this project helpful, consider giving it a star!
