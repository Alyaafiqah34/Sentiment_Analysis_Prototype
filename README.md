# Ahli 1 - Sentiment Classification Prototype

## Files
- `main.py`: loads and inspects the CSV, splits the data, applies TF-IDF,
  trains Naive Bayes and Logistic Regression, evaluates both, and saves
  confusion matrices and model files.
- `requirements.txt`: Python dependencies.
- `final_preprocessed_dataset.csv`: place the group's final preprocessed CSV here.

## Setup
1. Install Python and open this folder in VS Code.
2. Open Terminal > New Terminal.
3. Install packages:

   `python -m pip install -r requirements.txt`

4. Place the final CSV in this folder and rename it exactly:
   `final_preprocessed_dataset.csv`
5. Run:

   `python main.py`

## Input columns
The script expects:
- `preprocessed_text`: model input
- `sentiment`: target label

The uploaded final dataset also has `id`, `post_keyword`, and `original_text`.
Those columns are not used as model input in this first experiment.

## Main settings
- Test size: 20%
- Training size: 80%
- Random state: 42
- Stratified split
- TF-IDF: unigram + bigram, min_df=2, max_features=10000,
  sublinear_tf=True
- Naive Bayes: MultinomialNB
- Logistic Regression: max_iter=1000, class_weight='balanced'

TF-IDF is placed inside each Scikit-learn Pipeline. Therefore, the vectorizer
is fitted on training text only and then applied to the test text, reducing
test-data leakage.

## Outputs
All results are saved in `outputs/`:
- model_comparison.csv
- classification_reports.json
- confusion_matrix_naive_bayes.png
- confusion_matrix_logistic_regression.png
- predictions_naive_bayes.csv
- predictions_logistic_regression.csv
- naive_bayes_model.joblib
- logistic_regression_model.joblib

## Notes
The CSV contains social media text. Keep it private unless the group has
permission to share it publicly. Do not upload personal or confidential data.
