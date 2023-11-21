# Ticket Classification

This repository contains a solution for automating the classification of merchant customer tickets based on their contact reasons. The goal is to develop a machine learning solution that can handle about 250K tickets daily with an expected response time of around 200ms.

## Proposed Solution

The proposed solution is designed to address the task of classifying merchant customer tickets based on their contact reasons. Here are the key ideas and details:

1. Minimum Labeled Tickets: To train the machine learning models, a minimum number of labeled tickets for each contact reason is required. The proposed threshold is 120 tickets, with 100 for training, 10 for testing, and 10 for validation.

2. Sentence Length Limit: To simplify the process, the length of tickets is limited to five sentences. This reduces the dataset to approximately 40,000 tickets with 90,000 sentences.

3. Sentence Embeddings: The provided dataset includes sentence embeddings generated by the all-MiniLM-L6-v2 model. For simplicity and efficiency, the average embedding of the complete email is used as the email embedding for classification.

4. Classic ML Models: Several classic machine learning models are trained independently to predict each contact reason based on the sentence embeddings. Models such as SVM Classifier, Linear Regression, and Random Forest are suitable for training with the available dataset size.

5. Negative Sample Sampling: Negative samples are sampled for each contact reason to train the decision-making models effectively.

6. Model Selection: A grid search scrpit is provided using the provided dataset and evaluation metrics to select the best model for each contact reason. TODO: this experiment was prepared but not conducted.

The proposed approach makes certain assumptions and simplifications to provide a simple, fast, and scalable solution. Although there is room for improvement, this solution demonstrates the feasibility of automating ticket classification.

## Project Structure

- `src/`: Directory containing the source code files.
- `results/`: Directory to store the performance results.
- `data/`: Directory containing the dataset in Parquet format.
- `requirements.txt`: Python package requirements for the project.
- `README.md`: Solution documentation and instructions to reproduce the results.

## Codebase details

For detailed information about the data journey and preprocessing steps, refer to the `jump_start_your_journey.ipynb` notebook. The summary of the preprocessing steps can be found in the `src/preprocessing.py` script.

The `src/dataset.py` file provides a dataset structure that allows training models per contact reason and evaluating them per merchant account. The model logic is implemented in the `src/model.py` file.

The `src/metrics.py` file aggregates classification metrics. It's important to note that the current implementation uses a simple aggregation method, and it is recommended to explore better aggregation techniques for improved results.

The `src/model_selection.py` script performs a grid search for the best model based on the provided dataset.

To run the basic training script, use `main.py`. Additionally, a simple web interface based on Flask is available in `app.py`.

All important files have detailed docstrings to provide comprehensive documentation. The results of training and validation can be found in the `results/` directory.

## Getting Started

To get started with the project, follow the steps below:

1. Clone the repository:

```
git clone https://github.com/KaluginD/classification_project
cd classification_project
```

2. Run the data processing script:
```
python src/preprocessing.py
```
3. Run the main training script:
```
python main.py
```
4. Run the Flask application:
```
flask --app app run
```
5. Open `http://127.0.0.1:5000/` in your browser and send a request in the form: `http://127.0.0.1:5000/post/<account_id>/<email_sentence_embeddings>`
