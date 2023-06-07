import itertools
import typer
import logging

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.dataset import TicketsDataset
from src.model import ContactReasonPredictionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model selection")

SEARCH_SPACE = [
    (
        SVC,
        {
            "C": [0.1, 1.0, 3.0, 5.0],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        },
    ),
    (
        LogisticRegression,
        {
            "penalty": ["l1", "l2", "elasticnet"],
            "C": [0.1, 1.0, 3.0, 5.0],
        },
    ),
    (
        RandomForestClassifier,
        {
            "n_estimators": [50, 100, 200, 400],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 3, 5, 10],
        },
    ),
]

KEY_METRIC = "f1-score"


def select_model(data_path: str, path_to_save: str):
    dataset = TicketsDataset(data_path)

    best_score = 0.0
    best_model = None
    for base_model, params_dict in SEARCH_SPACE:
        params_space = list(itertools.product(*params_dict.values()))
        for param_values in params_space:
            model_kwargs = {
                key: value for key, value in zip(params_dict.keys(), param_values)
            }
            model = ContactReasonPredictionModel(
                base_model=base_model, model_kwargs=model_kwargs
            )
            aggregated_test_metrics, targets_results = model.train(dataset)
            aggregated_val_metrics, account_results = model.validate(dataset)
            if aggregated_val_metrics[KEY_METRIC] > best_score:
                logger.info(
                    f"New best validation {KEY_METRIC} = {aggregated_val_metrics[KEY_METRIC]}\n"
                    + f"Model: {base_model}, params: {model_kwargs}"
                )
                best_model = model

    best_model.save(path_to_save)


if __name__ == "__main__":
    typer.run(select_model)
