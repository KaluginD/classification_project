import logging

import typer

from src.dataset import TicketsDataset
from src.metrics import metrics_pretty_print
from src.model import ContactReasonPredictionModel

DEFAULT_DATA_PATH = "data/classification_dataset_filtered"
DEFAULT_MODEL_PATH = "data/models/model.joblib"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model training and evaluation")


def main(
    data_path: str = DEFAULT_DATA_PATH,
    path_to_save: str = DEFAULT_MODEL_PATH,
):
    dataset = TicketsDataset(data_path)
    model = ContactReasonPredictionModel()
    aggregated_test_metrics, _ = model.train(dataset)
    logger.info(
        "Training results aggregated per contact_reason, binary classification:\n"
        + metrics_pretty_print(aggregated_test_metrics)
    )
    aggregated_val_metrics, account_results = model.validate(dataset)
    logger.info(
        "Validation results aggregated per acoount_id, multiclass classification:\n"
        + metrics_pretty_print(aggregated_val_metrics)
    )
    logger.info(
        "Validation results per account, multiclass classification:\n"
        + metrics_pretty_print(account_results)
    )

    if path_to_save:
        model.save(path_to_save)


if __name__ == "__main__":
    typer.run(main)
