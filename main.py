import typer

from src.dataset import TicketsDataset
from src.model import ContactReasonPredictionModel
from src.metrics import metrics_pretty_print

DEFAULT_DATA_PATH = "data/classification_dataset_filtered"


def main(
    data_path: str = DEFAULT_DATA_PATH,
):
    dataset = TicketsDataset(data_path)
    model = ContactReasonPredictionModel()
    aggregated_test_metrics, targets_results = model.train(dataset)
    print(
        "Training results aggregated per contact_reason, binary classification:\n"
        + metrics_pretty_print(aggregated_test_metrics)
    )
    aggregated_val_metrics, account_results = model.validate(dataset)
    print(
        "Validation results aggregated per acoount_id, multiclass classification:\n"
        + metrics_pretty_print(aggregated_val_metrics)
    )
    print(
        "Validation results per account, multiclass classification:\n"
        + {metrics_pretty_print(account_results)}
    )


if __name__ == "__main__":
    typer.run(main)
