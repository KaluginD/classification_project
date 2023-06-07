import numpy as np
from tqdm import tqdm
from joblib import dump, load
import logging

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.dataset import TicketsDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Contact Reason Prediction Model")

TEST_PART = 1.0 / 11.0


class ContactReasonPredictionModel:
    def __init__(self, base_model=svm.SVC, model_args=[], model_kwargs={}):
        self.base_model = base_model
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.models = []
        self.target_classification_reports = []

    def train(self, dataset: TicketsDataset, random_state: int = None):
        self.targets = dataset.get_train_targets()

        logger.info(f"Training model for {len(self.targets)} targets...")

        for target in tqdm(self.targets.values()):
            X_target, y_target = dataset.get_training_data_for_target(target)
            X_train, X_test, y_train, y_test = train_test_split(
                X_target,
                y_target,
                test_size=TEST_PART,
                stratify=y_target,
                random_state=random_state,
            )
            target_model = self.base_model(*self.model_args, **self.model_kwargs)
            target_model.fit(X_train, y_train)

            self.models.append(target_model)

            y_pred = target_model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.target_classification_reports.append(report)
        return self.target_classification_reports

    def validate(self, dataset: TicketsDataset):
        validation_accounts = dataset.get_validation_accounts()

        logger.info(f"Validating model for {len(validation_accounts)} accounts...")

        reports = {}
        for account in tqdm(validation_accounts):
            (
                X_val,
                y_val,
                account_targets,
                targets_names,
            ) = dataset.get_validation_data_for_account(account)
            y_pred = np.stack(
                [self.models[target].predict(X_val) for target in account_targets]
            ).T

            num_targets = len(account_targets) + (len(account_targets) == 1)
            account_report = classification_report(
                y_val,
                y_pred,
                labels=range(num_targets),
                target_names=targets_names,
                output_dict=True,
                zero_division=0,
            )
            reports[account] = account_report
        return reports

    def save(self, path: str):
        dump((self.models, self.targets), path)

    def load(self, path: str):
        self.models, self.targets = load(path)
