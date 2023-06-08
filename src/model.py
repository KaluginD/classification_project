import json
import logging
from typing import Dict, List, Optional

import numpy as np
from joblib import dump, load
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataset import TicketsDataset
from src.metrics import aggregate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Contact Reason Prediction Model")

TEST_PART = 1.0 / 11.0


class ContactReasonPredictionModel:
    """
    ContactReasonPrediction Model.

    For each contact reason the model trains a base_model classifier for binary classification problem.
    The model stores information about what contact reasons every account has.
    During inference the model uses base_models for all contact reasons associated with a given account,
    therefore solving multiclass classification problem.

    Methods
    -------
    train(dataset, random_state)
        Stores all contact reasons present in the dataset.
        For each reason trains base_model classifier.
        Evalutes model's performance on test split of data.
        Returns evaluation results for all contact reasons.

    validate(dataset)
        Evaluates model's performace for each account.
        For each account the base_models trained to predict the account's contact reasons are used.

    forward(account_id, email_sentence_embeddings)
        Takes input in the format from inital dataset and generates prediction.
        If no contact reason was detected, returns string "None".

    save(path)
        Saves the model at provided location.

    load(path)
        Loads the model from provided location.
    """

    def __init__(
        self, base_model=svm.SVC, model_args: List = [], model_kwargs: Dict = {}
    ):
        self.base_model = base_model
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.models = []
        self.target_classification_reports = {}

    def train(self, dataset: TicketsDataset, random_state: Optional[int] = None):
        self.targets, self.targets_decoder = dataset.get_train_targets()
        self.account_targets = dataset.get_accounts_targets()

        logger.info(f"Training model for {len(self.targets)} targets...")

        for target_name, target_idx in tqdm(self.targets.items()):
            X_target, y_target = dataset.get_training_data_for_target(target_idx)
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
            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            self.target_classification_reports[target_name] = report
        return aggregate_metrics(self.target_classification_reports)

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
        return aggregate_metrics(reports)

    def forward(self, account_id: int, email_sentence_embeddings: str):
        account_id = str(account_id)
        if isinstance(email_sentence_embeddings, str):
            email_sentence_embeddings = json.loads(email_sentence_embeddings)
        if isinstance(email_sentence_embeddings, dict):
            email_sentence_embeddings = list(email_sentence_embeddings.values())
        if isinstance(email_sentence_embeddings, list):
            email_sentence_embeddings = np.array(email_sentence_embeddings)
        predictions = [
            self.models[target].predict(email_sentence_embeddings)[0]
            for target in self.account_targets[account_id]
        ]
        predicted_reasons = []
        for prediction, reason in zip(predictions, self.account_targets[account_id]):
            if prediction == 1:
                predicted_reasons.append(self.targets_decoder[reason])
        if len(predicted_reasons) == 0:
            reason = "None"
        else:
            reason = "::".join(predicted_reasons)
        return reason

    def save(self, path: str):
        dump(
            (self.models, self.targets, self.targets_decoder, self.account_targets),
            path,
        )

    def load(self, path: str):
        self.models, self.targets, self.targets_decoder, self.account_targets = load(
            path
        )
