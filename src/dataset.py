import itertools
import json
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing import CONTACT_REASON_TO_LIST
from src.ticket_messages import TicketMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Tickets Dataset")

VALIDATION_PART = 1.0 / 12.0


class TicketsDataset:
    def __init__(self, path, shuffle=False):
        self.dataset = pd.read_parquet(path)
        self.shuffle = shuffle

        self._prepare_dataset()
        self._prepare_ticket_per_reason_dataset_and_encoder()
        self._aggregate_tickets_per_reason()

        self._prepare_validation_data()
        self._prepare_train_data()

    def _prepare_dataset(self):
        logger.info("Processing contact reasons and embeddings...")

        self.ticket_message_list = TicketMessage.from_dataframe(dataframe=self.dataset)
        self.dataset["reasons"] = self.dataset["contact_reason"].apply(
            CONTACT_REASON_TO_LIST
        )
        self.dataset["embeddings"] = [
            np.array(list(i.email_sentence_embeddings.values()))
            for i in self.ticket_message_list
        ]
        self.dataset["mean_emb"] = self.dataset["embeddings"].apply(
            lambda i: i.mean(axis=0)
        )

    def _prepare_ticket_per_reason_dataset_and_encoder(self):
        logger.info(
            "Flattering tickets per contact reason and aggregating reasons per account..."
        )

        self.ticket_per_reason = self.dataset[
            ["account_id", "ticket_id", "reasons", "mean_emb"]
        ].loc[self.dataset.index.repeat(self.dataset["reasons"].apply(len))]
        self.ticket_per_reason["reason_str"] = list(
            itertools.chain.from_iterable(self.dataset["reasons"].values)
        )
        self.ticket_per_reason.drop("reasons", axis=1, inplace=True)

        self.encoder = {
            key: i
            for i, key in enumerate(self.ticket_per_reason["reason_str"].unique())
        }
        self.decoder = {encoded: name for name, encoded in self.encoder.items()}

        self.num_reasons = len(self.encoder)
        self.ticket_per_reason["reason"] = self.ticket_per_reason["reason_str"].apply(
            lambda i: self.encoder[i]
        )
        self.reasons_per_account = (
            self.ticket_per_reason[["account_id", "reason"]]
            .groupby("account_id")
            .agg(lambda i: sorted(list(set(list(i)))))
        )

    def _aggregate_tickets_per_reason(self):
        logger.info("Aggregating tickets per contact reason...")

        self.all_tickets = self.ticket_per_reason["ticket_id"].unique()
        self.all_ticket_ids_per_reason = [
            self.ticket_per_reason[
                self.ticket_per_reason["reason"] == i
            ].ticket_id.unique()
            for i in tqdm(range(self.num_reasons))
        ]

    def _prepare_validation_data(self):
        logger.info("Preparing validation data per account_id...")

        self.ticket_ids_per_reason, self.validation_ticket_ids_per_reason = [], []
        for curr_reason in self.all_ticket_ids_per_reason:
            if self.shuffle:
                np.random.shuffle(curr_reason)
            reason_len = len(curr_reason)
            split_point = int(reason_len * (1 - VALIDATION_PART))
            self.ticket_ids_per_reason.append(curr_reason[:split_point])
            self.validation_ticket_ids_per_reason.append(curr_reason[split_point:])
        self.validation_data_tickets = np.unique(
            np.concatenate(self.validation_ticket_ids_per_reason)
        )
        self.validation_data = self.dataset[
            self.dataset["ticket_id"].isin(self.validation_data_tickets)
        ]

        self.validation_data_accounts = self.validation_data["account_id"].unique()
        self.validation_data_per_account = {
            account_id: self.validation_data[
                self.validation_data["account_id"] == account_id
            ].copy(deep=True)
            for account_id in self.validation_data_accounts
        }
        for account in tqdm(self.validation_data_accounts):
            account_reasons = self.reasons_per_account.loc[account]["reason"]

            self.validation_data_per_account[account][
                "encoded_reasons"
            ] = self.validation_data_per_account[account]["reasons"].apply(
                lambda reasons: sorted([self.encoder[i] for i in reasons])
            )
            self.validation_data_per_account[account][
                "one_hot_reasons"
            ] = self.validation_data_per_account[account]["encoded_reasons"].apply(
                lambda reasons: [
                    1 if reason in reasons else 0 for reason in account_reasons
                ]
            )

    def _prepare_train_data(self):
        logger.info("Preparing training data per contact reason...")

        self.train_test_data = self.dataset[
            ~self.dataset["ticket_id"].isin(self.validation_data_tickets)
        ]
        self.ticket_ids_neg_per_reason = [
            self.train_test_data[
                ~self.train_test_data["ticket_id"].isin(self.ticket_ids_per_reason[i])
            ].ticket_id.values
            for i in tqdm(range(self.num_reasons))
        ]

    def get_train_targets(self):
        return self.encoder, self.decoder

    def get_validation_accounts(self):
        return self.validation_data_accounts

    def get_accounts_targets(self):
        return self.reasons_per_account["reason"].to_dict()

    def get_training_data_for_target(self, target):
        positive_tickets = self.ticket_ids_per_reason[target]
        negative_tickets = self.ticket_ids_neg_per_reason[target][
            : len(positive_tickets) * 4
        ]

        curr_reason_data = self.train_test_data[
            self.train_test_data["ticket_id"].isin(
                np.concatenate((positive_tickets, negative_tickets))
            )
        ][["ticket_id", "mean_emb"]]
        curr_reason_data["target"] = (
            curr_reason_data["ticket_id"].isin(positive_tickets).astype(int)
        )

        X_train = np.stack(curr_reason_data["mean_emb"].values)
        y_train = curr_reason_data["target"].values
        return X_train, y_train

    def get_validation_data_for_account(self, account):
        account = str(account)
        account_targets = self.reasons_per_account.loc[account]["reason"]
        targets_names = [self.decoder[target] for target in account_targets]
        validation_data = self.validation_data_per_account[account].copy(deep=True)
        X_val = np.stack(validation_data["mean_emb"].values)
        y_val = np.stack(validation_data["one_hot_reasons"].values)
        return X_val, y_val, account_targets, targets_names
