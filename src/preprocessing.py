import itertools

import numpy as np
import pandas as pd
import typer

from src.ticket_messages import TicketMessage

DEFAULT_DATASET_PATH = "data/classification_dataset"
DEFAULT_PATH_TO_SAVE = "data/classification_dataset_filtered"

CONTACT_REASON_TO_LIST = lambda line: [
    i.strip() for i in line.lower().replace('"', "").split("::")
]
CONTACT_REASON_LIST_TO_STRING = lambda reasons: "::".join(reasons)

CONTACT_REASON_MIN_TICKET_NUMBER = 120
EMAIL_MAX_LEN = 5


def preprocess_dataset(
    dataset_path: str = DEFAULT_DATASET_PATH,
    path_to_save: str = DEFAULT_PATH_TO_SAVE,
):
    """
    Preporcess dataset:
        - removes samples with empty email_sentence_embeddings field
        - preprocess contact reasons:
            * casts to lower case
            * removes double brackets
            * splits by '::' separator
            * removes whitespaces in the beggining and in the end
        - removes contact reasons with less then CONTACT_REASON_MIN_TICKET_NUMBER tickets
        - adds email length column
        - removes samples with emails longer then EMAIL_MAX_LEN sentences

    Parameters:
        - dataset_path (str) : path to inital dataset file
        - path_to_save (str) : path to save processed dataset

    Returns:
        None

    How to run:
    > python src/preprocessing.py
    """
    dataset = pd.read_parquet(dataset_path)
    dataset_not_na_email = dataset[~dataset["email_sentence_embeddings"].isna()]

    dataset_reasons = dataset_not_na_email[["account_id", "ticket_id"]]
    dataset_reasons["contact_reason"] = dataset_not_na_email["contact_reason"].apply(
        CONTACT_REASON_TO_LIST
    )

    rows_per_reason = dataset_reasons.loc[
        dataset_reasons.index.repeat(dataset_reasons["contact_reason"].apply(len))
    ].drop("ticket_num_reasons", axis=1)
    rows_per_reason["contact_reason"] = list(
        itertools.chain.from_iterable(dataset_reasons["contact_reason"].values)
    )
    reason_ticket_nums = (
        rows_per_reason[["ticket_id", "contact_reason"]]
        .groupby("contact_reason")
        .count()
        .sort_values("ticket_id", ascending=False)
        .rename(columns={"ticket_id": "num_tickets"})
    )
    contact_reasons_to_keep = set(
        reason_ticket_nums[
            reason_ticket_nums["num_tickets"] >= CONTACT_REASON_MIN_TICKET_NUMBER
        ].index
    )

    dataset_reasons["contact_reason"] = dataset_reasons["contact_reason"].apply(
        lambda reasons: list(filter(lambda i: i in contact_reasons_to_keep, reasons))
    )
    dataset_reasons = dataset_reasons[dataset_reasons["contact_reason"].apply(len) > 0]
    dataset_filtered_reasons = dataset_not_na_email.merge(
        dataset_reasons[["ticket_id", "contact_reason"]].rename(
            columns={"contact_reason": "contact_reason_list"}
        ),
        on="ticket_id",
    )

    filtered_ticket_message_list = TicketMessage.from_dataframe(
        dataframe=dataset_filtered_reasons
    )
    dataset_filtered_reasons["email_len"] = [
        len(i.email_sentence_embeddings) for i in filtered_ticket_message_list
    ]
    dataset_filtered_reasons_and_emails = dataset_filtered_reasons[
        dataset_filtered_reasons["email_len"] <= EMAIL_MAX_LEN
    ]
    dataset_filtered_reasons_and_emails[
        "contact_reason"
    ] = dataset_filtered_reasons_and_emails["contact_reason_list"].apply(
        CONTACT_REASON_LIST_TO_STRING
    )
    dataset_filtered_reasons_and_emails.drop(
        "contact_reason_list", axis=1, inplace=True
    )
    dataset_filtered_reasons_and_emails.to_parquet(path_to_save, index=False)


if __name__ == "__main__":
    typer.run(preprocess_dataset)
