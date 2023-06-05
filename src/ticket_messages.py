from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

@dataclass
class TicketMessage:
    account_id: str
    ticket_id: str
    raw_body: str
    channel: str
    unix_timestamp: float
    contact_reason: str = None
    processed_body: str = None
    email_sentence_embeddings: dict[str, list[float]] = None

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> list[TicketMessage]:
        ticket_messages = []
        for i, row in dataframe.iterrows():
            row_ticket_message = TicketMessage(
                account_id=str(row["account_id"]),
                ticket_id=str(row["ticket_id"]),
                raw_body=str(row["raw_body"]),
                channel=str(row["channel"]),
                unix_timestamp=float(row["unix_timestamp"]),
                contact_reason=str(row["contact_reason"]),
                processed_body=str(row["processed_body"]),
                email_sentence_embeddings=json.loads(row["email_sentence_embeddings"]) if row["email_sentence_embeddings"] else None,
            )
            ticket_messages.append(row_ticket_message)

        return ticket_messages
    
    @classmethod
    def to_dataframe(cls, ticket_messages_list: list[TicketMessage]) -> pd.DataFrame:
        dataframe = pd.DataFrame(
            {
                "account_id": [ticket.account_id for ticket in ticket_messages_list],
                "ticket_id": [ticket.ticket_id for ticket in ticket_messages_list],
                "raw_body": [ticket.raw_body for ticket in ticket_messages_list],
                "channel": [ticket.channel for ticket in ticket_messages_list],
                "unix_timestamp": [ticket.unix_timestamp for ticket in ticket_messages_list],
                "contact_reason": [ticket.contact_reason for ticket in ticket_messages_list],
                "processed_body": [ticket.processed_body for ticket in ticket_messages_list],
                "email_sentence_embeddings": [
                    json.dumps(ticket.email_sentence_embeddings) 
                    if ticket.email_sentence_embeddings else None for ticket in ticket_messages_list],
            }
        )
        return dataframe
    
    
