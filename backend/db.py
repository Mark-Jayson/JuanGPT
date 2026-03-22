import os
import requests
from datetime import datetime, timezone

def _headers():
    key = os.getenv("SUPABASE_KEY")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

def _url(table: str) -> str:
    return f"{os.getenv('SUPABASE_URL')}/rest/v1/{table}"


def create_conversation(title: str) -> dict:
    resp = requests.post(
        _url("conversations"),
        json={"title": title},
        headers={**_headers(), "Prefer": "return=representation"},
    )
    resp.raise_for_status()
    return resp.json()[0]


def get_conversations() -> list:
    resp = requests.get(
        _url("conversations"),
        params={"select": "*", "order": "updated_at.desc"},
        headers=_headers(),
    )
    resp.raise_for_status()
    return resp.json()


def get_messages(conversation_id: str) -> list:
    resp = requests.get(
        _url("messages"),
        params={
            "select": "*",
            "conversation_id": f"eq.{conversation_id}",
            "order": "created_at.asc",
        },
        headers=_headers(),
    )
    resp.raise_for_status()
    return resp.json()


def add_message(conversation_id: str, role: str, content: str, links: str = None):
    requests.post(
        _url("messages"),
        json={
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "links": links,
        },
        headers=_headers(),
    ).raise_for_status()

    requests.patch(
        _url("conversations"),
        json={"updated_at": datetime.now(timezone.utc).isoformat()},
        params={"id": f"eq.{conversation_id}"},
        headers=_headers(),
    ).raise_for_status()


def delete_conversation(conversation_id: str):
    requests.delete(
        _url("conversations"),
        params={"id": f"eq.{conversation_id}"},
        headers=_headers(),
    ).raise_for_status()
