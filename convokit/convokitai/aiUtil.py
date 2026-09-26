"""
Shared helpers for the ConvoKit AI classes.

On disk, AI fields live inside the regular ConvoKit metadata so that a dumped
ConvoKit AI corpus remains a valid ConvoKit corpus:

- speaker meta:      "is_ai", "ai_meta" ({"config", "role"})
- utterance meta:    "ai_meta" ({"config", "supports"})
- conversation meta: "ai_meta" ({"alias", "assistants", "supports"})
- corpus meta:       "has_ai", "ai_meta"

In memory, these fields are pulled out of .meta and exposed as attributes.
"""

from typing import Dict, Iterable, Optional, Tuple

# metadata keys holding AI fields, per object type
AI_META_KEYS = {
    "speaker": ("ai_meta", "is_ai"),
    "utterance": ("ai_meta",),
    "conversation": ("ai_meta",),
    "corpus": ("ai_meta", "has_ai"),
}

# corpus meta keys from earlier iterations of the format, migrated onto conversations on load
LEGACY_CORPUS_KEYS = ("supports", "assistants")

# keys from earlier iterations of the format that are dropped on load
LEGACY_AI_KEYS = ("ai_meta", "is_ai", "has_ai", "role", "sender_name", "agent_id")


def strip_keys(meta: Optional[Dict], keys: Iterable[str]) -> Dict:
    cleaned = dict(meta or {})
    for key in keys:
        cleaned.pop(key, None)
    return cleaned


def strip_legacy_ai_fields(meta: Optional[Dict]) -> Dict:
    return strip_keys(meta, LEGACY_AI_KEYS)


def split_ai_fields(
    meta: Optional[Dict], obj_type: str, strip_legacy: bool = True
) -> Tuple[Dict, Dict]:
    """
    Split a metadata dict into (regular metadata, AI fields) for the given object type.

    :param meta: metadata dict (or ConvoKitMeta)
    :param obj_type: "speaker", "utterance", "conversation", or "corpus"
    :param strip_legacy: whether to also drop the legacy keys in LEGACY_AI_KEYS
    :return: tuple of the cleaned metadata and a dict of the AI fields that were present
    """
    meta = dict(meta or {})
    ai_fields = {key: meta[key] for key in AI_META_KEYS[obj_type] if key in meta}
    keys = set(AI_META_KEYS[obj_type])
    if strip_legacy:
        keys |= set(LEGACY_AI_KEYS)
    return strip_keys(meta, keys), ai_fields


def as_dict(value) -> Dict:
    return dict(value) if isinstance(value, dict) else {}


def normalize_for_dump(value):
    """
    Recursively convert Support / Assistant objects into JSON-serializable dicts.
    """
    if hasattr(value, "to_dict") and not isinstance(value, dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): normalize_for_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_dump(item) for item in value]
    return value


def transcript_sort_key(obj) -> Tuple:
    """
    Sort key ordering Utterances / Supports by timestamp, then id.
    """
    # support timestamps may be strings while utterance timestamps are ints
    timestamp = obj.timestamp
    if timestamp is None:
        key = (0, 0)
    elif isinstance(timestamp, (int, float)):
        key = (0, timestamp)
    else:
        key = (1, str(timestamp))
    return key, str(obj.id)
