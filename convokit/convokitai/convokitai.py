"""
convokitai.py
================
This module defines the custom classes for the ConvoKit AI format,
extending the base ConvoKit architecture.
"""
import json
import os
from typing import Iterable

from convokit import Speaker as BaseSpeaker
from convokit import Conversation as BaseConversation
from convokit import Utterance as BaseUtterance
from convokit import Corpus as BaseCorpus


class Assistant:
    def __init__(
        self,
        id: str,
        config: dict = None,
        prompt: str = None,
        conversation_id: str = None,
        speakers: Iterable[str] = None,
        supports: Iterable["Support"] = None,
    ):
        self.id = id
        self.config = config or {}
        self.prompt = prompt or ""
        self.conversation_id = conversation_id
        self.speakers = list(speakers or [])
        self.supports = [
            s.to_dict() if hasattr(s, "to_dict") else s
            for s in (supports or [])
        ]

    def to_dict(self):
        return {
            "id": self.id,
            "config": self.config,
            "prompt": self.prompt,
            "conversation_id": self.conversation_id,
            "speakers": list(self.speakers),
            "supports": list(self.supports),
        }

    @classmethod
    def from_dict(cls, data: dict):
        if not isinstance(data, dict):
            return None
        return cls(
            id=data.get("id"),
            config=data.get("config") or {},
            prompt=data.get("prompt"),
            conversation_id=data.get("conversation_id"),
            speakers=data.get("speakers") or [],
            supports=data.get("supports") or [],
        )


class Support:
    def __init__(
        self,
        id: str,
        text: str,
        reply_to: str | None = None,
        speaker_id: str = None,
        assistant_id: str = None,
        timestamp: str | int | None = None,
        conversation_id: str = None,
        meta: dict = None,
    ):
        self.id = id
        self.text = text
        self.reply_to = reply_to
        self.speaker_id = speaker_id
        self.assistant_id = assistant_id
        self.timestamp = timestamp
        self.conversation_id = conversation_id
        self.meta = dict(meta or {})

    def to_dict(self):
        payload = {
            "id": self.id,
            "text": self.text,
            "reply_to": self.reply_to,
            "speaker_id": self.speaker_id,
            "assistant_id": self.assistant_id,
            "timestamp": self.timestamp,
            "conversation_id": self.conversation_id,
        }
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload

    @classmethod
    def from_dict(cls, data: dict):
        if not isinstance(data, dict):
            return None
        return cls(
            id=data.get("id"),
            text=data.get("text", ""),
            reply_to=data.get("reply_to"),
            speaker_id=data.get("speaker_id"),
            assistant_id=data.get("assistant_id"),
            timestamp=data.get("timestamp"),
            conversation_id=data.get("conversation_id"),
            meta=data.get("meta") or {},
        )


def _strip_legacy_ai_fields(meta: dict | None) -> dict:
    cleaned = dict(meta or {})
    for key in ("ai_meta", "is_ai", "has_ai", "role", "sender_name", "agent_id"):
        cleaned.pop(key, None)
    return cleaned


class Speaker(BaseSpeaker):
    def __init__(self, id: str, is_ai: bool = False, ai_meta: dict = None, meta: dict = None, **kwargs):
        super().__init__(id=id, meta=_strip_legacy_ai_fields(meta), **kwargs)
        self.meta = _strip_legacy_ai_fields(self.meta)
        self.is_ai = is_ai
        self.ai_meta = ai_meta or {}

    @property
    def is_ai(self):
        return getattr(self, "_is_ai", False)

    @is_ai.setter
    def is_ai(self, value):
        self._is_ai = bool(value)

    @property
    def ai_meta(self):
        ai_meta = getattr(self, "_ai_meta", {})
        if not isinstance(ai_meta, dict):
            ai_meta = {}
        return ai_meta

    @ai_meta.setter
    def ai_meta(self, value):
        if isinstance(value, dict):
            base = getattr(self, "_ai_meta", {}).copy() if isinstance(getattr(self, "_ai_meta", {}), dict) else {}
            self._ai_meta = {**base, **value}
        else:
            self._ai_meta = {}


class Conversation(BaseConversation):
    def __init__(self, owner=None, id: str = None, ai_meta: dict = None, meta: dict = None, **kwargs):
        meta = _strip_legacy_ai_fields(meta)
        super().__init__(id=id, meta=meta, **kwargs, owner=owner)
        if "ai_meta" in getattr(self, "meta", {}):
            legacy_ai_meta = self.meta.pop("ai_meta", {})
            ai_meta = legacy_ai_meta if not ai_meta else {**legacy_ai_meta, **(ai_meta or {})}
        self.ai_meta = ai_meta or {}

    @property
    def ai_meta(self):
        return getattr(self, "_ai_meta", {})

    @ai_meta.setter
    def ai_meta(self, value):
        if isinstance(value, dict):
            base = getattr(self, "_ai_meta", {}).copy() if isinstance(getattr(self, "_ai_meta", {}), dict) else {}
            merged = {**base, **value}
            for key, item in list(merged.items()):
                if key == "supports":
                    merged[key] = self._normalize_supports(item)
            self._ai_meta = merged
        else:
            self._ai_meta = {}

    @staticmethod
    def _normalize_supports(value):
        normalized = []
        for item in value or []:
            if isinstance(item, Support):
                normalized.append(item)
            elif isinstance(item, dict):
                support = Support.from_dict(item)
                if support is not None:
                    normalized.append(support)
            else:
                normalized.append(item)
        return normalized

    @property
    def supports(self):
        return self._normalize_supports(self.ai_meta.get("supports", []))

    @supports.setter
    def supports(self, value):
        self.ai_meta = {**dict(self.ai_meta), "supports": self._normalize_supports(value)}

    def get_transcript(self, supports: bool = False) -> str:
        alias_map = self.ai_meta.get("alias", {}) or {}
        utterances = list(self.iter_utterances())
        try:
            utterances = sorted(
                utterances,
                key=lambda u: (
                    u.timestamp if isinstance(u.timestamp, (int, float)) else 0,
                    str(getattr(u, "id", "")),
                ),
            )
        except Exception:
            pass

        support_records = sorted(
            self.supports,
            key=lambda s: (
                s.timestamp if isinstance(s.timestamp, (int, float)) else 0,
                str(getattr(s, "id", "")),
            ),
        )

        lines = []
        seen_support_ids = set()
        prior_text_by_speaker = {}

        for utterance in utterances:
            speaker_alias = alias_map.get(utterance.speaker.id, utterance.speaker.id)
            prior_texts = prior_text_by_speaker.setdefault(utterance.speaker.id, [])

            if supports:
                for support in support_records:
                    if support.id in seen_support_ids:
                        continue
                    if getattr(support, "speaker_id", None) != utterance.speaker.id:
                        continue
                    if getattr(support, "timestamp", None) is not None and support.timestamp > utterance.timestamp:
                        continue

                    reply_to = getattr(support, "reply_to", None) or ""
                    assistant_label = getattr(support, "assistant_id", "assistant")

                    if not reply_to:
                        lines.append(f"    Assistant[{assistant_label}]: {support.text}")
                        seen_support_ids.add(support.id)
                        continue

                    matching_reply = any(
                        existing == reply_to or existing.startswith(reply_to) or reply_to.startswith(existing)
                        for existing in prior_texts
                    )
                    if not matching_reply:
                        continue

                    lines.append(f"    Assistant[{assistant_label}] -> {speaker_alias}:")
                    lines.append(f"        {speaker_alias}: {reply_to}")
                    lines.append(f"        {assistant_label}: {support.text}")
                    seen_support_ids.add(support.id)

            lines.append(f"{speaker_alias}: {utterance.text}")
            prior_texts.append(utterance.text)

        return "\n".join(lines)


class Utterance(BaseUtterance):
    def __init__(self, id: str, speaker: Speaker, conversation_id: str, reply_to: str, timestamp: int, text: str, ai_meta: dict = None, meta: dict = None, **kwargs):
        super().__init__(id=id, speaker=speaker, conversation_id=conversation_id, reply_to=reply_to, timestamp=timestamp, text=text, meta=meta, **kwargs)
        self.meta.pop("ai_meta", None)
        if "ai_meta" not in self.__dict__:
            self.ai_meta = ai_meta or {}

    @property
    def ai_meta(self):
        return getattr(self, "_ai_meta", {})

    @ai_meta.setter
    def ai_meta(self, value):
        self._ai_meta = value or {}


class Corpus(BaseCorpus):
    def __init__(self, utterances=None, has_ai: bool = False, assistants: dict = None, supports: list = None, meta: dict = None, **kwargs):
        super().__init__(utterances=utterances, **kwargs)
        cleaned_meta = _strip_legacy_ai_fields(meta)
        if cleaned_meta:
            self.meta.update(cleaned_meta)
        self.meta.pop("has_ai", None)
        self.meta.pop("ai_meta", None)
        self.meta.pop("supports", None)
        self.meta.pop("assistants", None)
        self.has_ai = has_ai
        self.ai_meta = {}
        self.assistants = assistants or {}
        self.supports = supports or []

    @property
    def has_ai(self):
        return getattr(self, "_has_ai", False)

    @has_ai.setter
    def has_ai(self, value):
        self._has_ai = bool(value)

    @property
    def ai_meta(self):
        return getattr(self, "_ai_meta", {})

    @ai_meta.setter
    def ai_meta(self, value):
        if isinstance(value, dict):
            base = getattr(self, "_ai_meta", {}).copy() if isinstance(getattr(self, "_ai_meta", {}), dict) else {}
            self._ai_meta = {**base, **value}
        else:
            self._ai_meta = {}

    @property
    def supports(self):
        raw_supports = getattr(self, "_supports", self.meta.get("supports", []))
        if not isinstance(raw_supports, list):
            return []
        return [
            item if isinstance(item, Support) else Support.from_dict(item)
            for item in raw_supports
            if item is not None
        ]

    @supports.setter
    def supports(self, value):
        normalized = []
        for item in value or []:
            if isinstance(item, Support):
                normalized.append(item)
            elif isinstance(item, dict):
                parsed = Support.from_dict(item)
                if parsed is not None:
                    normalized.append(parsed)
            else:
                normalized.append(item)
        self._supports = normalized
        self.meta["supports"] = [
            support.to_dict() if hasattr(support, "to_dict") else support
            for support in normalized
        ]

    @property
    def assistants(self):
        raw_assistants = self.ai_meta.get("assistants", {})
        if not isinstance(raw_assistants, dict):
            raw_assistants = {}
        return {
            k: Assistant.from_dict(v) if isinstance(v, dict) else v
            for k, v in raw_assistants.items()
        }

    @assistants.setter
    def assistants(self, value):
        assistant_payload = {}
        for k, v in (value or {}).items():
            if isinstance(v, Assistant):
                assistant_payload[k] = v.to_dict()
            elif isinstance(v, dict):
                assistant_payload[k] = v
            else:
                assistant_payload[k] = v
        self.ai_meta = {**dict(self.ai_meta), "assistants": assistant_payload}

    @staticmethod
    def _normalize_for_dump(value):
        if isinstance(value, Support):
            return value.to_dict()
        if isinstance(value, Assistant):
            return value.to_dict()
        if isinstance(value, dict):
            return {str(key): Corpus._normalize_for_dump(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [Corpus._normalize_for_dump(item) for item in value]
        return value

    def dump(self, name: str, base_path: str = None, **kwargs):
        speaker_backups = {}
        conversation_backups = {}
        corpus_ai_meta = self._normalize_for_dump(self.ai_meta) if isinstance(self.ai_meta, dict) else {}
        corpus_has_ai = self.has_ai
        support_snapshot = list(self.supports)

        for speaker in self.iter_speakers():
            speaker_backups[speaker.id] = {
                "meta": dict(getattr(speaker, "meta", {}) or {}),
                "ai_meta": dict(getattr(speaker, "ai_meta", {}) or {}),
                "is_ai": bool(getattr(speaker, "is_ai", False)),
            }
            speaker.meta["ai_meta"] = dict(speaker.ai_meta)
            speaker.meta["is_ai"] = speaker.is_ai

        for conversation in self.iter_conversations():
            conversation.__class__ = Conversation
            conversation_backups[conversation.id] = {
                "meta": dict(getattr(conversation, "meta", {}) or {}),
                "ai_meta": dict(getattr(conversation, "ai_meta", {}) or {}),
            }
            assistant_ids = [
                assistant.id
                for assistant in self.assistants.values()
                if getattr(assistant, "conversation_id", None) == conversation.id
            ]
            relevant = [
                support
                for support in support_snapshot
                if getattr(support, "conversation_id", None) == conversation.id
            ]
            conversation_ai_meta = dict(getattr(conversation, "ai_meta", {}) or {})
            if assistant_ids:
                conversation_ai_meta["assistants"] = assistant_ids
            if relevant:
                conversation_ai_meta["supports"] = [support.to_dict() for support in relevant]
            normalized_conversation_ai_meta = self._normalize_for_dump(conversation_ai_meta)
            conversation.ai_meta = normalized_conversation_ai_meta
            conversation.meta["ai_meta"] = normalized_conversation_ai_meta

        corpus_ai_meta["assistants"] = {
            assistant_id: assistant.to_dict()
            for assistant_id, assistant in self.assistants.items()
        }
        self.ai_meta = self._normalize_for_dump(corpus_ai_meta) if isinstance(corpus_ai_meta, dict) else {}
        self.meta["supports"] = [
            support.to_dict() if hasattr(support, "to_dict") else support
            for support in support_snapshot
        ]
        self.meta["ai_meta"] = self._normalize_for_dump(self.ai_meta) if isinstance(self.ai_meta, dict) else {}
        self.meta["has_ai"] = bool(corpus_has_ai)

        dump_dir = os.path.join(base_path, name) if base_path else name
        os.makedirs(dump_dir, exist_ok=True)
        support_path = os.path.join(dump_dir, "supports.json")

        try:
            super().dump(name=name, base_path=base_path, **kwargs)
            with open(support_path, "w", encoding="utf-8") as fh:
                json.dump({"supports": [support.to_dict() for support in support_snapshot]}, fh, ensure_ascii=False, indent=2)
        finally:
            for speaker in self.iter_speakers():
                backup = speaker_backups.get(speaker.id)
                if backup is None:
                    continue
                speaker.meta = backup["meta"]
                speaker.ai_meta = backup["ai_meta"]
                speaker.is_ai = backup["is_ai"]
            for conversation in self.iter_conversations():
                backup = conversation_backups.get(conversation.id)
                if backup is None:
                    continue
                conversation.meta = backup["meta"]
                conversation.ai_meta = backup["ai_meta"]
            self.meta.pop("ai_meta", None)
            self.meta.pop("has_ai", None)
            self.meta.pop("supports", None)
            self.ai_meta = corpus_ai_meta
            self.has_ai = corpus_has_ai
            self.supports = support_snapshot

    @staticmethod
    def _normalize_meta_fields(obj):
        if obj is None:
            return {}, {}
        meta = dict(getattr(obj, "meta", {}) or {})
        for key in ("ai_meta", "is_ai", "has_ai", "role", "sender_name", "agent_id"):
            meta.pop(key, None)
        ai_meta = getattr(obj, "ai_meta", {})
        if not isinstance(ai_meta, dict):
            ai_meta = {}
        if "role" in getattr(obj, "meta", {}):
            ai_meta.setdefault("role", getattr(obj, "meta", {}).get("role"))
        is_ai = getattr(obj, "is_ai", False)
        has_ai = getattr(obj, "has_ai", False)
        if isinstance(obj, (Corpus, BaseCorpus)):
            meta.pop("assistants", None)
        return meta, {"ai_meta": ai_meta, "is_ai": is_ai, "has_ai": has_ai}

    @classmethod
    def load(cls, filename: str):
        base_corpus = BaseCorpus(filename=filename)
        custom_utterances = []
        custom_speakers = {}

        for utt in base_corpus.iter_utterances():
            base_spk = utt.speaker
            speaker_meta = dict(getattr(base_spk, "meta", {}) or {})
            speaker_ai_meta = speaker_meta.pop("ai_meta", {})
            speaker_is_ai = bool(speaker_meta.pop("is_ai", False))
            speaker_meta = _strip_legacy_ai_fields(speaker_meta)
            if base_spk.id not in custom_speakers:
                custom_speakers[base_spk.id] = Speaker(
                    id=base_spk.id,
                    is_ai=speaker_is_ai,
                    ai_meta=speaker_ai_meta,
                    meta=speaker_meta,
                )

            spk = custom_speakers[base_spk.id]
            utt_meta = dict(getattr(utt, "meta", {}) or {})
            utt_ai_meta = utt_meta.pop("ai_meta", {})
            utt_meta = _strip_legacy_ai_fields(utt_meta)
            custom_utt = Utterance(
                id=utt.id,
                speaker=spk,
                conversation_id=utt.conversation_id,
                reply_to=utt.reply_to,
                timestamp=utt.timestamp,
                text=utt.text,
                ai_meta=utt_ai_meta,
                meta=utt_meta,
            )
            custom_utterances.append(custom_utt)

        custom_corpus = cls(utterances=custom_utterances, meta=base_corpus.meta)
        corpus_meta = dict(getattr(base_corpus, "meta", {}) or {})
        ai_meta_raw = corpus_meta.pop("ai_meta", {})
        corpus_ai_meta = ai_meta_raw if isinstance(ai_meta_raw, dict) else {}
        custom_corpus.ai_meta = corpus_ai_meta
        custom_corpus.has_ai = bool(corpus_meta.pop("has_ai", False))

        assistants_raw = corpus_ai_meta.get("assistants", {})
        if isinstance(assistants_raw, dict):
            custom_corpus.assistants = {
                k: Assistant.from_dict(v) if isinstance(v, dict) else v
                for k, v in assistants_raw.items()
            }

        support_path = os.path.join(filename, "supports.json")
        if os.path.exists(support_path):
            with open(support_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            support_payload = payload.get("supports", []) if isinstance(payload, dict) else payload
            custom_corpus.supports = [
                Support.from_dict(item) if isinstance(item, dict) else item
                for item in support_payload
            ]

        for base_convo in base_corpus.iter_conversations():
            convo_meta = dict(getattr(base_convo, "meta", {}) or {})
            convo_ai_meta = dict(convo_meta.pop("ai_meta", {}) or {})
            convo_meta = _strip_legacy_ai_fields(convo_meta)
            custom_convo = custom_corpus.get_conversation(base_convo.id)
            if custom_convo is not None:
                custom_convo.__class__ = Conversation
                custom_convo.meta = convo_meta
                custom_convo.ai_meta = convo_ai_meta
                custom_convo.supports = convo_ai_meta.get("supports", [])

        if custom_corpus.supports:
            by_conversation = {}
            for support in custom_corpus.supports:
                conversation_id = getattr(support, "conversation_id", None)
                if conversation_id is None:
                    continue
                by_conversation.setdefault(conversation_id, []).append(support.to_dict())
            for conversation_id, records in by_conversation.items():
                convo = custom_corpus.get_conversation(conversation_id)
                if convo is not None:
                    convo.ai_meta = {**convo.ai_meta, "supports": records}

        return custom_corpus