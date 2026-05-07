"""
PharmaPath AI — LLM Report Parsing Service
============================================
Три бэкенда: mock (тест), ollama (локально), openai (облако).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger()

SYSTEM_PROMPT = """Ты аналитик фармацевтической компании.
Из текста отчёта медицинского представителя извлеки следующую информацию и верни СТРОГО в JSON:

{
  "sentiment": "Positive" | "Neutral" | "Negative",
  "competitors": ["список упомянутых конкурентных препаратов"],
  "objections": ["возражения врача"],
  "agreements": ["о чём договорились"],
  "key_topics": ["ключевые темы разговора"]
}

Если какое-то поле не обнаружено — верни пустой список.
Отвечай ТОЛЬКО JSON, без markdown, без пояснений."""


@dataclass
class ParsedReport:
    """Результат парсинга отчёта."""
    sentiment: str
    competitors: List[str]
    objections: List[str]
    agreements: List[str]
    key_topics: List[str]
    raw_response: str
    backend: str


class BaseLLMService(ABC):
    """Интерфейс LLM-сервиса."""

    @abstractmethod
    def parse_report(self, text: str) -> ParsedReport:
        ...


# ══════════════════════════════════════════════════════════════════════════════
#  MOCK (для тестов и демо без LLM)
# ══════════════════════════════════════════════════════════════════════════════

class MockLLMService(BaseLLMService):
    """Эвристический парсер (regex + ключевые слова). Без LLM."""

    # Ключевые слова
    _POSITIVE = {"заинтересован", "положительно", "готов", "назначить", "согласился", "попробовать"}
    _NEGATIVE = {"скептически", "отказ", "нет", "не готов", "не заинтересован", "возражает"}
    _COMPETITOR_PATTERNS = [
        r"(?:использует|назначает|применяет|упомянул)\s+(\w+)",
    ]
    _OBJECTION_KEYWORDS = ["цена", "побочн", "формуляр", "дорог", "эффект", "привыкли"]
    _AGREEMENT_KEYWORDS = ["договорились", "назначить", "визит через", "образцы", "круглый стол"]

    def parse_report(self, text: str) -> ParsedReport:
        text_lower = text.lower()

        # Sentiment
        pos = sum(1 for w in self._POSITIVE if w in text_lower)
        neg = sum(1 for w in self._NEGATIVE if w in text_lower)
        if pos > neg:
            sentiment = "Positive"
        elif neg > pos:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Competitors (наивный regex)
        competitors = []
        for pat in self._COMPETITOR_PATTERNS:
            competitors.extend(re.findall(pat, text, re.IGNORECASE))

        # Objections
        objections = []
        for kw in self._OBJECTION_KEYWORDS:
            if kw in text_lower:
                # Извлечь предложение с ключевым словом
                for sent in text.split("."):
                    if kw in sent.lower():
                        objections.append(sent.strip())
                        break

        # Agreements
        agreements = []
        for kw in self._AGREEMENT_KEYWORDS:
            if kw in text_lower:
                for sent in text.split("."):
                    if kw in sent.lower():
                        agreements.append(sent.strip())
                        break

        # Key topics
        key_topics = []
        drug_pattern = r"(?:обсудили|презентовал|детейлинг)\s+(\w+)"
        key_topics.extend(re.findall(drug_pattern, text, re.IGNORECASE))

        return ParsedReport(
            sentiment=sentiment,
            competitors=list(set(competitors))[:5],
            objections=list(set(objections))[:5],
            agreements=list(set(agreements))[:5],
            key_topics=list(set(key_topics))[:5],
            raw_response="mock_heuristic",
            backend="mock",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  OLLAMA (локальная Llama-3)
# ══════════════════════════════════════════════════════════════════════════════

class OllamaLLMService(BaseLLMService):
    """Парсинг через локальную модель (Ollama)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model

    def parse_report(self, text: str) -> ParsedReport:
        import httpx

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{SYSTEM_PROMPT}\n\nТекст отчёта:\n{text}",
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=30.0,
            )
            response.raise_for_status()
            raw = response.json().get("response", "{}")
            parsed = self._extract_json(raw)
            return ParsedReport(
                sentiment=parsed.get("sentiment", "Neutral"),
                competitors=parsed.get("competitors", []),
                objections=parsed.get("objections", []),
                agreements=parsed.get("agreements", []),
                key_topics=parsed.get("key_topics", []),
                raw_response=raw,
                backend=f"ollama/{self.model}",
            )
        except Exception as e:
            logger.error("Ollama call failed", error=str(e))
            return MockLLMService().parse_report(text)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Извлечь JSON из текста LLM (может содержать мусор вокруг)."""
        # Ищем первый { ... }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def create_llm_service(
    backend: str = "mock",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llama3",
) -> BaseLLMService:
    """Фабрика LLM-сервисов."""
    if backend == "ollama":
        return OllamaLLMService(base_url=ollama_url, model=ollama_model)
    # default: mock
    return MockLLMService()