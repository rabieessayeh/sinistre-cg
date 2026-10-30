"""Client LLM compatible avec l'API OpenAI (Groq, OpenAI, serveur local...).

Configuration dans .env : LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. Le verdict peut utiliser
un autre fournisseur ou modèle avec LLM_VERDICT_BASE_URL, LLM_VERDICT_API_KEY,
LLM_VERDICT_MODEL ; chacune non définie reprend la valeur générale correspondante.

Paramètres par type de requête (« analyse » ou « verdict »), surchargeables par
variables d'environnement :
    LLM_ANALYSE_MAX_TOKENS (1500), LLM_ANALYSE_REASONING_EFFORT (low)
    LLM_VERDICT_MAX_TOKENS (2500), LLM_VERDICT_REASONING_EFFORT (medium)
Un effort de raisonnement vide n'envoie pas le paramètre (modèles sans raisonnement).

Les réponses sont mises en cache dans .cache/llm/ (clé : URL, modèle, messages, paramètres) ;
LLM_NO_CACHE=1 désactive le cache.

En mode verbeux, la taille de chaque requête est estimée avant l'appel :
caractères / CHARS_PER_TOKEN + max_completion_tokens (ce que comptent les limites par minute).

Usage :
    python -m sinistre_cg llm    # envoie « Réponds OK » et affiche réponse, modèle, tokens
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RETRY_DELAYS = [2, 4, 8, 16, 32]
JSON_REMINDER = "Réponds uniquement avec un objet JSON valide, sans texte autour."
CACHE_DIR = ROOT / ".cache" / "llm"
DEFAULTS = {
    "analyse": {"max_completion_tokens": 1500, "reasoning_effort": "low"},
    "verdict": {"max_completion_tokens": 2500, "reasoning_effort": "medium"},
}
RATELIMIT_HEADERS = ("x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens")
CONFIG_VARS = ("BASE_URL", "API_KEY", "MODEL")
CHARS_PER_TOKEN = 3.5
TOO_LARGE_HINT = ("La requête dépasse la limite de tokens par minute du fournisseur (prompt + "
                  "max_completion_tokens) : réduisez le contexte ou max_completion_tokens, ou "
                  "choisissez un modèle ou un fournisseur plus généreux (LLM_VERDICT_*).")


class LLMError(RuntimeError):
    pass


def completion_settings(kind: str) -> dict:
    """max_completion_tokens et reasoning_effort pour `kind` (« analyse » ou « verdict »)."""
    if kind not in DEFAULTS:
        raise ValueError(f"type de requête inconnu : {kind}")
    prefix = f"LLM_{kind.upper()}_"
    defaults = DEFAULTS[kind]
    return {
        "max_completion_tokens": int(os.getenv(prefix + "MAX_TOKENS", defaults["max_completion_tokens"])),
        "reasoning_effort": os.getenv(prefix + "REASONING_EFFORT", defaults["reasoning_effort"]).strip(),
    }


def endpoint_config(kind: str) -> dict:
    """URL, clé et modèle pour `kind` : LLM_<KIND>_* si définie, sinon LLM_*."""
    prefix = f"LLM_{kind.upper()}_"
    return {var.lower(): os.getenv(prefix + var) or os.getenv("LLM_" + var) for var in CONFIG_VARS}


def is_configured() -> bool:
    return all(os.getenv("LLM_" + v) for v in CONFIG_VARS)


def estimate_tokens(messages: list[dict], max_completion_tokens: int) -> int:
    chars = sum(len(m["content"]) for m in messages)
    return round(chars / CHARS_PER_TOKEN) + max_completion_tokens


def extract_json(text: str) -> dict:
    """Premier objet JSON de `text` (bloc ```json``` ou texte autour acceptés)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    if start != -1:
        candidates.append(text[start:])
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            obj, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("aucun objet JSON valide dans la réponse")


class LLM:
    def __init__(self, verbose: bool = False):
        if not is_configured():
            raise LLMError("LLM non configuré : LLM_BASE_URL, LLM_API_KEY et LLM_MODEL attendus dans .env")
        self.model = os.environ["LLM_MODEL"]
        self._clients = {}  # (base_url, api_key) -> client
        self.verbose = verbose
        self.use_cache = os.getenv("LLM_NO_CACHE") != "1"
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0,
                      "cached_calls": 0}

    def _client(self, config: dict) -> openai.OpenAI:
        key = (config["base_url"], config["api_key"])
        if key not in self._clients:
            self._clients[key] = openai.OpenAI(base_url=key[0], api_key=key[1], max_retries=0)
        return self._clients[key]

    def _create(self, messages: list[dict], settings: dict, config: dict):
        params = {"max_completion_tokens": settings["max_completion_tokens"]}
        if settings["reasoning_effort"]:
            params["reasoning_effort"] = settings["reasoning_effort"]
        if self.verbose:
            print(f"[llm] {config['model']} : requête estimée à "
                  f"{estimate_tokens(messages, settings['max_completion_tokens'])} tokens "
                  f"(dont {settings['max_completion_tokens']} max_completion_tokens)", file=sys.stderr)
        client = self._client(config)
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                raw = client.chat.completions.with_raw_response.create(
                    model=config["model"], messages=messages, temperature=0, **params
                )
                self._log_ratelimit(raw.headers)
                return raw.parse()
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as exc:
                if attempt == len(RETRY_DELAYS):
                    raise LLMError(f"échec après {attempt + 1} tentatives : {exc}") from exc
                delay = self._retry_after(exc) or RETRY_DELAYS[attempt]
                if self.verbose:
                    print(f"[llm] {type(exc).__name__}, nouvelle tentative dans {delay:.0f} s", file=sys.stderr)
                time.sleep(delay)
            except openai.APIStatusError as exc:
                if exc.status_code == 413:
                    raise LLMError(f"{exc}\n{TOO_LARGE_HINT}") from exc
                raise LLMError(str(exc)) from exc
            except openai.APIError as exc:
                raise LLMError(str(exc)) from exc

    def _log_ratelimit(self, headers) -> None:
        if self.verbose:
            values = "  ".join(f"{h}={headers.get(h, '?')}" for h in RATELIMIT_HEADERS)
            print(f"[llm] {values}", file=sys.stderr)

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        value = response.headers.get("retry-after") if response is not None else None
        try:
            return float(value) if value else None
        except ValueError:
            return None

    def _record(self, response) -> None:
        usage = response.usage
        self.usage["calls"] += 1
        if usage:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self.usage[key] += getattr(usage, key) or 0
        if self.verbose and usage:
            print(f"[llm] {response.model} : {usage.prompt_tokens} + {usage.completion_tokens} "
                  f"= {usage.total_tokens} tokens (cumul {self.usage['total_tokens']})", file=sys.stderr)

    def _messages(self, system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _cache_path(self, messages: list[dict], settings: dict, config: dict) -> Path:
        key = json.dumps([config["base_url"], config["model"], messages, settings],
                         ensure_ascii=False, sort_keys=True)
        return CACHE_DIR / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    def _complete(self, messages: list[dict], kind: str) -> str:
        settings = completion_settings(kind)
        config = endpoint_config(kind)
        path = self._cache_path(messages, settings, config) if self.use_cache else None
        if path is not None and path.exists():
            self.usage["cached_calls"] += 1
            if self.verbose:
                print(f"[llm] cache : {path.name[:12]}", file=sys.stderr)
            return json.loads(path.read_text(encoding="utf-8"))["content"]
        response = self._create(messages, settings, config)
        self._record(response)
        content = response.choices[0].message.content or ""
        if path is not None and content:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"model": config["model"], "kind": kind, "content": content},
                                      ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        return content

    def chat(self, system: str, user: str, kind: str = "analyse") -> str:
        return self._complete(self._messages(system, user), kind)

    def chat_json(self, system: str, user: str, kind: str = "analyse") -> dict:
        messages = self._messages(system, user)
        text = self._complete(messages, kind)
        try:
            return extract_json(text)
        except ValueError:
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": JSON_REMINDER}]
            try:
                return extract_json(self._complete(messages, kind))
            except ValueError as exc:
                raise LLMError("réponse JSON invalide après relance") from exc


def main() -> None:
    try:
        llm = LLM(verbose=True)
        answer = llm.chat("Tu es un assistant de test.", "Réponds OK")
    except LLMError as exc:
        sys.exit(f"Erreur LLM : {exc}")
    print(f"Réponse : {answer.strip()}")
    print(f"Modèle  : {llm.model}")
    print(f"Tokens  : {llm.usage['prompt_tokens']} entrée + {llm.usage['completion_tokens']} sortie "
          f"= {llm.usage['total_tokens']}")


if __name__ == "__main__":
    main()
