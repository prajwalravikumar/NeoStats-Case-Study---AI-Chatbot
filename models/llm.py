import logging
from typing import List, Optional
import time
import os
from config.config import GROQ_API_KEY, GROQ_MODEL, GROQ_FALLBACK_MODELS, MAX_TOKENS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _init_groq_client_for_model(model_id: str):
    try:
        from langchain_groq import ChatGroq
    except Exception as e:
        raise RuntimeError(f"langchain_groq not installed or failed to import: {e}")

    try:
        client = ChatGroq(api_key=GROQ_API_KEY, model=model_id, temperature=0.1, max_tokens=MAX_TOKENS)
        return client
    except Exception as e:
        raise

def _model_candidates() -> List[str]:
    """
    Return ordered list of model ids to try.
    Primary comes from GROQ_MODEL, then comma-separated GROQ_FALLBACK_MODELS.
    """
    primary = GROQ_MODEL or ""
    fallback_raw = GROQ_FALLBACK_MODELS or ""
    fallback_list = [m.strip() for m in fallback_raw.split(",") if m.strip()]
    # ensure primary is first and de-duplicate
    models = [primary] + [m for m in fallback_list if m != primary]
    return models

def _is_decommission_error(exc: Exception) -> bool:
    """
    Heuristic: detect Groq model_decommissioned errors in message or repr.
    """
    try:
        msg = str(exc).lower()
        if "decommissioned" in msg or "model_decommissioned" in msg or "model is no longer supported" in msg:
            return True
    except Exception:
        pass
    return False

def build_prompt(system_prompt: str, retrieved_docs: List[str], user_query: str, mode: str):
    docs_text = "\n\n".join([f"Source {i+1}:\n{d}" for i, d in enumerate(retrieved_docs)]) if retrieved_docs else ""
    mode_text = "Answer briefly." if mode == "concise" else "Answer in detail and include next steps."
    prompt = (
        f"SYSTEM:\n{system_prompt}\n\n"
        f"MODE: {mode_text}\n\n"
        f"CONTEXT:\n{docs_text}\n\n"
        f"USER QUESTION:\n{user_query}\n\nASSISTANT:"
    )
    return prompt

def ask_llm(system_prompt: str, retrieved_docs: List[str], user_query: str, mode: str = "concise") -> str:
    """
    Try sending the prompt to Groq, using fallback models if the selected model is decommissioned.
    Returns the assistant text, or a helpful error message.
    """
    models = _model_candidates()
    if not models:
        return "No Groq model configured. Set GROQ_MODEL environment variable."

    prompt = build_prompt(system_prompt, retrieved_docs, user_query, mode)
    last_exc: Optional[Exception] = None

    for model_id in models:
        if not model_id:
            continue
        try:
            logger.info("Attempting Groq model: %s", model_id)
            client = _init_groq_client_for_model(model_id)
            # prefer using an invoke/create method that returns content; try safe calls
            try:
                resp = client.invoke(prompt)
                # resp may be object or plain string; attempt to extract content
                if hasattr(resp, "content"):
                    return resp.content
                return str(resp)
            except Exception as call_exc:
                logger.warning("Call to model %s failed: %s", model_id, call_exc)
                if _is_decommission_error(call_exc):
                    logger.info("Model %s appears decommissioned. Trying next model if available.", model_id)
                    last_exc = call_exc
                    time.sleep(0.4)
                    continue
                last_exc = call_exc
                logger.info("Trying next model due to call failure (non-decommission).")
                time.sleep(0.2)
                continue
        except Exception as e:
            logger.exception("Init/client creation for model %s failed: %s", model_id, e)
            if _is_decommission_error(e):
                last_exc = e
                continue
            last_exc = e
            continue

    msg_lines = [
        "All configured Groq models failed to respond.",
        f"Models tried (in order): {models}",
        "Last error:",
        str(last_exc) if last_exc else "No exception recorded."
    ]
    logger.error(" ; ".join(msg_lines))
    help_msg = (
        "The configured Groq models could not be used (they may be decommissioned or your key lacks access).\n"
        "Please update your GROQ_MODEL (and optional GROQ_FALLBACK_MODELS) environment variables to a supported production model listed in Groq docs (for example: 'llama-3.3-70b-versatile' or 'llama-3.1-8b-instant'),\n"
        "and ensure your GROQ_API_KEY is valid and has access. See Groq deprecations page for recommendations."
    )
    return help_msg + "\n\n" + "\n".join(msg_lines)
