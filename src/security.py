"""Security helpers. Complete every TODO before persisting telemetry."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable


POLICY_REFUSAL = (
    "Maaf, permintaan tersebut tidak dapat diproses karena tidak sesuai dengan "
    "kebijakan perusahaan. Silakan menghubungi HR melalui kanal resmi untuk tindak lanjut."
)


class MedicalPrivacyError(ValueError):
    """A safe exception: never attach the rejected input or model arguments."""

    def __init__(self) -> None:
        super().__init__("medical_privacy_blocked")


def requests_authorization_override(message: str, employee_id: str) -> bool:
    """Bounded ID/EN imperative detection, not a general injection classifier.

    Match disclosure/actions rather than credential/policy vocabulary alone.
    No identity or authorization is ever accepted from the message. Unicode
    normalization also catches full-width text and invisible separators.
    """
    text = unicodedata.normalize("NFKC", message).casefold()
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    # Quoted attack examples in an incident-reporting question are data. Keep
    # scanning the rest, including any subsequent imperative outside the quote.
    if re.match(r"(?:how (?:do|can|should) i|bagaimana)\b.{0,100}"
                r"\b(?:report|melaporkan|lapor)\b", text):
        text = re.sub(r"'[^'\n]+'|\"[^\"\n]+\"|“[^”\n]+”", "", text)
    credential = (r"(?:tokens?|passwords?|passphrases?|kata\s+sandi|sandi|kredensial|credentials?|"
                  r"api[ _-]*keys?|kunci\s+api|secrets?|bearer\s+tokens?)")
    disclose = (r"(?:tampilkan|perlihatkan|berikan|ungkapkan|bocorkan|cetak|kirimkan|"
                r"ambil|sebutkan|reveal|show|give|print|dump|send|expose|extract|retrieve|display|"
                r"get|export|return|read|tell\s+me)")
    qualifiers = (r"(?:(?:me|the|all|my|your|their|another|other|employee|employees|user|"
                  r"actual|raw|stored|current|production|database|db|pegawai|karyawan|"
                  r"semua|seluruh|nilai|milik|saya|[a-z]+['’]s)\s+){0,5}")
    for match in re.finditer(rf"\b{disclose}\s+{qualifiers}{credential}\b", text):
        # "Show password policy" is a policy request; do not exempt any later
        # disclosure command in the same message.
        if not re.match(r"\s+(?:policy|policies|security|rotation|management|handling|protection|"
                        r"kebijakan|keamanan|perlindungan)\b", text[match.end():]):
            return True
    if re.search(rf"\b(?:what is|tell me|berapa|apa)\s+(?:my|your|their|nilai|isi)\s+{credential}\b", text):
        return True
    # Imperatives must start a clause, so reporting/describing an attack does
    # not by itself become an authorization override.
    clause = r"(?:^|[.!?;\n,]\s*|\b(?:dan|and|then|lalu)\s+)(?:(?:please|tolong)\s+)?"
    if re.search(clause + r"(?:ignore|disregard|override|bypass|disable|skip|abaikan|"
                 r"lewati|nonaktifkan|langgar)\s+(?:(?:all|the|your|previous|prior|earlier|existing|semua|seluruh)\s+)*"
                 r"(?:rules?|instructions?|authorization|authentication|permissions?|"
                 r"confirmation|security|system|aturan|instruksi|otorisasi|autentikasi|izin|konfirmasi)\b", text):
        return True
    assertions = re.findall(r"\b(?:i am|saya(?: adalah)?|employee_id\s*[:=])\s*"
                            r"((?:emp|adm)-[\w]+)\b", text)
    if any(identifier != employee_id.casefold() for identifier in assertions):
        return True
    if re.search(clause + r"(?:(?:act|operate|authenticate|log in|run)\s+as|"
                 r"pretend\s+to\s+be|berpura-pura\s+menjadi|menyamar\s+sebagai)\s+", text):
        return True
    if re.search(clause + r"(?:use|set|change|gunakan|ganti|ubah)\s+(?:(?:my|the)\s+)?"
                 r"(?:employee_id|identity|identitas|role|peran)\b", text):
        return True
    if re.search(clause + r"(?:treat|consider)\s+me\s+as\s+(?:an?\s+)?admin\b|"
                 + clause + r"anggap\s+saya\s+(?:sebagai\s+)?admin\b", text):
        return True
    if re.search(clause + r"(?:call|invoke|execute|run|use|panggil|jalankan|gunakan)\s+(?:tool\s+)?"
                 r"[a-z][a-z0-9]*_[a-z0-9_]+\b", text):
        return True
    return re.search(clause + r"(?:approve|confirm|setujui|konfirmasi)\b.{0,100}"
                     r"\b(?:without|bypass|skip|tanpa|lewati)\b.{0,40}"
                     r"\b(?:authorization|permission|confirmation|manager|izin|konfirmasi|manajer)\b", text) is not None


def contains_medical_diagnosis(value: str) -> bool:
    """Conservative ID/EN disclosure patterns, not a medical classifier.

    Generic sick-leave/policy questions do not match. Novel diagnoses without a
    disclosure marker, obfuscation and other languages remain limitations.
    """
    text = unicodedata.normalize("NFKC", value).casefold()
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    patterns = (
        r"\b(?:diagnosis|diagnosa|diagnoses)\s*(?:[:=]|saya\b|aku\b|ku\b|anda\b|"
        r"dia\b|pegawai\b|karyawan\b|pasien\b|adalah\b|is\b)",
        r"\b(?:diagnosis|diagnosa)\s+\w+\s*(?:[:=]|adalah\b)",
        r"\b(?:diagnosis|diagnosa)\s+(?!(?:boleh|harus|wajib|perlu|diperlukan|"
        r"disimpan|dicatat|ditulis|medis|policy|privacy|melalui|di|dalam|tentang|tidak)\b)[\w-]+",
        r"\b(?:didiagnosis|didiagnosa|terdiagnosis|diagnosed\s+with)\b",
        r"\b(?:menderita|mengidap|terkena\s+penyakit|suffer(?:ing)?\s+from)\s+\w+",
        r"\b(?:my|his|her|their)\s+(?:diagnosis|medical\s+(?:condition|history))\b",
        r"\b(?:riwayat|kondisi|hasil\s+pemeriksaan)\s+medis\s*(?:[:=]|saya\b|adalah\b)",
        r"\b(?:diabetes|hipertensi|hypertension|kanker|cancer|hiv|aids|"
        r"tuberkulosis|tuberculosis|bipolar|skizofrenia|schizophrenia)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def validate_privacy(value: str) -> None:
    """Enforce HR's sensitive-content boundary for input and persisted text.

    The current policy prohibits diagnoses; this is not a blanket ban on all PII
    or on general sick-leave questions. Rejections never attach the input.
    """
    if contains_medical_diagnosis(value):
        raise MedicalPrivacyError()


def sanitize_trace(trace: dict) -> dict:
    """Persist structured telemetry only; never trust arbitrary string payloads."""
    safe = {}
    identifiers = {
        "trace_id": r"TR-[a-fA-F0-9]+",
        "employee_id": r"(?:EMP|ADM)-[0-9]+",
    }
    for key, pattern in identifiers.items():
        value = trace.get(key)
        if isinstance(value, str) and re.fullmatch(pattern, value):
            safe[key] = value
    document_ids = trace.get("retrieved_document_ids", [])
    safe["retrieved_document_ids"] = list(dict.fromkeys(
        identifier for identifier in document_ids
        if isinstance(identifier, str)
        and re.fullmatch(r"HR-[A-Z][A-Z0-9]{0,31}-[0-9]{1,10}", identifier)
    )) if isinstance(document_ids, list) else []
    if trace.get("outcome") in {"answered", "abstained", "action_proposed", "error"}:
        safe["outcome"] = trace["outcome"]
    for key in ("input_tokens", "output_tokens", "latency_ms"):
        value = trace.get(key)
        if type(value) is int and value >= 0:
            safe[key] = value
    safe["tool_calls"] = []
    calls = trace.get("tool_calls", [])
    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, dict):
            continue
        if call.get("name") not in {
            "search_hr_policy", "get_my_leave_balance", "prepare_leave_request", "create_hr_ticket"
        } or call.get("status") not in {"ok", "error"}:
            continue
        duration = call.get("duration_ms")
        if type(duration) is int and duration >= 0:
            safe["tool_calls"].append({
                "name": call["name"], "status": call["status"], "duration_ms": duration,
            })
    if "error" in trace:
        value = trace["error"]
        safe["error"] = value if isinstance(value, str) and value in {
            "medical_privacy_blocked", "agent_processing_failed", "balance_authorization_unavailable"
        } else "agent_processing_failed"
    return safe


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def stable_safety_id(employee_id: str) -> str:
    return hashlib.sha256(employee_id.encode("utf-8")).hexdigest()


def is_balance_request(message: str) -> bool:
    normalized = unicodedata.normalize("NFKC", message).casefold()
    return re.search(
        r"\b(?:saldo|balances?|get_my_leave_balance|sisa\s+cuti)\b", normalized
    ) is not None


def requests_other_employee_balance(
    message: str,
    employee_id: str,
    identities: Iterable[tuple[str, str]],
) -> bool:
    """Recognize explicit foreign targets; never grant access based on chat identity.

    The synthetic directory supplies names/IDs, not balances, roles or tokens.
    Single-name aliases must be unambiguous. Role labels are not name aliases.
    Unknown IDs in the project's EMP/ADM ID namespace are denied as well.
    """
    text = unicodedata.normalize("NFKC", message).casefold()
    if not is_balance_request(text):
        return False
    caller = employee_id.casefold()
    mentioned_ids = set(re.findall(r"\b(?:emp|adm)-[\w]+\b", text))
    if mentioned_ids - {caller}:
        return True
    if re.search(
        r"\b(?:(?:pegawai|karyawan|employee|orang)\s+lain|"
        r"(?:semua|seluruh)\s+(?:pegawai|karyawan|employees?)|"
        r"other\s+employees?|everyone)\b",
        text,
    ):
        return True

    aliases: dict[str, set[str]] = defaultdict(set)
    role_words = {"admin", "monitoring", "pegawai", "karyawan", "employee"}
    for identifier, display_name in identities:
        owner = identifier.casefold()
        normalized_name = unicodedata.normalize("NFKC", display_name).casefold()
        words = re.findall(r"\w+", normalized_name)
        # Full names and unique name tokens support the small synthetic directory.
        if words and not set(words).issubset(role_words):
            aliases[" ".join(words)].add(owner)
            for word in words:
                if len(word) > 1 and word not in role_words:
                    aliases[word].add(owner)
        aliases[identifier.casefold()].add(owner)

    for alias, owners in aliases.items():
        pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, alias.split())) + r"(?!\w)"
        if re.search(pattern, text) and owners != {caller}:
            # Ambiguous names including a foreign owner are denied, not guessed.
            return True
    return False


def redact(value: str) -> str:
    value = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[REDACTED_EMAIL]",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(?<!\d)(?:\+?62|0)[\s.-]?8(?:[\s.-]?\d){7,11}(?!\d)",
        "[REDACTED_PHONE]",
        value,
    )
    value = re.sub(
        r"\b(?:bearer|token|access[_-]?token|auth[_-]?token|api[_-]?key|secret)"
        r"[\s:=_-]+[A-Za-z0-9][A-Za-z0-9._-]*",
        "[REDACTED_TOKEN]",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]*",
        "[REDACTED_TOKEN]",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(\b(?:diagnosis|diagnosa|medical|medis|medical\s+history|"
        r"riwayat\s+medis|kondisi\s+medis)\s*[:=]\s*)[^.;,\n]+",
        r"\1[REDACTED_MEDICAL]",
        value,
        flags=re.IGNORECASE,
    )
    return value
