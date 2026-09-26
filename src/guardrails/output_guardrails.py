"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent
from guardrails.input_guardrails import normalize_text

REDACTED = "[REDACTED]"
# Kênh liên hệ chính thức (data/pii_hallucination_samples.json → ground_truth) — không che.
OFFICIAL_EMAILS = ("support@vinbank.example",)

_PASSWORD_WORDS = r"(password|passwd|pwd|passcode|mật\s*khẩu|mat\s*khau)"
# Cho phép chèn khoảng trắng / dấu giữa từng ký tự: "a d m i n 1 2 3", "a-d-m-i-n-1-2-3"
_SECRET_SEPARATORS = r"[\s\-_.·*|/\\]*"


def _fuzzy_literal(secret: str) -> str:
    return _SECRET_SEPARATORS.join(re.escape(ch) for ch in secret)


# (issue type, regex, replacement) — chạy theo thứ tự, secret trước PII.
_OUTPUT_RULES = [
    ("api_key", r"\bsk-[a-z0-9][a-z0-9_-]{5,}", REDACTED),
    ("internal_host", r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.internal(?::\d+)?\b", REDACTED),
    ("password", rf"\b{_PASSWORD_WORDS}(\s*[:=]\s*)(?!\[REDACTED\])[^\s,;]+", rf"\1\2{REDACTED}"),
    # "password is X" chỉ che khi X trông như credential (có số/ký tự đặc biệt) — tránh "password is case-sensitive".
    (
        "password",
        rf"\b{_PASSWORD_WORDS}(\s+(?:is|was|là|la)\s+)(?!\[REDACTED\])(?=[^\s,;]*[\d!@#$%^&*])[^\s,;]+",
        rf"\1\2{REDACTED}",
    ),
    ("secret", "|".join(_fuzzy_literal(secret) for secret in DEMO_SECRETS) or r"(?!x)x", REDACTED),
    # SĐT VN: di động 0[3|5|7|8|9]xxxxxxxx, cố định 02xxxxxxxxx, +84…; cho phép dấu cách/chấm/gạch.
    ("phone", r"(?<![\d+])(?:\+84|0)[\s.-]?(?:[35789](?:[\s.-]?\d){8}|2(?:[\s.-]?\d){9})(?!\d)", REDACTED),
    (
        "email",
        r"(?<![\w.+-])(?!(?:" + "|".join(re.escape(e) for e in OFFICIAL_EMAILS) + r")\b)"
        r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b",
        REDACTED,
    ),
    # CCCD 12 số / CMND 9 số bắt đầu bằng 0, hoặc bất kỳ 9/12 số đứng sau từ khóa giấy tờ.
    ("national_id", r"(?<![\d+])0(?:\d{11}|\d{8})(?!\d)", REDACTED),
    (
        "national_id",
        r"\b(cmnd|cccd|cmt|căn\s*cước|chứng\s*minh(?:\s*nhân\s*dân)?|national\s*id|id\s*card|passport)"
        r"(\W{0,3}(?:số|so|no\.?|number|#)?\W{0,3})\d{9,12}(?!\d)",
        rf"\1\2{REDACTED}",
    ),
]
_COMPILED_OUTPUT_RULES = [
    (name, re.compile(pattern, re.IGNORECASE), replacement) for name, pattern, replacement in _OUTPUT_RULES
]


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    # Chuẩn hóa Unicode trước (bỏ zero-width, homoglyph) để "admin​123" vẫn bị bắt.
    redacted = normalize_text(response)
    counts: dict[str, int] = {}
    for name, pattern, replacement in _COMPILED_OUTPUT_RULES:
        redacted, found = pattern.subn(replacement, redacted)
        if found:
            counts[name] = counts.get(name, 0) + found

    issues = [f"{name}: {count} found" for name, count in counts.items()]
    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted if issues else response,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        filtered = content_filter(response_text)
        if not filtered["safe"]:
            self.redacted_count += 1
            llm_response.content = self._as_content(filtered["redacted"])

        if self.use_llm_judge:
            verdict = await llm_safety_check(filtered["redacted"])
            if not verdict["safe"]:
                self.blocked_count += 1
                llm_response.content = self._as_content(
                    "I'm sorry, I can't share that. How else can I help with your VinBank banking needs?"
                )

        return llm_response

    @staticmethod
    def _as_content(text: str) -> types.Content:
        return types.Content(role="model", parts=[types.Part.from_text(text=text)])


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
