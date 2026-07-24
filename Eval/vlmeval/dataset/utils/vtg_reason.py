import re

REASON_PROMPT_SUFFIX = (
    "\n\nYou may reason briefly inside <thinking></thinking> tags before giving the final answer.\n\n"
    "Format:\n"
    "<thinking>\n"
    "Brief reasoning here.\n"
    "</thinking>\n"
    "<answer>\n"
    "The relevant time interval here.\n"
    "</answer>"
)


def wrap_vtg_prompt_with_reason(prompt: str) -> str:
    return f"{prompt}{REASON_PROMPT_SUFFIX}"


def extract_answer_from_tags(text: object) -> str:
    if text is None:
        return ""
    s = str(text)
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    matches = re.findall(pattern, s, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return s
