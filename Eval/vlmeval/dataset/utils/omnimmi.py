import json
import re
from typing import Any

import pandas as pd


_DICT_REGEX = re.compile(r'\{[\s\S]*\}')


def has_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and pd.isna(value))


def parse_python_dict_text(text: str) -> dict[str, Any] | None:
    """Parse a python-dict-like string possibly wrapped in markdown fences."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    raw = raw.replace("```python", "").replace("```json", "").replace("```", "").strip()

    m = _DICT_REGEX.search(raw)
    if m:
        raw = m.group(0).strip()

    raw_json_try = raw.replace("True", "true").replace("False", "false")
    try:
        obj = json.loads(raw_json_try)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    import ast

    try:
        obj = ast.literal_eval(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def normalize_yesno(value: Any) -> str:
    if value is None:
        return ''
    s = str(value).strip().lower()
    if s in ('yes', 'y', 'true', '1'):
        return 'yes'
    if s in ('no', 'n', 'false', '0'):
        return 'no'
    if 'yes' in s:
        return 'yes'
    if 'no' in s:
        return 'no'
    return ''


def clip_score(value: Any) -> int | None:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        try:
            v = float(str(value).strip())
        except Exception:
            return None
    v = int(round(v))
    return int(min(5, max(0, v)))


def try_parse_json_pred(pred_value: Any) -> dict[str, Any] | None:
    if pred_value is None or (isinstance(pred_value, float) and pd.isna(pred_value)):
        return None
    raw = str(pred_value).strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def pred_has_response(pred_value: Any) -> bool:
    """Whether model 'speaks' under </Silence>/</Response> protocol."""
    if not has_value(pred_value):
        return False
    raw = str(pred_value).strip()
    if not raw:
        return False

    # Pure-text protocol
    if raw.startswith('</Silence>') or raw.lower().startswith('</silence>'):
        return False
    # Treat standby as silence for streaming turn-taking.
    if raw.startswith('</Standby>') or raw.lower().startswith('</standby>'):
        return False
    if raw.startswith('</Response>') or raw.lower().startswith('</response>'):
        return True

    # JSON event package protocol
    obj = try_parse_json_pred(pred_value)
    if not obj:
        # Non-empty but no explicit protocol marker: treat as response
        return True
    events = obj.get('response_events', [])
    if not isinstance(events, list) or not events:
        return False
    any_nonempty = False
    all_silence_like = True
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ans = str(ev.get('answer', ev.get('raw', ''))).strip()
        if not ans:
            continue
        any_nonempty = True
        # Silence/standby markers mean "no response"
        if ans.startswith('</Silence>') or ans.lower().startswith('</silence>'):
            continue
        if ans.startswith('</Standby>') or ans.lower().startswith('</standby>'):
            continue
        if ans.startswith('</Response>') or ans.lower().startswith('</response>'):
            return True
        # Any other non-empty token counts as a response
        all_silence_like = False
    # If we only saw silence-like markers (or empty), treat as no response
    if any_nonempty and all_silence_like:
        return False
    # Events exist but didn't contain explicit markers; be conservative
    return not all_silence_like


def pred_trigger_times(pred_value: Any) -> list[float]:
    """Extract trigger times for pa from JSON prediction (response_events.time)."""
    obj = try_parse_json_pred(pred_value)
    if not obj:
        return []
    events = obj.get('response_events', [])
    if not isinstance(events, list):
        return []
    out: list[float] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        t = ev.get('time')
        try:
            out.append(float(t))
        except Exception:
            continue
    return out


def extract_pred_text_for_mdsg(row: pd.Series) -> dict[int, str]:
    """Map qa_idx -> predicted text for md/sg from prediction string (supports JSON event package)."""
    pred_val = row.get('prediction', '')
    if not has_value(pred_val):
        return {}
    raw = str(pred_val).strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    events = obj.get('response_events', [])
    if not isinstance(events, list):
        return {}
    out: dict[int, str] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        qi = ev.get('qa_idx')
        if qi is None:
            continue
        try:
            qi_int = int(qi)
        except Exception:
            continue
        ans = str(ev.get('answer', ev.get('raw', ''))).strip()
        if ans:
            out[qi_int] = ans
    return out

