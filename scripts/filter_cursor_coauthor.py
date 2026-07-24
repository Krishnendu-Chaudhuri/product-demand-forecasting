import re

cleaned = re.sub(
    rb"(?m)^Co-authored-by:\s*Cursor\s*<cursoragent@cursor\.com>\s*\n?",
    b"",
    message,
)
cleaned = re.sub(
    rb"(?m)^Co-authored-by:.*(?:cursoragent|Cursor Agent).*\n?",
    b"",
    cleaned,
    flags=re.IGNORECASE,
)
return cleaned
