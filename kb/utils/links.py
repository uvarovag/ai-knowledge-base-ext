"""Links in texts the model rewrites: found, masked before the call, restored after.

GigaChat cannot copy a long percent-encoded link: it drops or swaps a few
characters and the link leads nowhere. So a model never writes a link. Every
link of the source is replaced by a short placeholder ("[ссылка-1]"), the
model moves placeholders where the links belong, and the code puts the exact
source links back.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

URL_PATTERN = re.compile(r"https?://[^\s«»\"<>]+")
URL_TRAILING_PUNCTUATION = ".,;:!?)"

PLACEHOLDER = "[ссылка-{number}]"
# Tolerates the ways a model re-types a placeholder: "[ссылка 1]", "[Ссылка_1]".
PLACEHOLDER_PATTERN = re.compile(r"\[\s*ссылка[\s_-]*(\d+)\s*\]", flags=re.IGNORECASE)

# Fields of a model reply that hold text with placeholders.
TEXT_FIELDS: tuple[str, ...] = ("question", "answer")


def find_links(text: str) -> list[str]:
    """Return the links of a text, without the punctuation that ends a sentence."""
    return [
        link.rstrip(URL_TRAILING_PUNCTUATION) for link in URL_PATTERN.findall(text)
    ]


def mask_links(texts: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """Replace every link with a numbered placeholder, the same link with one number.

    Returns the masked texts and placeholder number -> link.
    """
    numbers: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        link = match.group(0).rstrip(URL_TRAILING_PUNCTUATION)
        tail = match.group(0)[len(link) :]
        if link not in numbers:
            numbers[link] = str(len(numbers) + 1)
        return PLACEHOLDER.format(number=numbers[link]) + tail

    masked = [URL_PATTERN.sub(replace, text) for text in texts]
    return masked, {number: link for link, number in numbers.items()}


def unmask_links(reply: dict[str, Any], links: dict[str, str]) -> dict[str, Any]:
    """Put the source links back into the text fields of a model reply, in place.

    A placeholder the source did not have is left as is, so the entry
    validation rejects it rather than a link being guessed.
    """
    for field in TEXT_FIELDS:
        if isinstance(reply.get(field), str):
            reply[field] = PLACEHOLDER_PATTERN.sub(
                lambda match: links.get(match.group(1), match.group(0)), reply[field]
            )
    return reply
