"""Links in texts the model rewrites: found, masked before the call, restored after.

GigaChat cannot copy a long percent-encoded link: it drops or swaps a few
characters and the link leads nowhere. So a model never writes a link. Every
link of the source is replaced by a short placeholder ("[ссылка-1]"), the
model moves placeholders where the links belong, and the code puts the exact
source links back. A placeholder the model made up has no link to go back
to: the model is asked again, told which placeholders exist (ask_with_links).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import config
from kb.utils.logs import logger

URL_PATTERN = re.compile(r"https?://[^\s«»\"<>]+")
URL_TRAILING_PUNCTUATION = ".,;:!?)"

PLACEHOLDER = "[ссылка-{number}]"
# Tolerates the ways a model re-types a placeholder: "[ссылка 1]", "[Ссылка_1]",
# and declines it to fit the sentence: "по [ссылке-1]", "перейдите по [ссылку-1]".
PLACEHOLDER_PATTERN = re.compile(
    r"\[\s*ссылк[а-яё]*[\s_-]*(\d+)\s*\]", flags=re.IGNORECASE
)

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


def find_placeholders(reply: dict[str, Any]) -> list[str]:
    """Return the placeholders left in the text fields of a reply."""
    return [
        match.group(0)
        for field in TEXT_FIELDS
        if isinstance(reply.get(field), str)
        for match in PLACEHOLDER_PATTERN.finditer(reply[field])
    ]


def placeholder_hint(leftovers: list[str], links: dict[str, str]) -> str:
    """Tell the model which placeholders it made up and which ones exist.

    Appended to the user prompt of a repeated call: at temperature 0 the same
    prompt returns the same reply, so the repeat has to say what was wrong.
    """
    made_up = ", ".join(dict.fromkeys(leftovers))
    if not links:
        return (
            f"\n\nВНИМАНИЕ: в прошлом ответе были метки {made_up}, но в записи "
            "нет ссылок. Меток не пиши."
        )
    allowed = ", ".join(PLACEHOLDER.format(number=number) for number in links)
    return (
        f"\n\nВНИМАНИЕ: в прошлом ответе были метки {made_up}, которых нет в "
        f"записи. Используй только метки {allowed} — без изменений и без склонения."
    )


def ask_with_links(
    ask: Callable[[str], dict[str, Any] | None],
    links: dict[str, str],
    label: str,
) -> dict[str, Any] | None:
    """Call the model and put the links back, asking again on a made-up placeholder.

    ask takes a hint to append to the user prompt, empty on the first call.
    Up to config.LINK_PLACEHOLDER_ATTEMPTS calls; a reply that still holds an
    unknown placeholder is returned as is, and the entry validation rejects it.
    """
    hint = ""
    reply: dict[str, Any] | None = None
    for attempt in range(1, config.LINK_PLACEHOLDER_ATTEMPTS + 1):
        reply = ask(hint)
        if reply is None:
            return None
        unmask_links(reply, links)
        leftovers = find_placeholders(reply)
        if not leftovers:
            return reply
        logger.warning(
            "%s: unknown link placeholder %s, attempt %d/%d",
            label,
            ", ".join(dict.fromkeys(leftovers)),
            attempt,
            config.LINK_PLACEHOLDER_ATTEMPTS,
        )
        hint = placeholder_hint(leftovers, links)
    return reply
