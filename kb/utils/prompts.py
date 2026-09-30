"""Prompt building helpers."""

from __future__ import annotations

import config


# Style rules for every prompt that writes a question or an answer of the
# base, after Ilyakhov's «Пиши, сокращай». One block, so the prompts cannot
# drift apart; the model reads it, so it is in Russian.
WRITING_STYLE = """СТИЛЬ ТЕКСТА — «Пиши, сокращай» (Ильяхов)
Вопрос и ответ пиши просто, коротко и по делу. Сокращай слова, а не смысл.
- Одна мысль — одно предложение. Предложения короткие.
- Глагол вместо отглагольного существительного: «произведите проверку» →
  «проверьте», «осуществляется согласование» → «согласуйте».
- Без канцелярита и воды: «необходимо», «в данном случае», «является»,
  «осуществляется», «на сегодняшний день», «в целях», «данный».
- Без вводных слов, оценок и вежливости: «к сожалению», «обращаем внимание»,
  «просим», «будем рады».
- Конкретика вместо общих слов: название раздела, кнопки, шаблона, роли.
- Ответ не повторяет вопрос и начинается сразу с сути.
- Сокращая, не теряй факты, условия, числа, ссылки и названия.
Было: «Для осуществления заказа мебели необходимо произвести обращение к
сервис-менеджеру». Стало: «Чтобы заказать мебель, обратитесь к
сервис-менеджеру»."""


def render_prompt(template: str, **values: str | int) -> str:
    """Substitute <<KEY>> placeholders in a prompt template.

    Plain str.format cannot be used here: prompt templates contain the curly
    braces of the JSON output examples.
    """
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"<<{key.upper()}>>", str(value))
    return rendered


def format_categories() -> str:
    """Render the configured categories as a bullet list for a prompt."""
    return "\n".join(
        f"- {name} — {description}" for name, description in config.CATEGORIES.items()
    )


def format_category_names() -> str:
    """Render the configured category keys as a comma separated list."""
    return ", ".join(config.CATEGORIES)
