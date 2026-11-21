"""Prompt construction for the two formats in common.models.

plain: "Question: {q}\nAnswer:" (+ k exemplars before it), generation cut at STOP_STRINGS.
chat:  the model's chat template; the user turn is the question plus CHAT_INSTRUCTION.
       Few-shot exemplars become earlier user/assistant turns. Stops at EOS only, since
       chat models write multi-paragraph answers that "\n\n" would cut off.
"""

QUESTION_TEMPLATE = "Question: {question}\nAnswer:"

# Generation is cut at the first of these (base models keep inventing new questions).
STOP_STRINGS = ["\n\n", "\nQuestion:"]

# Asks for the same final-answer phrase our reference solutions end with, so strict
# accuracy means the same thing for every model.
CHAT_INSTRUCTION = 'Solve the problem step by step. End your response with "The answer is N." where N is the final number.'


def qa_prompt(question: str) -> str:
    return QUESTION_TEMPLATE.format(question=question)


def completion(solution: str) -> str:
    return " " + solution


def few_shot_prefix(exemplars) -> str:
    return "".join(qa_prompt(e["question"]) + completion(e["solution"]) + "\n\n" for e in exemplars)


def chat_messages(question: str, exemplars=()) -> list:
    msgs = []
    for e in exemplars:
        msgs += [{"role": "user", "content": f"{e['question']}\n{CHAT_INSTRUCTION}"},
                 {"role": "assistant", "content": e["solution"]}]
    return msgs + [{"role": "user", "content": f"{question}\n{CHAT_INSTRUCTION}"}]


def build_prompt(fmt, question, exemplars, tok):
    """Return the full prompt string for one question."""
    if fmt == "plain":
        return few_shot_prefix(exemplars) + qa_prompt(question)
    if fmt == "chat":
        return tok.apply_chat_template(chat_messages(question, exemplars), tokenize=False,
                                       add_generation_prompt=True)
    raise ValueError(fmt)


def truncate(text: str, fmt: str = "plain") -> str:
    if fmt != "plain":
        return text.strip()
    cut = len(text)
    for s in STOP_STRINGS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()
