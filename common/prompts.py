"""Prompt templates shared by training and evaluation.

Base models get the plain format below; post-trained models will use their chat
template (see ROADMAP.md).
"""

QUESTION_TEMPLATE = "Question: {question}\nAnswer:"

# Generation is cut at the first of these (base models keep inventing new questions).
STOP_STRINGS = ["\n\n", "\nQuestion:"]


def qa_prompt(question: str) -> str:
    return QUESTION_TEMPLATE.format(question=question)


def completion(solution: str) -> str:
    return " " + solution


def few_shot_prefix(exemplars) -> str:
    return "".join(qa_prompt(e["question"]) + completion(e["solution"]) + "\n\n" for e in exemplars)


def truncate(text: str) -> str:
    cut = len(text)
    for s in STOP_STRINGS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()
