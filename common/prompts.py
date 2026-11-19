"""Prompt templates shared by training and evaluation.

Every condition sees the same surface format. The only thing that changes between
conditions is what is removed (the question, for response-only) or corrupted.
"""

QUESTION_TEMPLATE = "Question: {question}\nAnswer:"

# Response-only tuning keeps the template tokens and drops the question, analogous to
# Hewitt et al. removing the instruction but keeping the response.
RESPONSE_ONLY_PROMPT = "Answer:"

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
