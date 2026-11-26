"""Validate localized positive Horn derivations and reuse the suite's language-channel diagnostics."""

from itertools import permutations
import re

import closed_language as language


DETAIL_KEYS = language.DETAIL_KEYS | {
    "proof_steps_supported",
    "proof_steps_relevant",
    "proof_concludes_goal",
    "proof_answer_agree",
}
ATOM = re.compile(r"[A-Z][a-z]{1,15}/[A-Z][a-z]{1,15}")
FORMS = {
    "de": {
        "source": "Die Quelle {key} belegt die Aussage {head}",
        "step": "Aus {body} ergibt sich nach Regel {key} die Aussage {head}",
        "answer": "Damit ist für den Fund {name} das Merkmal {mark} belegt",
        "and": " und ",
    },
    "en": {
        "source": "The source {key} establishes the claim {head}",
        "step": "From {body} it follows by rule {key} that {head} holds",
        "answer": "The record therefore establishes the mark {mark} for the find {name}",
        "and": " and ",
    },
}


def closure(expected):
    established = set(expected["sources"].values())
    while True:
        consequences = {
            rule["then"]
            for rule in expected["rules"].values()
            if all(x in established for x in rule["if"])
        }
        if consequences <= established:
            return established
        established.update(consequences)


def validate_expected(expected):
    if type(expected) is not dict or set(expected) != {"sources", "rules", "goals"}:
        return False
    sources, rules, goals = (expected[k] for k in ("sources", "rules", "goals"))

    def atom(x):
        return isinstance(x, str) and ATOM.fullmatch(x) is not None

    if (
        type(sources) is not dict
        or not 1 <= len(sources) <= 32
        or type(rules) is not dict
        or not 1 <= len(rules) <= 40
    ):
        return False
    if not all(
        isinstance(k, str) and re.fullmatch(r"F[0-9]{1,3}", k) and atom(v)
        for k, v in sources.items()
    ):
        return False
    if (
        type(goals) is not list
        or len(goals) != 2
        or not all(map(atom, goals))
        or len(set(goals)) != 2
    ):
        return False
    if goals[0].split("/")[0] != goals[1].split("/")[0] or set(goals) & set(
        sources.values()
    ):
        return False
    for key, rule in rules.items():
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"R[0-9]{1,3}", key)
            or type(rule) is not dict
            or set(rule) != {"if", "then"}
        ):
            return False
        body = rule["if"]
        if (
            type(body) is not list
            or not 1 <= len(body) <= 3
            or not all(map(atom, body))
            or len(set(body)) != len(body)
            or not atom(rule["then"])
        ):
            return False
    return len(set(goals) & closure(expected)) == 1


def statements(expected, lang):
    forms = FORMS[lang]
    allowed = {}
    for key, head in expected["sources"].items():
        statement = forms["source"].format(key=key, head=head).casefold()
        allowed[statement] = (head, ())
    for key, rule in expected["rules"].items():
        head = rule["then"]
        for order in permutations(rule["if"]):
            phrase = forms["and"].join(order)
            statement = forms["step"].format(body=phrase, key=key, head=head).casefold()
            allowed[statement] = (head, tuple(rule["if"]))
    return allowed


def inspect(expected, lang, proof):
    allowed = statements(expected, lang)
    parts = language.clauses(proof)
    used = {}
    supported = bool(parts) and len(parts) <= len(expected["rules"]) + len(
        expected["sources"]
    )
    last = None
    for sentence in parts:
        if sentence not in allowed:
            supported = False
            break
        head, body = allowed[sentence]
        if head in used or not all(atom in used for atom in body):
            supported = False
            break
        used[head] = body
        last = head
    goal = supported and last in expected["goals"]
    relevant = False
    if goal:
        needed = set()
        pending = [last]
        while pending:
            atom = pending.pop()
            if atom not in needed:
                needed.add(atom)
                pending.extend(used[atom])
        relevant = needed == set(used)
    return supported, relevant, goal, last, parts


def conclusion(lang, atom):
    name, mark = atom.split("/")
    return FORMS[lang]["answer"].format(name=name, mark=mark)


def score(item, reply):
    from hidden import strict_json

    lang = item["lang"]
    expected = item["expected"]
    justification_field, answer_field = language.FIELDS[lang]
    try:
        parsed = strict_json(reply.get("content", ""))
    except (TypeError, ValueError, RecursionError):
        parsed = None
    proof = ""
    answer = ""
    if isinstance(parsed, dict):
        if isinstance(parsed.get(justification_field), str):
            proof = parsed[justification_field]
        if isinstance(parsed.get(answer_field), str):
            answer = parsed[answer_field]
    valid, relevant, goal, last, parts = inspect(expected, lang, proof)
    # Only after replay succeeds does the shared report grader accept these particular
    # grounded statements. It continues to own all language and optional-channel denominators.
    certificate = [{s} for s in parts] if valid and relevant and goal else [set()]
    entailed = set(expected["goals"]) & closure(expected)
    finals = [{conclusion(lang, atom).casefold() for atom in entailed}]
    result = language.score_report(item, reply, certificate, finals)
    agree = bool(
        goal
        and language.checked_statements(answer, [{conclusion(lang, last).casefold()}])
    )
    result["detail"].update(
        proof_steps_supported=float(valid),
        proof_steps_relevant=float(relevant),
        proof_concludes_goal=float(goal),
        proof_answer_agree=float(agree),
    )
    result["correct"] = bool(result["correct"] and agree)
    result["score"] = float(result["correct"])
    return result
