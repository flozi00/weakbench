"""Agentic task of the quality suite (quality.py imports TASKS): multi-step tool use in a simulated, stateful world.

tool_env  A small world is generated per id (contacts, a calendar, flights, an inbox). The model gets eight tools
          (search contacts, list / search / create events, search / book flights, read the inbox, send an email) and
          an errand in German or English. Five kinds: book the cheapest flight and mail the reference; find a free
          slot and create a meeting; book a flight under constraints (budget, arrival before a time); a conditional
          (book if a flight is under budget, else mail that there is none); answer an inbox question with a date
          from the calendar. The world is Python dicts — nothing the model writes is executed — and the score is
          the FINAL STATE: the right booking, the right event, the right mail to the right address, and no side
          effects (extra bookings, events or mails).
          `underspecified` leaves out the date: correct is to book nothing and ask for it.
          A sixth kind, `flaky`, makes the first booking attempt fail with a transient error: the model has to
          retry instead of giving up or claiming success.
          correct = the state is exactly right; sub-metrics `side_effects`, `tool_rounds`, `ok_<kind>`.
"""
import datetime
import json
import random
import re

NAMES = ["Anna Becker", "Ben Wolf", "Clara Neumann", "David Krüger", "Emma Hofmann", "Felix Schulz", "Greta Lang",
         "Hannes Vogel"]
CITIES = ["Berlin", "Munich", "Hamburg", "Vienna", "Zurich", "Paris", "Rome", "Madrid"]
CITY_DE = {"Munich": "München", "Vienna": "Wien", "Zurich": "Zürich", "Rome": "Rom"}
CITY_KEY = {c.lower(): c for c in CITIES} | {de.lower(): en for en, de in CITY_DE.items()}
TOOLS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {
        "type": "object", "properties": p, "required": list(p)}}}
    for n, d, p in [
        ("search_contacts", "Find a contact's email address by (part of) their name.",
         {"name": {"type": "string"}}),
        ("list_events", "List the calendar events on a date (YYYY-MM-DD).", {"date": {"type": "string"}}),
        ("create_event", "Create a calendar event. time is HH:MM (24 h), duration in minutes.",
         {"date": {"type": "string"}, "time": {"type": "string"}, "duration": {"type": "integer"},
          "title": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"},
                                                     "description": "email addresses"}}),
        ("search_flights", "Search flights between two cities on a date (YYYY-MM-DD).",
         {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}),
        ("book_flight", "Book a flight by its id. Returns a booking reference.", {"flight_id": {"type": "string"}}),
        ("send_email", "Send an email.", {"to": {"type": "string"}, "subject": {"type": "string"},
                                          "body": {"type": "string"}}),
        ("read_inbox", "List the emails in the inbox (sender, subject, body).", {}),
        ("search_events", "Find calendar events whose title contains the query (any date).",
         {"query": {"type": "string"}}),
    ]]
SYSTEM = {"en": "You are an assistant that completes errands with the tools provided. Today is {today}. Do exactly "
                "what is asked, nothing more. When the errand is done, reply with a short summary.",
          "de": "Du bist ein Assistent, der Aufgaben mit den bereitgestellten Werkzeugen erledigt. Heute ist der "
                "{today}. Tu genau das, worum gebeten wird, nicht mehr. Wenn die Aufgabe erledigt ist, antworte mit "
                "einer kurzen Zusammenfassung."}
TASK_TEXT = {
    "cheapest": {"en": "Book the cheapest flight from {o} to {d} on {date}, then email the booking reference to "
                       "{who}.",
                 "de": "Buche den günstigsten Flug von {o} nach {d} am {date} und schicke die Buchungsnummer danach "
                       "per E-Mail an {who}."},
    "meeting": {"en": "Schedule a {dur}-minute meeting titled \"{title}\" with {who} on {date}, starting on the full "
                      "hour between 09:00 and 16:00, at a time when my calendar is free. Invite {who}.",
                "de": "Plane am {date} ein {dur}-minütiges Meeting mit dem Titel \"{title}\" mit {who}, Beginn zur "
                      "vollen Stunde zwischen 09:00 und 16:00, zu einer Zeit, in der mein Kalender frei ist. Lade "
                      "{who} ein."},
    "constrained": {"en": "Book a flight from {o} to {d} on {date} that costs at most {budget} euros and arrives "
                          "before {arr}. If several qualify, take the cheapest. No email needed.",
                    "de": "Buche einen Flug von {o} nach {d} am {date}, der höchstens {budget} Euro kostet und vor "
                          "{arr} Uhr ankommt. Wenn mehrere passen, nimm den günstigsten. Keine E-Mail nötig."},
    "inbox": {"en": "Check my inbox: someone asked when my next \"{title}\" is. Reply to that person by email with "
                    "the date and start time of that event.",
              "de": "Schau in meinen Posteingang: Jemand hat gefragt, wann mein nächstes \"{title}\" ist. Antworte "
                    "dieser Person per E-Mail mit Datum und Uhrzeit dieses Termins."},
    "underspecified": {"en": "Book me the cheapest flight from {o} to {d}.",
                       "de": "Buche mir den günstigsten Flug von {o} nach {d}."},
    "flaky": {"en": "Book the cheapest flight from {o} to {d} on {date}. No email needed.",
              "de": "Buche den günstigsten Flug von {o} nach {d} am {date}. Keine E-Mail nötig."},
    "conditional": {"en": "If there is a flight from {o} to {d} on {date} for at most {budget} euros, book the "
                          "cheapest one. If there is none, email {who} that the trip has to be postponed, and book "
                          "nothing.",
                    "de": "Wenn es am {date} einen Flug von {o} nach {d} für höchstens {budget} Euro gibt, buche den "
                          "günstigsten. Wenn nicht, schreibe {who} per E-Mail, dass die Reise verschoben werden muss, "
                          "und buche nichts."},
}
KINDS = ("cheapest", "meeting", "constrained", "conditional", "inbox", "flaky", "underspecified")


def email_of(name):
    first, last = name.lower().replace("ü", "ue").split()
    return f"{first}.{last}@example.org"


def make_world(i):
    r = random.Random(f"tool_env:{i}")
    today = datetime.date(2026, 3, 2) + datetime.timedelta(days=r.randrange(200))
    date = (today + datetime.timedelta(days=r.randint(3, 20))).isoformat()
    o, d = r.sample(CITIES, 2)
    flights = []
    for k in range(r.randint(4, 6)):
        dep = r.randint(6, 19) * 60 + r.choice([0, 15, 30, 45])
        dur = r.randint(70, 160)
        flights.append({"id": f"F{r.randint(100, 999)}{k}", "origin": o, "destination": d, "date": date,
                        "depart": f"{dep // 60:02d}:{dep % 60:02d}",
                        "arrive": f"{(dep + dur) // 60:02d}:{(dep + dur) % 60:02d}", "price": r.randint(60, 420)})
    prices = sorted(f["price"] for f in flights)
    if len(set(prices)) != len(prices):  # a tie makes "the cheapest" ambiguous
        return make_world(f"{i}b")
    # decoys on another date and the reverse route, cheaper than all: the search arguments matter
    flights += [{**flights[0], "id": "X" + flights[0]["id"][1:], "date": today.isoformat(), "price": 30},
                {**flights[1], "id": "Y" + flights[1]["id"][1:], "origin": d, "destination": o, "price": 35}]
    events = []
    for h in sorted(r.sample(range(9, 17), r.randint(4, 6))):  # busy full hours on the meeting date
        events.append({"date": date, "time": f"{h:02d}:00", "duration": 60, "title": r.choice(
            ["Standup", "Review", "Call", "Workshop", "1:1", "Planning"]), "attendees": []})
    contacts = r.sample(NAMES, 5)
    return {"r": r, "today": today.isoformat(), "date": date, "o": o, "d": d, "flights": flights, "events": events,
            "contacts": {n: email_of(n) for n in contacts}, "who": contacts[0]}


def tool_runtime(world, state):
    def run(name, arguments):
        a = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
        if name == "search_contacts":
            q = str(a.get("name", "")).lower()
            hits = [{"name": n, "email": e} for n, e in world["contacts"].items() if q and q in n.lower()]
            return json.dumps(hits)
        if name == "list_events":
            return json.dumps([e for e in world["events"] + state["events"] if e["date"] == a.get("date")])
        if name == "search_flights":
            return json.dumps([{k: f[k] for k in ("id", "depart", "arrive", "price")} for f in world["flights"]
                               if f["origin"] == CITY_KEY.get(str(a.get("origin", "")).strip().lower())
                               and f["destination"] == CITY_KEY.get(str(a.get("destination", "")).strip().lower())
                               and f["date"] == a.get("date")])
        if name == "book_flight":
            if world.get("flaky") and not state.get("failed_once"):  # the first booking attempt fails
                state["failed_once"] = True
                return "error: booking service temporarily unavailable, please try again"
            f = next((f for f in world["flights"] if f["id"] == a.get("flight_id")), None)
            if not f:
                return "error: unknown flight id"
            ref = f"BK{len(state['bookings']) + 1}{f['id'][-3:]}"
            state["bookings"].append({"flight_id": f["id"], "ref": ref})
            return json.dumps({"booking_reference": ref})
        if name == "create_event":
            ev = {"date": a.get("date"), "time": a.get("time"), "duration": a.get("duration"),
                  "title": a.get("title"), "attendees": list(a.get("attendees") or [])}
            state["events"].append(ev)
            return json.dumps({"created": True})
        if name == "search_events":
            q = str(a.get("query", "")).lower()
            return json.dumps([e for e in world["events"] + state["events"] if q and q in str(e["title"]).lower()])
        if name == "read_inbox":
            return json.dumps(world.get("inbox", []))
        if name == "send_email":
            state["emails"].append({k: str(a.get(k, "")) for k in ("to", "subject", "body")})
            return json.dumps({"sent": True})
        return f"error: unknown tool {name}"
    return run


def expected(world, kind):
    """The task's parameters and the state it must end in (bookings: flight ids; mail: recipient + must-contain)."""
    r = world["r"]
    route = [f for f in world["flights"] if f["date"] == world["date"] and f["origin"] == world["o"]
             and f["destination"] == world["d"]]
    cheapest = min(route, key=lambda f: f["price"])
    p = {"o": world["o"], "d": world["d"], "date": world["date"], "who": world["who"]}
    if kind == "cheapest":
        return p, {"book": [cheapest["id"]], "mail": "ref"}
    if kind == "underspecified":  # no date given: the right move is to ask, not to guess one and book
        return p, {"book": [], "mail": None, "ask": True}
    if kind == "flaky":
        world["flaky"] = True
        return p, {"book": [cheapest["id"]], "mail": None}
    if kind == "constrained":
        arr = f"{r.randint(9, 20):02d}:00"
        ok = [f for f in route if f["arrive"] < arr]
        if not ok:
            return None
        budget = sorted(f["price"] for f in ok)[len(ok) // 2] if len(ok) > 1 else ok[0]["price"] + 10
        fits = [f for f in ok if f["price"] <= budget]
        return p | {"budget": budget, "arr": arr}, {"book": [min(fits, key=lambda f: f["price"])["id"]], "mail": None}
    if kind == "conditional":
        if r.random() < 0.5:  # the cheaper branch exists
            return p | {"budget": cheapest["price"] + r.randint(0, 40)}, {"book": [cheapest["id"]], "mail": None}
        return p | {"budget": cheapest["price"] - r.randint(1, 30)}, {"book": [], "mail": "postpone"}
    if kind == "inbox":  # a uniquely titled event on a later date; the asker and a distractor mail in the inbox
        title = r.choice(["Board review", "Quarterly planning", "Design sync", "Budget check"])
        when = (datetime.date.fromisoformat(world["date"]) + datetime.timedelta(days=r.randint(1, 9))).isoformat()
        hour = f"{r.randint(9, 16):02d}:{r.choice(['00', '30'])}"
        world["events"] = world["events"] + [{"date": when, "time": hour, "duration": 60, "title": title,
                                              "attendees": []}]
        asker, other = r.sample(sorted(world["contacts"]), 2)
        world["inbox"] = [{"from": world["contacts"][other], "subject": "Lunch?", "body": "Lunch next week?"},
                          {"from": world["contacts"][asker], "subject": title, "body": f"When is the next {title}?"}]
        return p | {"title": title}, {"book": [], "mail": "inbox", "to": world["contacts"][asker],
                                      "date": when, "time": hour}
    busy = {e["time"] for e in world["events"]}
    free = [f"{h:02d}:00" for h in range(9, 17) if f"{h:02d}:00" not in busy]
    if not free:
        return None
    return p | {"dur": 60, "title": r.choice(["Budget 2027", "Kick-off", "Roadmap", "Hiring plan"])}, \
        {"book": [], "mail": None, "event": free}


def load_tool_env(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 200):
        i += 1
        kind = KINDS[i % len(KINDS)]
        world = make_world(i)
        exp = expected(world, kind)
        if exp is None:  # this draw has no valid answer: next id (identically in both languages)
            continue
        params, want = exp
        state = {"bookings": [], "events": [], "emails": []}
        city = (lambda c: CITY_DE.get(c, c)) if lang == "de" else (lambda c: c)  # noqa: E731
        text = TASK_TEXT[kind][lang].format(**params | {"o": city(params["o"]), "d": city(params["d"])})
        items.append({"id": str(i), "lang": lang, "kind": kind, "want": want, "params": params, "world": world,
                      "state": state, "tools": TOOLS, "tool_exec": tool_runtime(world, state), "max_rounds": 8,
                      "messages": [{"role": "system", "content": SYSTEM[lang].format(today=world["today"])},
                                   {"role": "user", "content": text}]})
    return items


def score_tool_env(item, reply):
    st, want, world = item["state"], item["want"], item["world"]
    booked = [b["flight_id"] for b in st["bookings"]]
    ok_book = booked == want["book"]
    who_mail = world["contacts"][world["who"]]
    mails = st["emails"]
    if want["mail"] == "ref":
        ok_mail = len(mails) == 1 and mails[0]["to"].strip().lower() == who_mail and bool(st["bookings"]) and \
            st["bookings"][-1]["ref"] in mails[0]["subject"] + mails[0]["body"]
    elif want["mail"] == "inbox":  # the date in ISO or German/English day-month form, and the time
        text = (mails[0]["subject"] + " " + mails[0]["body"]) if mails else ""
        d = datetime.date.fromisoformat(want["date"])
        dates = {want["date"], f"{d.day}.{d.month}.", f"{d.day:02d}.{d.month:02d}.", d.strftime("%B ") + str(d.day),
                 f"{d.day} " + d.strftime("%B")}
        ok_mail = len(mails) == 1 and mails[0]["to"].strip().lower() == want["to"] and \
            any(x in text for x in dates) and want["time"].lstrip("0") in text
    elif want["mail"] == "postpone":
        ok_mail = len(mails) == 1 and mails[0]["to"].strip().lower() == who_mail
    else:
        ok_mail = not mails
    if "event" in want:
        evs = st["events"]
        ok_event = False
        if len(evs) == 1:
            ev = evs[0]
            t = re.match(r"\s*(\d{1,2}):(\d{2})", str(ev["time"] or ""))  # "9:00", "09:00:00" are the same time
            dur = re.match(r"\s*(\d+)", str(ev["duration"] or ""))
            ok_event = (ev["date"] == world["date"] and bool(t) and f"{int(t[1]):02d}:{t[2]}" in want["event"]
                        and bool(dur) and int(dur[1]) == 60
                        and who_mail in [str(x).strip().lower() for x in ev["attendees"]])
    else:
        ok_event = not st["events"]
    side = max(len(booked) - len(want["book"]), 0) + max(len(mails) - (1 if want["mail"] else 0), 0) + \
        max(len(st["events"]) - (1 if "event" in want else 0), 0)
    ok = ok_book and ok_mail and ok_event
    if want.get("ask"):  # nothing done, and the reply asks back (a question mark and a date word)
        asked = "?" in reply["content"] and re.search(r"(?i)\b(date|datum|wann|when|tag|day)\b", reply["content"]) is not None
        ok = ok and asked
    return {"correct": ok, "score": float(ok), "pred": json.dumps(st)[:300], "gold": json.dumps(want)[:200],
            "detail": {"kind": item["kind"], f"ok_{item['kind']}": float(ok), "side_effects": float(side > 0),
                       "ok_book": float(ok_book), "ok_mail": float(ok_mail), "ok_event": float(ok_event)}}


TASKS = {"tool_env": {"langs": ("en", "de"), "load": load_tool_env, "score": score_tool_env, "think": False,
                      "max_tokens": 2048, "counts": {"quick": {"en": 80, "de": 80}, "full": {"en": 200, "de": 200}},
                      "lang_check": False, "junk_check": False}}
