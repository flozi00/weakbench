import json
import unittest
import closed_gamebook as game


def fixture(lang):
    labels = (
        ("Mohn", "Ocker", "Zeder", "Klee", "Harz")
        if lang == "de"
        else ("poppy", "ochre", "cedar", "clover", "resin")
    )
    return {
        "layers": [["S"], ["U", "V"], ["E", "F"]],
        "motifs": dict(zip("SUVEF", labels)),
        "cues": [["oben", "unten"], ["still", "rasch"]]
        if lang == "de"
        else [["upper", "lower"], ["calm", "swift"]],
        "routes": [[[0, 0], "E"], [[1, 0], "F"]],
        "callbacks": [["E", "U"]],
    }


def witness(lang):
    s = fixture(lang)
    mo = s["motifs"]
    if lang == "de":
        text = {
            "S": "Auf der Einladung stand Mohn, und der Bote trug einen grünen Hut.",
            "U": "Das Wort Ocker tanzte über die Leinwand, als sich der Vorhang hob.",
            "V": "Im Pausenraum las ich Zeder auf der Kanne mit dem letzten Schluck.",
            "E": "Ich fand Klee auf dem Programm und erinnerte mich an Ocker im Saal.",
            "F": "Draußen buchstabierte ein Kind Harz auf seinem frisch beschlagenen Fenster.",
        }
    else:
        text = {
            "S": "The invitation said poppy and the messenger wore a green hat.",
            "U": "The name ochre danced across the canvas as the curtain opened.",
            "V": "During the interval I read cedar on the nearly empty kettle.",
            "E": "I found clover on the programme and recalled the earlier ochre mark.",
            "F": "Outside a child traced resin across the foggy window of the bus.",
        }
    result = {k: {"text": v, "links": []} for k, v in text.items()}
    for k, dests, stage in (("S", "UV", 0), ("U", "EF", 1), ("V", "FF", 1)):
        for i, (cue, end) in enumerate(zip(s["cues"][stage], dests)):
            label = game.FORMS[lang][i].format(cue=cue, motif=mo[end])
            result[k]["links"].append({"label": label, "to": end})
    return {"cards": result}


def sample_prompt(lang, spec=None):
    spec = fixture(lang) if spec is None else spec
    lead = (
        "Schreibe die Absätze für die Karten und verbinde sie nach den beschriebenen Regeln."
        if lang == "de"
        else "Write the paragraphs for these passages and connect them using the stated rules."
    )
    return lead + "\n" + game.table(spec, lang)


class GamebookTests(unittest.TestCase):
    def score(self, answer, lang="en"):
        return game.score(
            {
                "lang": lang,
                "expected": fixture(lang),
                "messages": [{"role": "user", "content": sample_prompt(lang)}],
            },
            {"content": json.dumps(answer)},
        )

    def test_fresh_unrestricted_paragraphs_and_link_order(self):
        for lang in ("de", "en"):
            self.assertTrue(game.validate_expected(fixture(lang)))
            a = witness(lang)
            for card in a["cards"].values():
                card["links"].reverse()
            r = self.score(a, lang)
            self.assertTrue(r["correct"])
            self.assertEqual(r["detail"]["book_link_promises"], 1)
            a["cards"]["F"]["text"] = (
                "Ein Händler rief Harz durch die Nacht und verlor dabei seinen gestreiften Schal."
                if lang == "de"
                else "A trader called resin into the night and lost his striped scarf in the wind."
            )
            self.assertTrue(self.score(a, lang)["correct"])

    def test_all_routes_including_unmentioned_readings(self):
        a = witness("en")
        # F is reachable from U, but U is absent on a different path to F.
        a["cards"]["F"]["text"] += " I remember that ochre mark now."
        r = self.score(a)
        self.assertFalse(r["correct"])
        self.assertLess(r["detail"]["book_motif_continuity"], 1)
        self.assertEqual(r["detail"]["book_reader_promises"], 1)

    def test_choice_signpost_and_visible_cue(self):
        a = witness("en")
        a["cards"]["S"]["links"][0]["label"] = "Read upper and call for cedar"
        self.assertEqual(
            self.score(a)["detail"]["book_link_promises"], 0.8333333333333334
        )
        a["cards"]["S"]["links"][0]["label"] = (
            "Read upper and lower before the ochre door"
        )
        self.assertFalse(self.score(a)["correct"])
        a = witness("en")
        a["cards"]["S"]["links"][0]["to"] = "S"
        self.assertFalse(self.score(a)["correct"])

    def test_token_boundaries_and_open_prose(self):
        for text in (
            "upper42",
            "x_upper",
            "upper٤٢",
            "upper‿x",
            "upper’s",
            "upper's",
            "upper\u200dword",
            "upper\u0301code",
        ):
            self.assertNotIn("upper", game.words(text))
        self.assertEqual(game.words("«Flöte»"), game.words("«Flo\u0308te»"))
        self.assertIn("upper", game.words("(upper), then listen."))
        for apostrophe in ("'", "’", "ʼ"):
            self.assertIn("upper", game.words(apostrophe + "upper" + apostrophe))
            self.assertNotIn("upper", game.words("upper" + apostrophe + "s"))
        for lang in ("de", "en"):
            spec = fixture(lang)
            key = spec["layers"][0][0]
            token = spec["motifs"][key]
            for suffix in ("_code", "42", "٣", "‿work", "’s"):
                a = witness(lang)
                a["cards"][key]["text"] = a["cards"][key]["text"].replace(
                    token, token + suffix
                )
                self.assertFalse(self.score(a, lang)["correct"])
            a = witness(lang)
            a["cards"][key]["links"][0]["label"] += (
                " Dieses Ziel erreiche ich niemals."
                if lang == "de"
                else " I never reach that destination."
            )
            self.assertFalse(self.score(a, lang)["correct"])

    def test_visible_and_observed_language_channels(self):
        probes = {
            "de": "Die Regeln sind klar und wir können den nächsten Hinweis lesen.",
            "en": "The rules are clear and we can read the next sign.",
        }
        for lang, other in (("de", "en"), ("en", "de")):
            spec = fixture(lang)
            book = json.dumps(witness(lang), ensure_ascii=False)
            req = sample_prompt(lang)
            item = {
                "expected": spec,
                "lang": lang,
                "messages": [{"role": "user", "content": req}],
            }
            checked = game.score(item, {"content": book, "reasoning": probes[lang]})
            self.assertTrue(checked["correct"])
            self.assertEqual(checked["detail"]["correct_with_all_channels"], 1)
            absent = game.score(item, {"content": book})
            self.assertTrue(absent["correct"])
            self.assertEqual(absent["detail"]["correct_with_all_channels"], 0)
            self.assertNotIn("reasoning_language_ok", absent["detail"])
            for trace in (probes[other], "1234"):
                bad = game.score(item, {"content": book, "reasoning": trace})
                self.assertFalse(bad["correct"])
                self.assertEqual(bad["detail"]["book_playable"], 1)
                self.assertEqual(bad["detail"]["all_channels_verified"], 0)
            combined = game.score(
                item,
                {
                    "content": book,
                    "reasoning": probes[lang],
                    "reasoning_content": probes[other],
                },
            )
            self.assertFalse(combined["correct"])
            for extra in (probes[other], "12345"):
                segmented = game.score(
                    item, {"content": book, "reasoning_parts": [probes[lang], extra]}
                )
                self.assertFalse(segmented["correct"])
                self.assertEqual(segmented["detail"]["all_channels_verified"], 0)
            segmented = game.score(
                item, {"content": book, "reasoning_parts": [probes[lang], probes[lang]]}
            )
            self.assertEqual(segmented["detail"]["correct_with_all_channels"], 1)
            item["messages"][0]["content"] = (
                probes[other] + "\n" + game.table(spec, lang)
            )
            self.assertFalse(
                game.score(item, {"content": book, "reasoning": probes[lang]})[
                    "correct"
                ]
            )

    def test_incomplete_evasive_or_wrong_language(self):
        for text in (
            "poppy.",
            "Es ist ein dunkler Tag und das Wort poppy ist auf der Karte.",
            "poppy poppy poppy poppy poppy poppy.",
        ):
            a = witness("en")
            a["cards"]["S"]["text"] = text
            self.assertFalse(self.score(a)["correct"])
        a = witness("en")
        del a["cards"]["F"]
        self.assertFalse(self.score(a)["correct"])
        s = fixture("en")
        s["routes"] += s["routes"]
        self.assertFalse(game.validate_expected(s))


if __name__ == "__main__":
    unittest.main()
