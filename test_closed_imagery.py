"""Small grammatical word banks unrelated to the private tasks."""
import copy
import unittest

import closed_imagery as poem


def fixture(lang):
    if lang=="de":
        images=[["der Ball","dem Ball","sg","Spiel",[0,2]],
                ["die Räder","den Rädern","pl","Werkzeug",[0,1]],
                ["der Ast","dem Ast","sg","Baum",[1,2]],
                ["die Ringe","den Ringen","pl","Schmuck",[2]]]
        traits=[["rollt","rollen","im Rollen"],["knarrt","knarren","im Knarren"],["glänzt","glänzen","im Glanz"]]
    else:
        images=[["the ball","the ball","sg","game",[0,2]],
                ["the wheels","the wheels","pl","tools",[0,1]],
                ["the branch","the branch","sg","tree",[1,2]],
                ["the rings","the rings","pl","jewels",[2]]]
        traits=[["rolls","roll","a rolling motion"],["creaks","creak","a creak"],["gleams","gleam","a gleam"]]
    return {"language":lang,"images":images,"traits":traits,"anchors":[0,1,3],"length":3,"turn":1,"drafts":1}


def witness(lang):
    if lang=="de":
        return "Der Ball rollt wie die Räder.\nIm Knarren gleichen die Räder dem Ast.\nDer Ast glänzt wie die Ringe."
    return "The ball rolls like the wheels.\nThe wheels share a creak with the branch.\nThe branch gleams like the rings."


class ImageryTests(unittest.TestCase):
    def evaluate(self,lang,text,spec=None):
        return poem.score({"lang":lang,"expected":spec or fixture(lang)},{"content":text})

    def test_valid_language_and_localized_form(self):
        for lang in ("de","en"):
            self.assertTrue(poem.validate_expected(fixture(lang)))
            self.assertTrue(self.evaluate(lang,witness(lang))["correct"])
            self.assertFalse(self.evaluate(lang,witness("en" if lang=="de" else "de"))["correct"])
        wrong_case=witness("de").replace("dem Ast","der Ast")
        self.assertEqual(self.evaluate("de",wrong_case)["detail"]["verse_grammar"],2/3)
        wrong_number=witness("en").replace("wheels share","wheels shares")
        self.assertFalse(self.evaluate("en",wrong_number)["correct"])

    def test_semantic_and_style_failures(self):
        original=witness("en")
        for key,modified in [
            ("verse_shared_traits",original.replace("ball rolls","ball creaks")),
            ("verse_linked",original.replace("wheels share","rings share")),
            ("verse_varied_forms",original.replace("The wheels share a creak with the branch.","The wheels creak like the branch.")),
            ("verse_length",original+"\nAgain."),
            ("verse_grammar",original.replace("rolls like","really rolls like")),
        ]:
            with self.subTest(key=key):
                value=self.evaluate("en",modified)
                self.assertFalse(value["correct"])
                self.assertLess(value["detail"][key],1)
        for key,field,mod in [
            ("verse_cross_domain","images",["game"]),
            ("verse_anchors","anchors",[0,2,3]),
        ]:
            spec=fixture("en")
            if field=="images":
                spec[field][1][3]=mod[0]
            else:
                spec[field]=mod
            self.assertEqual(self.evaluate("en",original,spec)["correct"],False)

    def test_two_versions_must_change_an_image(self):
        spec=fixture("en")
        spec["drafts"]=2
        spec["images"].append(["the leaf","the leaf","sg","forest",[1,2]])
        first=witness("en")
        second=first.replace("branch","leaf")
        for text,good in ((first,False),(first+"\n\n"+first,False),(first+"\n\n"+second,True)):
            self.assertEqual(self.evaluate("en",text,spec)["correct"],good)
        same_images=first.replace("The ball rolls like the wheels.","The ball shares a rolling motion with the wheels.")
        result=self.evaluate("en",first+"\n\n"+same_images,spec)
        self.assertEqual(result["detail"]["verse_grammar"],1)
        self.assertEqual(result["detail"]["verse_distinct_drafts"],0)
        self.assertFalse(result["correct"])

    def test_verifier_contract(self):
        spec=fixture("de")
        for field,wrong in [("turn",True),("anchors",[0,0,3]),("length",0),("traits",[None]),("images",{}),("language","fr")]:
            self.assertFalse(poem.validate_expected(dict(spec,**{field:wrong})))
        invalid=copy.deepcopy(spec)
        invalid["images"][0][4]=[True]
        self.assertFalse(poem.validate_expected(invalid))
        for bad in (None,{},["xx"],"x"*2500, "", witness("de").lower()):
            self.assertFalse(self.evaluate("de",bad)["correct"])


if __name__=="__main__":
    unittest.main()
