import weakness_report as wr


def test_report():
    rows = [{"task": "lookalike", "lang": "en", "extra": {"ok_control": 0.9, "ok_lookalike": 0.4, "took_twin": 0.5}},
            {"task": "arith_length", "lang": "de", "extra": {"ok_d2": 1.0, "ok_d4": 0.9, "ok_d6": 0.3, "ok_d8": 0.0}},
            {"task": "option_order", "lang": "en", "extra": {"group_agree": 0.6}}]
    gaps, baits, cliffs, cons = wr.report(rows)
    assert gaps[0][1] == "lookalike/en" and abs(gaps[0][0] - 0.5) < 1e-9
    assert baits[0][2] == "took_twin" and cliffs == [("arith_length/de", "d", 6, "2:1.00 4:0.90 6:0.30 8:0.00")]
    assert cons[0][2] == "group_agree"


if __name__ == "__main__":
    test_report()
    print("ok")
