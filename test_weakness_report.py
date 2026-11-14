import weakness_report as wr


def test_report():
    rows = [{"task": "lookalike", "lang": "en", "extra": {"ok_control": 0.9, "ok_lookalike": 0.4, "took_twin": 0.5}},
            {"task": "arith_length", "lang": "de", "extra": {"ok_d2": 1.0, "ok_d4": 0.9, "ok_d6": 0.3, "ok_d8": 0.0}},
            {"task": "option_order", "lang": "en", "extra": {"group_agree": 0.6}}]
    gaps, baits, cliffs, cons = wr.report(rows)
    assert gaps[0][1] == "lookalike/en" and abs(gaps[0][0] - 0.5) < 1e-9
    assert baits[0][2] == "took_twin" and cliffs == [("arith_length/de", "d", 6, "2:1.00 4:0.90 6:0.30 8:0.00")]
    assert cons[0][2] == "group_agree"
    _, _, cl, _ = wr.report([{"task": "nesting_depth", "lang": "en", "extra": {
        "ok_eval_d1": 1.0, "ok_eval_d2": 0.2, "ok_balance_d1": 0.9, "ok_balance_d2": 0.9}},
        {"task": "x", "lang": "en", "extra": {"ok_d1": 0.2, "ok_d2": 0.9}},
        {"task": "de_compound_gender", "lang": "de", "extra": {"ok_2parts": 0.9, "ok_3parts": 0.3}}])
    assert ("nesting_depth/en", "eval_d", 2, "1:1.00 2:0.20") in cl and not any(c[0] == "x/en" for c in cl)
    assert any(c[0] == "de_compound_gender/de" and c[1] == "parts" for c in cl)


if __name__ == "__main__":
    test_report()
    print("ok")
