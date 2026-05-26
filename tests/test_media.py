from sam_vos.media import build_prompt, parse_box, parse_labels, parse_points


def test_parse_points_and_labels():
    assert parse_points("1,2;3,4") == [(1.0, 2.0), (3.0, 4.0)]
    assert parse_labels("1,0") == [1, 0]


def test_parse_box():
    assert parse_box("1,2,3,4") == (1.0, 2.0, 3.0, 4.0)


def test_build_prompt_defaults_labels_to_positive():
    prompt = build_prompt("10,20", None, None)
    assert prompt.labels == [1]
    assert prompt.points == [(10.0, 20.0)]
