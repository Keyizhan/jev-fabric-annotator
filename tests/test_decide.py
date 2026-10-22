from jevfabric.decide import decide_shape


def v(*letters): return [(x, 0.8) for x in letters]


def test_plain_agreement():
    assert decide_shape(v("C", "C"), []) == ("C", "jev")


def test_one_vote_is_not_enough():
    assert decide_shape(v("F"), []) is None


def test_geometry_settles_b_versus_e():
    assert decide_shape(v("E", "B"), ["B", "B", "B"]) == ("B", "geo")


def test_geometry_needs_three_agreeing_votes():
    assert decide_shape(v("E", "E"), ["B", "B"]) is None


def test_falls_back_to_jev_when_geometry_is_silent():
    votes = v("E", "E", "E", "E", "E", "E")
    assert decide_shape(votes[:5], []) is None
    assert decide_shape(votes, []) == ("E", "jev_uncertain")


def test_geometry_does_not_override_c_d_f():
    assert decide_shape(v("C", "C", "E"), ["B", "B", "B"]) == ("C", "jev")
