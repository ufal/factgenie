def test_demo_ok():
    assert 1 + 1 == 2


def test_demo_ko():
    assert 1 + 1 == 3


def test_demo_exception():
    return 1 / 0
