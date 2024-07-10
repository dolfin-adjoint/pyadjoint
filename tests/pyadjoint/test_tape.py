from pyadjoint import *  # noqa: F403


def test_reverse_over_forward_configuration():
    assert not reverse_over_forward_enabled()

    continue_reverse_over_forward()
    assert reverse_over_forward_enabled()
    pause_reverse_over_forward()
    assert not reverse_over_forward_enabled()

    continue_reverse_over_forward()
    assert reverse_over_forward_enabled()
    with stop_reverse_over_forward():
        assert not reverse_over_forward_enabled()
    assert reverse_over_forward_enabled()
    pause_reverse_over_forward()
    assert not reverse_over_forward_enabled()

    @no_reverse_over_forward
    def test():
        assert not reverse_over_forward_enabled()

    continue_reverse_over_forward()
    assert reverse_over_forward_enabled()
    test()
    assert reverse_over_forward_enabled()
    pause_reverse_over_forward()
    assert not reverse_over_forward_enabled()
