from temporal import DrowsinessMonitor


def run(events, fps=10):
    m, out = DrowsinessMonitor(), None
    for i, (c, y) in enumerate(events):
        out = m.update(i / fps, c, y)
    return out


def test_blink_is_ignored():
    ev = [(False, False)] * 50 + [(True, False)] * 3 + [(False, False)] * 50
    assert run(ev)["state"] == "OK"


def test_long_closure_alerts():
    ev = [(False, False)] * 20 + [(True, False)] * 20      # 2 s closed
    r = run(ev)
    assert r["state"] == "ALERT" and "prolonged eye closure" in r["reasons"]


def test_two_yawns_alert():
    yawn, rest = [(False, True)] * 15, [(False, False)] * 50
    assert run(yawn + rest + yawn + rest[:5])["state"] == "ALERT"


def test_one_yawn_is_caution():
    assert run([(False, True)] * 15 + [(False, False)] * 20)["state"] == "CAUTION"


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("PASS", k)
