import math
import threading
import time
from types import SimpleNamespace

from gazeread.app.server import wait_until_logged


def test_wait_until_logged_waits_for_in_flight_frame():
    p = SimpleNamespace(logged_t=-math.inf)
    t_end = time.monotonic()

    def capture_thread():
        time.sleep(0.05)
        p.logged_t = t_end - 0.01        # a frame captured before the end is still being written
        time.sleep(0.05)
        p.logged_t = t_end + 0.02        # first frame captured after the end

    threading.Thread(target=capture_thread).start()
    t0 = time.monotonic()
    assert wait_until_logged(p, t_end, timeout_s=1.0)
    assert time.monotonic() - t0 >= 0.09


def test_wait_until_logged_times_out():
    p = SimpleNamespace(logged_t=0.0)
    assert not wait_until_logged(p, 10.0, timeout_s=0.05)
