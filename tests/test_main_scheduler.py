from podcast_worker import main


def test_drain_respects_process_wide_capacity(monkeypatch):
    monkeypatch.setattr(main.cfg.settings, "max_concurrent_generations", 1)
    monkeypatch.setattr(main.state, "durable_work_count", 1)
    monkeypatch.setattr(main.state, "durable_work_stop", main.threading.Event())

    def unexpected_claim(*_args, **_kwargs):
        raise AssertionError("capacity-full scheduler must not claim more work")

    monkeypatch.setattr(main.persistence, "_claim_next_work", unexpected_claim)

    assert main._drain_durable_work("work_fresh") == 0


def test_reclaimer_retries_when_work_is_initially_unclaimable(monkeypatch):
    calls = []
    stop = main.threading.Event()

    def drain():
        calls.append(len(calls))
        if len(calls) == 2:
            stop.set()
        return 0

    class ImmediateWake:
        def wait(self, _interval):
            return False

        def clear(self):
            return None

    monkeypatch.setattr(main.state, "durable_work_stop", stop)
    monkeypatch.setattr(main.state, "durable_work_wake", ImmediateWake())
    monkeypatch.setattr(main, "_drain_durable_work", drain)
    monkeypatch.setattr(main.cfg.settings, "work_lease_seconds", 1)

    main._durable_work_reclaimer()

    assert calls == [0, 1]
