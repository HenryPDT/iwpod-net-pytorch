"""MeterBuffer unit tests (CPU-only, no data)."""
from iwpod.meters import AverageMeter, MeterBuffer, gpu_mem_usage, host_mem_usage


def test_average_meter_window_and_global():
    m = AverageMeter(window_size=3)
    for v in (1.0, 2.0, 3.0, 4.0):
        m.update(v)
    # window holds last 3 -> avg 3.0; global covers all 4 -> 2.5
    assert m.avg == 3.0
    assert m.global_avg == 2.5
    assert m.latest == 4.0


def test_meter_buffer_filter_and_clear():
    buf = MeterBuffer(window_size=10)
    buf.update(iter_time=0.1, data_time=0.02, total_loss=2.5, lr=1e-3)
    buf.update(iter_time=0.3, data_time=0.04, total_loss=2.0, lr=1e-3)
    times = buf.get_filtered_meter("time")
    assert set(times) == {"iter_time", "data_time"}
    assert times["iter_time"].avg == 0.2
    buf.clear_meters()
    assert buf["iter_time"].avg == 0.0


def test_mem_helpers_cpu_safe():
    from iwpod.meters import gpu_mem_peak
    assert gpu_mem_usage() >= 0.0
    assert gpu_mem_peak() >= 0.0
    assert host_mem_usage() >= 0.0
