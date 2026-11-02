"""Preview throttling, snapshot isolation, native cells and spawned shared IPC."""
import multiprocessing as mp
import time
import numpy as np
import pytest
from flowdesk.solver import Config, Solver
from flowdesk.live import PreviewEmitter, PreviewMailbox


def test_preview_is_bounded_throttled_and_detached_from_solver():
    c = Config(nx=1024, ny=512, energy=True)
    f = np.ones((512, 1024), bool)
    f[100:200, 100:200] = False
    q = np.ones(f.shape)
    now = [0.]
    frames, calls = [], []
    emitter = PreviewEmitter(frames.append, c, f, clock=lambda: now[0])
    def values():
        calls.append(1)
        return q, q, q, q
    for i, timestamp in enumerate((0., .01, .19, .2, .21, .4)):
        now[0] = timestamp
        emitter.emit(i, timestamp, values)
    assert len(frames) == len(calls) == 3
    assert frames[-1].u.shape == (192, 320)
    assert frames[-1].u.dtype == np.float32
    assert np.all(frames[-1].u[~frames[-1].fluid] == 0.)
    frames[0].u[:] = 100.
    assert np.all(q == 1.) and np.max(frames[1].u) == 1.


def frame(iteration, nx=16, ny=8):
    c = Config(nx=nx, ny=ny)
    f = np.ones((ny, nx), bool)
    q = np.ones(f.shape)
    frames = []
    PreviewEmitter(frames.append, c, f).emit(iteration, 0., lambda: (q, q, q, None))
    return frames[0]


def publish_and_wait(mailbox, ready, release):
    mailbox.publish(frame(7, nx=512, ny=256))
    ready.set()
    release.wait(10)


def test_latest_frame_and_locked_reader_never_build_a_backlog():
    mailbox = PreviewMailbox(mp.get_context('spawn'))
    for i in range(12):
        mailbox.publish(frame(i))
    assert mailbox.take_latest().iteration == 11
    assert mailbox.take_latest() is None
    assert len(mailbox.buffer) == 2*1024**2
    mailbox.lock.acquire()
    try:
        started = time.perf_counter()
        mailbox.publish(frame(12))
        assert mailbox.take_latest() is None
        assert time.perf_counter()-started < 1.
    finally:
        mailbox.lock.release()
    mailbox.publish(frame(13))
    assert mailbox.take_latest().iteration == 13


def test_shared_preview_arrives_before_spawned_producer_finishes():
    context = mp.get_context('spawn')
    mailbox = PreviewMailbox(context)
    ready, release = context.Event(), context.Event()
    process = context.Process(target=publish_and_wait, args=(mailbox, ready, release))
    process.start()
    try:
        assert ready.wait(20)
        assert process.is_alive()
        value = mailbox.take_latest()
        assert value.iteration == 7 and value.u.shape == (192, 320)
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0


@pytest.mark.parametrize('mode', ['steady', 'transient'])
def test_preview_does_not_change_computed_flow_or_temperature(mode):
    c = Config(mode=mode, nx=32, ny=12, frames=3, duration=.01,
               energy=True, thermal_wall='isothermal', max_iterations=10)
    frames = []
    r = Solver(c).run(preview=frames.append)
    reference = Solver(c).run()
    for key in ('u', 'v', 'pressure', 'temperature'):
        np.testing.assert_array_equal(getattr(r, key), getattr(reference, key))
    assert frames[0].iteration == 0
    assert frames[-1].iteration > 0
    assert len(frames) >= 2
    np.testing.assert_array_equal(frames[-1].u, r.u[-1])
    np.testing.assert_array_equal(frames[-1].temperature, r.temperature[-1])


def test_native_preview_reports_original_cells_and_geometry(tmp_path):
    from flowdesk.cad import example_geometry
    from flowdesk.meshing import generate, MeshSettings
    g = example_geometry()
    g.loops[0][1][0] = g.loops[0][2][0] = .12
    mesh = generate(g, MeshSettings(target_cells=100))
    path = tmp_path/'mesh.flowmesh'
    mesh.save(path)
    c = Config(scenario='custom', mesh_path=str(path), mode='steady',
        nx=64, ny=24, energy=True, thermal_wall='isothermal', max_iterations=20)
    frames = []
    r = Solver(c).run(preview=frames.append)
    assert set(frames[-1].boundaries) == {'wall', 'inlet', 'outlet', 'open'}
    assert frames[-1].config.length == pytest.approx(.12)
    np.testing.assert_array_equal(frames[-1].u, r.u[-1])
    np.testing.assert_array_equal(frames[-1].temperature, r.temperature[-1])


def test_coarse_grid_and_polar_angles_are_identified():
    frames = []
    Solver(Config(mode='steady', nx=160, ny=48, max_iterations=10)).run(preview=frames.append)
    assert any(f.phase == 'coarse' for f in frames)
    assert frames[-1].phase == 'flow' and frames[-1].config.nx == 160
    # Profile callback must preserve each angle as the temporary mesh changes.
    frames = []
    Solver(Config(scenario='elliptic_profile', mode='steady', profile_cells=400,
        speed=20., rho=1.225, nu=1.5e-5, max_iterations=10, profile_sweep=(-4., 4.))).run(preview=frames.append)
    assert {f.config.profile_angle for f in frames} == {-4., 4.}
    assert all(f.boundaries is not None for f in frames)


def publish_and_exit(mailbox):
    for i in range(5):
        mailbox.publish(frame(i, nx=512, ny=256))


def test_producer_exits_even_when_gui_never_consumes_large_previews():
    context = mp.get_context('spawn')
    mailbox = PreviewMailbox(context)
    process = context.Process(target=publish_and_exit, args=(mailbox,))
    process.start()
    try:
        process.join(20)
        assert not process.is_alive(), 'Preview producer is blocked by an unread frame'
        assert process.exitcode == 0
        assert mailbox.take_latest().iteration == 4
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
