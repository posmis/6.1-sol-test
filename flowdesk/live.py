"""Bounded, nonblocking live previews. These are never saved as solved results."""
from dataclasses import dataclass, replace
import pickle
import time
import numpy as np


@dataclass(frozen=True)
class PreviewFrame:
    config: object
    iteration: int
    physical_time: float
    phase: str
    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray
    fluid: np.ndarray
    temperature: np.ndarray | None = None
    boundaries: dict | None = None
    mach: np.ndarray | None = None
    density: np.ndarray | None = None


class PreviewMailbox:
    """One latest snapshot in shared memory, with no multiprocessing feeder thread.

    A slow renderer skips snapshots. Neither process ever waits for the other;
    a killed worker cannot leave the GUI waiting on a pipe or a held lock.
    Pickle is internal IPC between our processes, never a user file format.
    """
    def __init__(self, context, capacity=2*1024**2):
        self.capacity = capacity
        self.buffer = context.RawArray('B', capacity)
        self.length = context.RawValue('i', 0)
        self.sequence = context.RawValue('Q', 0)
        self.lock = context.Lock()
        self.seen = 0

    def publish(self, frame):
        data = pickle.dumps(frame, protocol=5)
        if len(data) > self.capacity and frame.boundaries is not None:
            data = pickle.dumps(replace(frame, boundaries=None), protocol=5)
        if len(data) > self.capacity or not self.lock.acquire(False):
            return
        try:
            np.frombuffer(self.buffer, np.uint8)[:len(data)] = np.frombuffer(data, np.uint8)
            self.length.value = len(data)
            self.sequence.value += 1
        finally:
            self.lock.release()

    def take_latest(self):
        if self.sequence.value == self.seen or not self.lock.acquire(False):
            return None
        try:
            if self.sequence.value == self.seen:
                return None
            data = np.frombuffer(self.buffer, np.uint8)[:self.length.value].tobytes()
            self.seen = self.sequence.value
        finally:
            self.lock.release()
        return pickle.loads(data)


class PreviewEmitter:
    """Sample only when due: at most 320x192 float32 pixels, up to 5 Hz."""
    def __init__(self, callback, config, fluid, cell_ids=None, boundaries=None,
                 interval=.2, clock=time.perf_counter):
        self.callback, self.interval, self.clock = callback, interval, clock
        self.last = float('-inf')
        nx, ny = min(config.nx, 320), min(config.ny, 192)
        rows = ((np.arange(ny)+.5)*config.ny/ny).astype(int)
        columns = ((np.arange(nx)+.5)*config.nx/nx).astype(int)
        self.indices = np.ix_(rows, columns) if cell_ids is None else cell_ids[np.ix_(rows, columns)]
        self.fluid = fluid[np.ix_(rows, columns)].copy()
        self.config = replace(config, nx=nx, ny=ny)
        self.boundaries = boundaries

    def emit(self, iteration, physical_time, values, phase='flow', force=False):
        if self.callback is None:
            return
        now = self.clock()
        if not force and now-self.last < self.interval:
            return
        self.last = now
        u, v, pressure, temperature, *gas = values()
        def sample(q):
            return np.where(self.fluid, np.asarray(q)[self.indices], 0.).astype(np.float32)
        self.callback(PreviewFrame(self.config, int(iteration), float(physical_time), phase,
            sample(u), sample(v), sample(pressure), self.fluid,
            None if temperature is None else sample(temperature), self.boundaries,
            mach=sample(gas[0]) if gas else None, density=sample(gas[1]) if gas else None))


def structured_values(solver):
    c, f = solver.c, solver.fluid
    ref = np.mean(solver.k[:, -1][f[:, -1]])
    pressure = (solver.p-2*solver.k/3+2*ref/3)*c.rho
    return *solver.centres(), pressure, solver.temperature
