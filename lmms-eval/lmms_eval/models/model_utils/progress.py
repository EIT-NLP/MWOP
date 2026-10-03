import os
import sys
import time
from loguru import logger as eval_logger
from tqdm import tqdm

def _is_batch_mode() -> bool:
    if os.environ.get('SLURM_JOB_ID'):
        return True
    if not sys.stderr.isatty():
        return True
    return False

def _format_time(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return '?'
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f'{m}m{s:02d}s'
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f'{h}h{m:02d}m'

class SlurmProgress:

    def __init__(self, total: int, desc: str='Progress', disable: bool=False, log_interval: float=30.0):
        self.total = total
        self.desc = desc
        self.disable = disable
        self.log_interval = log_interval
        self.n = 0
        self._start_time = time.monotonic()
        self._last_log_time = 0.0
        self._rank = int(os.environ.get('RANK', os.environ.get('SLURM_ARRAY_TASK_ID', 0)))

    def update(self, n: int=1) -> None:
        self.n += n
        if self.disable:
            return
        now = time.monotonic()
        is_final = self.n >= self.total
        if not is_final and now - self._last_log_time < self.log_interval:
            return
        self._last_log_time = now
        elapsed = now - self._start_time
        speed = self.n / elapsed if elapsed > 0 else 0
        eta = (self.total - self.n) / speed if speed > 0 else 0
        pct = 100 * self.n / self.total if self.total > 0 else 0
        eval_logger.info(f'[rank {self._rank}] {self.desc}: {self.n}/{self.total} ({pct:.1f}%) | {speed:.2f} it/s | elapsed {_format_time(elapsed)} | ETA {_format_time(eta)}')

    def close(self) -> None:
        if self.disable:
            return
        elapsed = time.monotonic() - self._start_time
        speed = self.n / elapsed if elapsed > 0 else 0
        eval_logger.info(f'[rank {self._rank}] {self.desc}: done {self.n}/{self.total} in {_format_time(elapsed)} ({speed:.2f} it/s)')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def make_progress(total: int, desc: str='Progress', disable: bool=False, log_interval: float=30.0) -> 'tqdm | SlurmProgress':
    if _is_batch_mode():
        return SlurmProgress(total=total, desc=desc, disable=disable, log_interval=log_interval)
    return tqdm(total=total, desc=desc, disable=disable)
