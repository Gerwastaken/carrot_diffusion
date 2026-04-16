import atexit
import inspect
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from colorama import init as colorama_init
from termcolor import cprint

colorama_init()


class Timer:
    def __init__(self):
        self.curr_state = 'init'
        self.curr_time = time.time()
        self.record = {}
        atexit.register(self.log)
        # colorama_init()

    def tick(self, next_state=None):
        if next_state is None:
            # 获取调用 tick 的上一帧
            frame = inspect.stack()[1]
            filepath = os.path.relpath(frame.filename)
            lineno = frame.lineno
            next_state = f"{filepath}:{lineno}"
        next_time = time.time()
        self._add_record(self.curr_state, next_state,
                         next_time - self.curr_time)
        self.curr_state = next_state
        self.curr_time = next_time

    def _add_record(self, s0, s1, delta):
        if (s0, s1) not in self.record:
            self.record[(s0, s1)] = []
        self.record[(s0, s1)].append(delta)

    def log(self):
        if self.curr_state == 'init':
            return
        # for key in self.record.keys():
        #     mean_delta = sum(self.record[key]) / len(self.record[key])
        #     print(f'{key}: {mean_delta}s, sum: {len(self.record[key])}')
        for (s0, s1), times in self.record.items():
            count = len(times)
            total = sum(times)
            avg = total / count
            mid = sorted(times)[count // 2]
            print(
                f"{Fore.YELLOW}{s0} {Fore.GREEN}→ {Fore.YELLOW}{s1}: "
                f"{Fore.MAGENTA}avg {avg:.6f}s, "
                f"{Fore.MAGENTA}mid {mid:.6f}s, "
                f"{Fore.BLUE}total {total:.6f}s, "
                f"{Fore.CYAN}count {count}"
            )


timer = Timer()


def _show(a, depth=0):
    space0 = '  ' * depth
    space = '  ' * (depth+1)
    if isinstance(a, dict):
        cprint('{')
        for k in a.keys():
            cprint(space + f'{k}: ', end="")
            _show(a[k], depth=depth+1)
        cprint(space0 + '}')
    elif isinstance(a, list):
        if len(a):
            cprint(f'list[{len(a)}]:')
            cprint(space, end="")
            _show(a[0], depth=depth+1)
        else:
            cprint('list[]')
    elif isinstance(a, tuple):
        cprint('(')
        for x in a:
            cprint(space, end="")
            _show(x, depth=depth+1)
        cprint(space0 + ')')
    elif isinstance(a, np.ndarray):
        cprint(f'np.ndarray: {a.dtype}, {a.shape}', 'yellow')
    elif isinstance(a, torch.Tensor):
        cprint(f'torch.Tensor: {a.dtype}, {a.shape}, {a.device}', 'blue')
    else:
        cprint(f'{type(a)}')


def show(a):
    _show(a)


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        return True

    def __getattr__(self, name):
        return getattr(self.files[0], name)


def log_to(log_path):
    f = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)

class SafeCV2:
    def __init__(self):
        self.frames = {}       # 窗口名 -> 最新帧
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        atexit.register(self._cleanup)  # 程序退出自动清理

    def imshow(self, window_name, frame):
        """线程安全的imshow"""
        with self.lock:
            self.frames[window_name] = frame.copy()
        self.event.set()

    def _resize_if_needed(self, frame):
        MAX_SIZE = 800
        h, w = frame.shape[:2]
        if max(h, w) <= MAX_SIZE:
            return frame
        scale = MAX_SIZE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))

    def _loop(self):
        while self.running:
            self.event.wait(timeout=0.05)
            self.event.clear()
            with self.lock:
                for name, frame in self.frames.items():
                    cv2.imshow(name, self._resize_if_needed(frame[..., ::-1]))
            cv2.waitKey(1)
        cv2.destroyAllWindows()

    def _cleanup(self):
        self.running = False
        self.event.set()
        if self.thread.is_alive():
            self.thread.join()


_safe_cv2 = SafeCV2()
imshow = _safe_cv2.imshow
