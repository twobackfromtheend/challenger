import atexit
import json
import random
import subprocess
import threading
import time
from pathlib import Path

child_process = None

child_known_dead = False


def add_item(i: int, d: float, r: float, e: bool):
    data = {'i': i, 'd': d, 'r': r, 'e': e}
    line = json.dumps(data)
    line += '\n'
    message = line.encode('utf-8')
    try:
        child_process.stdin.write(message)
        child_process.stdin.flush()
    except Exception as e:
        global child_known_dead
        if child_known_dead:
            return
        child_known_dead = True
        raise Exception("===== GUI died =====")


def print_file(f):
    for line in f:
        print(line.decode('utf-8').rstrip())


def start_child_process():
    _dir: Path = Path(__file__).parent

    global child_process
    # noinspection PyTypeChecker
    child_process = subprocess.Popen(
        "python child.py",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=_dir,
    )
    atexit.register(lambda: child_process.kill())  # behave like a daemon
    read_out = threading.Thread(target=print_file, args=[child_process.stdout], daemon=True)
    read_out.start()
    read_err = threading.Thread(target=print_file, args=[child_process.stderr], daemon=True)
    read_err.start()


if __name__ == '__main__':
    start_child_process()
    i = 0
    while True:
        add_item(i, random.random(), random.random(), random.random() < 0.2)
        time.sleep(0.2)
        i += 1
