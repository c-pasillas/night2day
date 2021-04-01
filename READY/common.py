import pathlib
import argparse
import os
import sys
import time
import sqlite3
import subprocess
import logging

reset, bold, underline, invert = '\u001b[0m', '\u001b[1m', '\u001b[4m', '\u001b[7m'
def is_byte(i):
    return 0 <= i <= 256
def color(byte):
    assert(is_byte(byte))
    # f'\u001b[48;5;{byte}m' background
    return f'\u001b[38;5;{byte}m'
def rgb(r, g, b):
    assert(is_byte(r) and is_byte(g) and is_byte(b))
    return f'\u001b[38;2;{r};{g};{b}m'
def gray(byte):
    return rgb(byte, byte, byte)

blue = rgb(3, 156, 199)
yellow = rgb(255, 255, 0)
yellow2 = rgb(110, 110, 80)
orange = rgb(255, 100, 0)

if sys.stderr.isatty():
    date_fmt = f'{gray(130)}%H:%M{gray(100)}:{reset}%S'
    base = (f'%(asctime)s{gray(170)}.%(msecs).3d ' +
            f'{blue}%(filename)s %(lineno)d {yellow}→ ' +
            f'{gray(150)}def {orange}%(funcName)s{gray(150)}() ')
    info_fmt  = base + f'{yellow}⬛{reset} %(message)s'
    debug_fmt = base + f' {yellow2}⬛{reset} %(message)s'
else:
    date_fmt = '%H:%M:%S'
    base = ('%(asctime)s.%(msecs).3d ' +
            '%(filename)s %(lineno)d → ' +
            'def %(funcName)s() ')
    info_fmt  = base + '⬛ %(message)s'
    debug_fmt = base + ' ⬛ %(message)s'

log = logging.getLogger('night2day')
log.setLevel(logging.DEBUG)

for level, fmt in [(logging.INFO, info_fmt), (logging.DEBUG, debug_fmt)]:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))
    if level == logging.DEBUG:
        handler.addFilter(lambda record: record.levelno == logging.DEBUG)
    log.addHandler(handler)

recent_dbs = pathlib.Path.home() / '.night2day'
def get_recents():
    if recent_dbs.exists():
        with open(recent_dbs) as f:
            return [line.strip() for line in f.readlines()]
    else:
        return []
def add_to_recents(path):
    current = get_recents()
    current.remove(str(path))
    with open(recent_dbs, 'w') as f:
        f.write(str(path) + '\n')
        for line in current:
            f.write(line + '\n')

def name_raw_h5_dir(path):
    p = pathlib.Path(path).resolve()
    if not p.is_dir():
        return
    for c in p.iterdir():
        if not c.is_dir() or c.name[:3].lower() != 'raw':
            continue
        if any(f.suffix == '.h5' for f in c.iterdir()):
            return c

db_name = 'night2day.db'
def find_db_from_cwd():
    c = pathlib.Path.cwd()
    d = c / db_name
    if d.exists():
        return d
    d = c.parent / db_name
    if d.exists():
        return d

def locate_db(path):
    if not path:
        return find_db_from_cwd()
    p = pathlib.Path(path).resolve()
    if p.is_file() and p.name == db_name:
        return p
    d = p / db_name
    if d.exists():
        return d

setup_case_db = """
    create table if not exists Settings (name, value);
    create table if not exists HDF (filename, channels, satellite, date, start_time, end_time);
    """
def create_db(root_dir):
    h5_dir = name_raw_h5_dir(root_dir)
    if not h5_dir:
        h5_dir = name_raw_h5_dir(pathlib.Path(root_dir).resolve().parent)
    if not h5_dir:
        raise Exception(f'Expected {root_dir} to contain a subfolder named "raw" with .h5 files in it.')
    db = h5_dir.parent / db_name
    if db.exists():
        raise Exception(f'During creation, found existing db {db}')
    conn = sqlite3.connect(db)
    conn.executescript(setup_case_db)
    conn.execute('insert into Settings values (?, ?)', ('h5_dir_name', h5_dir.name))
    conn.commit()
    conn.close()
    return db
