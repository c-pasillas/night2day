#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import subprocess
import logging
import sqlite3
import time

import common
from common import log, bold, reset, color, rgb
import pack_case
import normalize
import learning_prep
import model_validaiton


def shell_setup():
    bin_dir = Path.home() / 'bin'
    cmd_symlink = bin_dir / 'night2day'
    main_file = Path(__file__).resolve()
    profile = Path.home() / '.profile'
    zshrc = Path.home() / '.zshrc'
    x = 'export PATH="$HOME/bin:$PATH"'
# TODO shell setup
#   create $HOME/bin folder if not already exists
#   symlink $HOME/bin/night2day -> __file__ (if not already there)
#   add to PATH (if not already there) by adding to .profile and .zshrc
#   export PATH="$HOME/bin:$PATH"

def recent(args):
    # create or read home folder db
    pass
def status(args):
    log.info("here")
def info(args):
    pass
def log_cmd(args):
    pass
def pack_case_cmd(args):
    db_path = common.locate_db(None)
    if not db_path:
        db_path = common.create_db(".")
    pack_case.pack_case(db_path)
def normalize_cmd(args):
    db_path = common.locate_db(None)
    normalize.normalize(db_path)
def learning_cmd(args):
    db_path = common.locate_db(None)
    learning_prep.learning_prep(db_path)
def model_val_cmd(args):
    db_path = common.locate_db(None)
    model_validation.model_val(db_path) 

# night2day status [case root dir | night2day.db file]
# night2day info file
# night2day log [root | db file]
# night2day pack-case [root | db | cwd]
# night2day normalize
# night2day prep-learning

desc = f'''Process satellite images for machine learning.'''
parser = argparse.ArgumentParser(description=desc)
parser.set_defaults(func=status)
subparsers = parser.add_subparsers()

msg = (f'Pack a case into a single array',
       '''Process and pack a case into a single array.
       Matches time-correlated images, regularizes dimensions across all images.''')
pack_case_p = subparsers.add_parser('pack-case', help=msg[0], description=msg[1])
pack_case_p.set_defaults(func=pack_case_cmd)
pack_case_p.add_argument('-q', '--quiet', action='count', default=0)

norm_p = subparsers.add_parser('normalize', help='Normalize and derive channels')
norm_p.set_defaults(func=normalize_cmd)
norm_p.add_argument('-q', '--quiet', action='count', default=0)

learn_p = subparsers.add_parser('learn-prep', help='Create file for input to learning')
learn_p.set_defaults(func=learning_cmd)
learn_p.add_argument('-q', '--quiet', action='count', default=0)


model_val_p = subparsers.add_parser('model-val', help='Create  validation file for model validation')
model_val_p.set_defaults(func=model_val_cmd)
model_val_p.add_argument('-q', '--quiet', action='count', default=0)

"""The default is to display both log.info() and log.debug() statements.
But if the user runs this program with the -q or --quiet flags, then only
display the log.info() statements."""
args = parser.parse_args()
if hasattr(args, 'quiet') and args.quiet > 0:
    log.setLevel(logging.INFO)
args.func(args)


# Full workflow steps
# night2day pack-case
# night2day normalize
# night2day learn-prep
#    edit learning_channels.txt
# night2day learn-prep


