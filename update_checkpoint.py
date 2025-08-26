#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from pedpred.saving_loading import TrainingState


def update_checkpoint(file):
	file = Path(file)
	file_stat = os.stat(file)
	state = TrainingState(file)
	state._save_file(file)
	os.utime(file, ns=(file_stat.st_atime_ns, file_stat.st_mtime_ns))

def main():
	for arg in sys.argv[1:]:
		try:
			update_checkpoint(arg)
		except Exception as e:
			print(e, file=sys.stderr)

if __name__ == '__main__': main()
