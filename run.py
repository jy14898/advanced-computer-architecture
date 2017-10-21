#!/usr/bin/env python2

from processor import Processor

# set program to run
import programs.add_test

program = programs.add_test.get_program()

proc = Processor(program)

try:
    while True:
        proc.step()
except ValueError as e:
    proc.dump()
    raise
