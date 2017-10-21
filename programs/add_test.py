from instruction import *

# 4 noops just to be safe
program = [
    LOAD("R1", "R0", 1),
    NOOP(),
    NOOP(),
    NOOP(),
    NOOP(),
    LOAD("R2", "R0", 2),
    NOOP(),
    NOOP(),
    NOOP(),
    NOOP(),
    ADD("R3","R1","R2"),
    NOOP(),
    NOOP(),
    NOOP(),
    NOOP(),
    NOOP()
]

memory = [
    1, 3, 5
]
