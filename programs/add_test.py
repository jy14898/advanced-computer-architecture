from assembly import *

# 4 noops just to be safe
instructions = assemble([
    # LOAD("R1", "R0", 1),
    # NOOP(),
    # NOOP(),
    # NOOP(),
    # NOOP(),
    # LOAD("R2", "R0", 2),
    # NOOP(),
    # NOOP(),
    # NOOP(),
    # NOOP(),
    ADD("R3","R1","R2"),
    LOAD("R2", "R0", 2),
    SUB("R3","R2","R3"),
    DIV("R3","R3","R4"),
    BEQ(123,"R1","R0"),
    J(0),
    XOR("R3","R1","R2"), # branch delay
    XOR("R4","R1","R3"), # branch delay
    NOOP(),
    NOOP(),
    NOOP(),
    NOOP(),
    NOOP(),
    NOOP(),
    NOOP(),
    NOOP(),
    NOOP()
])

memory = [
    1, 3, 5
]
