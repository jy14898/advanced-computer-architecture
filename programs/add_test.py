from assembly import *

# c = 0
#
# do:
#     a = c + 20
#     b = c + 10
#     c += 5
# until a / b == 1

program = {
    "instructions": assemble([
        XOR("R1","R1","R1"), # R1 = R1 xor R1 = 0
        BEQ(0,"R0","R0"),
        NOOP(),
        NOOP(),
        NOOP(),
    # start:
        ADDI("R2","R1",20),  # R2 = R1 + 20
        ADDI("R3","R1",2),   # R3 = R1 + 10
        NOOP(),
        NOOP(),
        NOOP(),
        DIV("R4","R2","R3"), # R4 = R2 / R3
        NOOP(),
        NOOP(),
        NOOP(),
        STOR("R4","R0",0),   # MEM[R0 + 0] = R4
        ADDI("R1", "R1", 5), # R1 = R1 + 5
        NOOP(),              # need NOPs here due to R1 being used in start
        NOOP(),
        J(4),                # PC = start
        J(100),
        NOOP(),
        NOOP(),
        NOOP()
        # LOAD("R1", "R0", 1),
        # # NOOP(),
        # # NOOP(),
        # # NOOP(),
        # # NOOP(),
        # # LOAD("R2", "R0", 2),
        # # NOOP(),
        # # NOOP(),
        # # NOOP(),
        # # NOOP(),
        # ADD("R3","R1","R2"),
        # LOAD("R2", "R1", 2),
        # SUB("R3","R2","R3"),
        # DIV("R3","R3","R4"),
        # BEQ(123,"R1","R0"),
        # J(0),
        # XOR("R3","R1","R2"), # branch delay
        # XOR("R4","R1","R3"), # branch delay
        # NOOP(),
        # NOOP(),
        # NOOP(),
        # NOOP(),
        # NOOP(),
        # NOOP(),
        # NOOP(),
        # NOOP(),
        # NOOP()
    ]),
    "data": [
        123, 3, 5
    ]
}
