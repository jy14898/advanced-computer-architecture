from assembly import *

program = {
    "instructions": assemble([
        ADDI("R1","R1",1), # R1 = R1 xor R1 = 0
        ADD("R4","R4","R2"), # R1 = R1 xor R1 = 0
        ADDI("R2","R2",2), # R1 = R1 xor R1 = 0
        MULI("R1","R1",2), # R1 = R1 xor R1 = 0
        ADDI("R1","R1",3), # R1 = R1 xor R1 = 0
        ADDI("R3","R3",3), # R1 = R1 xor R1 = 0
        MULI("R4","R4",4), # R1 = R1 xor R1 = 0
        ADDI("R1","R1",4), # R1 = R1 xor R1 = 0
        J(0),                # PC = start
        NOOP(),
        NOOP(),
        NOOP()
    ]),
    "data": [
        123, 3, 5
    ]
}
