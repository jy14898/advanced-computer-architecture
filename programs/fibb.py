from assembly import *

program = {
    "instructions": assemble([
        ADDI("R1","R1",1), # R1 = R1 xor R1 = 0
        ADDI("R2","R2",1), # R1 = R1 xor R1 = 0
        
        ADDI("R3","R2",0),
        ADD("R1","R1",4), # R1 = R1 xor R1 = 0
        J(0),                # PC = start
        NOOP(),
        NOOP(),
        NOOP()
    ]),
    "data": [
        123, 3, 5
    ]
}
