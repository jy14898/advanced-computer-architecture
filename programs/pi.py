from assembly import *

program = {
    "instructions": assemble([
        ADDI("R1","R0",4000000),
        ADDI("R2","R0",1),
        # START
        DIV("R3","R1","R2"), 
        ADD("R4","R4","R3"), # + 4/1
        ADDI("R2","R2",2),   
        DIV("R3","R1","R2"), 
        SUB("R4","R4","R3"), # - 4/3
        ADDI("R2","R2",2),
        J(2),
        NOOP(),
        NOOP(),
        NOOP()
    ]),
    "data": [
        123, 3, 5
    ]
}
