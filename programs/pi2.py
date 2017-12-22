from assembly import *

program = {
    "instructions": assemble([
        ADDI("R1","R0",40000000000),
        ADDI("R2","R0",30000000000),
        ADDI("R6","R0",2),
        ADDI("R7","R0",3),
        ADDI("R8","R0",4),
        ADDI("R9","R0",10),
        # START
        MUL("R10","R6","R7"),
        MUL("R10","R10","R8"),
        DIV("R3","R1","R10"), 
        ADD("R2","R2","R3"), 

        ADDI("R6","R6",2),   
        ADDI("R7","R7",2),   
        ADDI("R8","R8",2),   

        MUL("R10","R6","R7"),
        MUL("R10","R10","R8"),
        DIV("R3","R1","R10"), 
        SUB("R2","R2","R3"), 

        SUBI("R9","R9", 1), 

        BGTZ(6, "R9"),
        HALT(),
    ]),
    "data": [
        123, 3, 5
    ]
}
