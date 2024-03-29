from assembly import *

program = {
    "instructions": assemble([
        ADDI("R1","R0",0), # int a = 0;
        ADDI("R2","R0",1), # int b = 1;
        
        # while (1) {
        ADDI("R3", "R1", 0),    # int c = a;
        ADDI("R1", "R2", 0),    # a = b;
        ADD ("R2", "R2", "R3"), # b = b + c;
        # }
        J(2),

        NOOP(),
        NOOP(),
        NOOP()
    ]),
    "data": [
        123, 3, 5
    ]
}
