from assembly import *

program = {
    "instructions": assemble([
        ADDI("R1","R0",0), # int a = 0;
        ADDI("R2","R0",1), # int b = 1;
        
        # while (1) {
        ADDI("R3", "R1", 0),    # int c = a;
        ADDI("R1", "R2", 0),    # a = b;
        ADD ("R2", "R2", "R3"), # b = b + c;
        SUBI("R3", "R2", 300000000000000000000000000000000),  # c = b - 100;
        BGTZ(8, "R3"),          # if (c > 0) break;
        # }
        J(2),
        HALT(),
        NOOP(),
        NOOP(),
        NOOP()
    ]),
    "data": [
        123, 3, 5
    ]
}
