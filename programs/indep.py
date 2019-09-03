from assembly import *

program = {
    "instructions": assemble([
        ADDI("R2","R0",1),
        ADDI("R2","R0",1),
        ADDI("R2","R0",1),
        J(0),
    ]),
    "data": [
        123, 3, 5
    ]
}
