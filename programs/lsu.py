from assembly import *

program = {
    "instructions": assemble([
        LOAD("R1", "R0", 0),
        ADDI("R2", "R0", 1337),
        STOR("R2", "R0", 0),
        LOAD("R3", "R0", 0),
        HALT(),
    ]),
    "data": [
        123, 3, 5
    ]
}
