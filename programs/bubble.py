from assembly import *
import random

random.seed(0)
n = 20
data = list(reversed(xrange(n)))

#random.shuffle(data)
program = {
    "instructions": assemble([
        ADDI("R1","R0",n - 2), # j = n - 2
        # loop 1 body
        SUBI("R2","R0",1), # i = -1
        ADDI("R2","R2",1), # i = i + 1
        # loop 2 body
        #STOR("R1", "R2", 0), # MEM[i + 0] = j
        LOAD("R3", "R2", 0),  # a = MEM[i + 0]
        LOAD("R4", "R2", 1),  # b = MEM[i + 1]
        SUB("R5", "R3","R4"), # c = a - b      ; if c > 0 then a > b
        # jump over block if they're sorted
        BLEZ(9, "R5"),
        # swap start
        STOR("R3", "R2", 1), # MEM[i + 1] = a 
        STOR("R4", "R2", 0), # MEM[i + 0] = b 
        # swap end
        # loop 2 body end
        BNE(2, "R2","R1"),  # pc = body2 if R2 != R1 else pc + 1
        # loop 1 body end
        SUBI("R1","R1",1), # j = j - 1 
        BGEZ(1, "R1"),     # pc = body1 if R1 >= 0 else pc + 1
        HALT(),

        


       # # LOOP START
       # ADDI("R2","R0",0), # i = 0
       # # LOOP START
       # LOAD("R3", "R2", 0),  # a = MEM[i + 0]
       # LOAD("R4", "R2", 1),  # b = MEM[i + 1]
       # SUB("R5", "R3","R4"), # c = a - b      ; if c > 0 then a > b

       # # skip over if if we don't swap
       # BLEZ(some_number, "R5"),
       # # IF START
       # STOR("R3", "R2", 1), # MEM[i + 1] = a 
       # STOR("R4", "R2", 0), # MEM[i + 0] = b 
       # # IF END
       # ADDI("R2","R2", 1),
       # BEQ(1, "R1", "R2"), # goto 1 if j == i
       # J(3)
       # # LOOP END
    
        HALT(),
        NOOP(),
        NOOP(),
        NOOP()
    ]),
    "data": data, 
}
