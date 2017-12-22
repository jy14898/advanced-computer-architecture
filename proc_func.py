import functools
import json
import traceback
import itertools
import copy

from contextlib import contextmanager

from collections import namedtuple

class Processor(object):
    def __init__(self, instructions, data):
        self.instructions = instructions

        self.pc = 0
        self.registers = list(0 for _ in xrange(16))

    operations = {
        "ADD":  lambda ins: ins["a"] + ins["b"],
        "SUB":  lambda ins: ins["a"] - ins["b"],
        "DIV":  lambda ins: 0 if ins["b"] == 0 else ins["a"] // ins["b"],
        "MUL":  lambda ins: ins["a"] * ins["b"],
        "AND":  lambda ins: ins["a"] & ins["b"],
        "OR":   lambda ins: ins["a"] | ins["b"],
        "XOR":  lambda ins: ins["a"] ^ ins["b"],
    }

    def step(self):
        instruction = self.instructions[self.pc]

        opcode, reg_read_sel1, reg_read_sel2, reg_write_sel, immediate = instruction
        if opcode in ["ADD","SUB","DIV","MUL","AND","OR" ,"XOR", "ADDI","SUBI","DIVI","MULI","ANDI","ORI","XORI"]:
            is_immediate = opcode[-1:] == 'I'

            op = opcode[:-1] if is_immediate else opcode

            data = {
                "a": self.registers[reg_read_sel1],
                "b": self.registers[reg_read_sel2] if not is_immediate else immediate,
            }

            self.registers[reg_write_sel] = Processor.operations[op](data)
            self.pc += 1

        elif opcode in ["J"]:
            self.pc = immediate
