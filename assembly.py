from collections import namedtuple

ADD  = namedtuple("ADD",  ["rd", "rv0", "rv1"])
SUB  = namedtuple("SUB",  ["rd", "rv0", "rv1"])
DIV  = namedtuple("DIV",  ["rd", "rv0", "rv1"]) # a = q*b + r | this gets q
MUL  = namedtuple("MUL",  ["rd", "rv0", "rv1"])
AND  = namedtuple("AND",  ["rd", "rv0", "rv1"])
OR   = namedtuple("OR",   ["rd", "rv0", "rv1"])
XOR  = namedtuple("XOR",  ["rd", "rv0", "rv1"])

ADDI = namedtuple("ADDI", ["rd", "rv0", "imm"])
SUBI = namedtuple("SUBI", ["rd", "rv0", "imm"])
DIVI = namedtuple("DIVI", ["rd", "rv0", "imm"])
MULI = namedtuple("MULI", ["rd", "rv0", "imm"])
ANDI = namedtuple("ANDI", ["rd", "rv0", "imm"])
ORI  = namedtuple("ORI",  ["rd", "rv0", "imm"])
XORI = namedtuple("XORI", ["rd", "rv0", "imm"])

BEQ  = namedtuple("BEQ",  ["ad", "rv0", "rv1"]) # PC = location if r0 == r1
BNE  = namedtuple("BNE",  ["ad", "rv0", "rv1"]) # r0 != r1

BGEZ = namedtuple("BGEZ", ["ad", "rv0"]) # >= 0
BGTZ = namedtuple("BGTZ", ["ad", "rv0"]) #  > 0
BLEZ = namedtuple("BLEZ", ["ad", "rv0"]) # <= 0
BLTZ = namedtuple("BLTZ", ["ad", "rv0"]) #  < 0

J  = namedtuple("J",  ["ad"])
JR = namedtuple("JR", ["rv"])

LOAD = namedtuple("LOAD", ["rd", "rv", "imm"]) # rd = MEM[rv + imm]
STOR = namedtuple("STOR", ["rd", "rv", "imm"]) # MEM[rv + imm] = rd

NOOP = namedtuple("NOOP", [])

# dont create directly
instruction = namedtuple("instruction", ["opcode", "reg_read_sel1", "reg_read_sel2", "reg_write_sel", "immediate"])

def r(reg):
    return int(reg[1:])

def to_instruction(line):
    if isinstance(line, ADD):
        return instruction("ADD", r(line[1]), r(line[2]), r(line[0]), 0)
    if isinstance(line, SUB):
        return instruction("SUB", r(line[1]), r(line[2]), r(line[0]), 0)
    if isinstance(line, DIV):
        return instruction("DIV", r(line[1]), r(line[2]), r(line[0]), 0)
    if isinstance(line, MUL):
        return instruction("MUL", r(line[1]), r(line[2]), r(line[0]), 0)
    if isinstance(line, AND):
        return instruction("AND", r(line[1]), r(line[2]), r(line[0]), 0)
    if isinstance(line, OR):
        return instruction("OR",  r(line[1]), r(line[2]), r(line[0]), 0)
    if isinstance(line, XOR):
        return instruction("XOR", r(line[1]), r(line[2]), r(line[0]), 0)

    if isinstance(line, ADDI):
        return instruction("ADDI", r(line[1]), 0, r(line[0]), line[2])
    if isinstance(line, SUBI):
        return instruction("SUBI", r(line[1]), 0, r(line[0]), line[2])
    if isinstance(line, DIVI):
        return instruction("DIVI", r(line[1]), 0, r(line[0]), line[2])
    if isinstance(line, MULI):
        return instruction("MULI", r(line[1]), 0, r(line[0]), line[2])
    if isinstance(line, ANDI):
        return instruction("ANDI", r(line[1]), 0, r(line[0]), line[2])
    if isinstance(line, ORI):
        return instruction("ORI",  r(line[1]), 0, r(line[0]), line[2])
    if isinstance(line, XORI):
        return instruction("XORI", r(line[1]), 0, r(line[0]), line[2])

    if isinstance(line, BEQ):
        return instruction("BEQ",  r(line[1]), r(line[2]), 0, line[0])
    if isinstance(line, BNE):
        return instruction("BNE",  r(line[1]), r(line[2]), 0, line[0])

    if isinstance(line, J):
        return instruction("J",  0, 0, 0, line[0])
    if isinstance(line, JR):
        return instruction("JR",  r(line[0]), 0, 0, 0)

    if isinstance(line, LOAD):
        return instruction("LOAD", r(line[1]), 0, r(line[0]), line[2])
    if isinstance(line, STOR):
        return instruction("STOR", r(line[1]), r(line[0]), 0, line[2])

    if isinstance(line, BGEZ):
        return instruction("BGEZ", r(line[1]), 0, 0, line[0])
    if isinstance(line, BGTZ):
        return instruction("BGTZ", r(line[1]), 0, 0, line[0])
    if isinstance(line, BLEZ):
        return instruction("BLEZ", r(line[1]), 0, 0, line[0])
    if isinstance(line, BLTZ):
        return instruction("BLTZ", r(line[1]), 0, 0, line[0])

    if isinstance(line, NOOP):
        return instruction("NOOP", 0, 0, 0, 0)

def assemble(program):
    return list(to_instruction(line) for line in program)
