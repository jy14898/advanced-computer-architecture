#!/usr/bin/env python2
from collections import namedtuple

# example programs require recursion, do i need link operations?

# Assumes all instructions are same fixed length, so we can just pull a whole instruction at a time without needing to decode num of arguments

# should i have $R0 = 0 too?

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
