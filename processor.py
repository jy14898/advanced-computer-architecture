import functools
import json


# -a note someone made here:http://slideplayer.com/slide/9255919/
# -https://www.cise.ufl.edu/~mssz/CompOrg/Figure5.6-PipelineControlLines.gif

class Component(object):
    def __init__(self):
        self.input_keys  = []
        self.output_keys = []

        self.output_values = {}

        self.clocked = False

    def name(self):
        return "component"

    '''
    dont actually use this yet
    '''
    def get_state(self):
        return {}

    def update_values(self, input_values):
        raise NotImplementedError('{}: Components must override update_values()!'.format(self.name()))

    def update_state(self, input_values):
        assert self.clocked == True
        raise NotImplementedError('{}: Components must override update_state()!'.format(self.name()))

    def add_input(self, key):
        self.input_keys.append(key)

    def add_output(self, key):
        self.output_keys.append(key)
        self.output_values[key] = None # before update

    def enable_clock(self):
        self.clocked = True

class ComponentConnectionOrchestrator(object):
    def generate_drawio(self):
        for ((out_component, out_key), (in_component, in_key)) in self.connections:
            print "#{" + "\"from\": \"{}\", \"to\": \"{}\", \"label\": \"{} -> {}\"".format(out_component.name(), in_component.name(), out_key, in_key) + "}"

        print "name"
        for component in self.components:
            print "{}".format(component.name())

        # connect:
    def generate_nomnoml(self):
        print "NOMNOML START"

        for component in self.components:
            print "[{}]".format(component.name())

        for ((out_component, out_key), (in_component, in_key)) in self.connections:
            print "[{}] {} -> {} [{}]".format(out_component.name(), out_key, in_key, in_component.name())

    def generate_json(self):
        ob = {}

        ob["components"] = {}

        for component in self.components:
            ob["components"][id(component)] = {
                "name": component.name(),
                "input_keys": component.input_keys,
                "output_keys": component.output_keys
            }

        ob["connections"] = []
        for ((out_component, out_key), (in_component, in_key)) in self.connections:
            ob["connections"].append([id(out_component), out_key, id(in_component), in_key])

        print json.dumps(ob)

    def __init__(self):
        self.components = []
        self.connections = [
            # ((out_component, out_key), (in_component, in_key))
        ]

        self.clock_phase = 0

    def add_component(self, component):
        assert component not in self.components

        self.components.append(component)

    def add_connection(self, (out_component, out_key), (in_component, in_key)):
        assert ((out_component, out_key), (in_component, in_key)) not in self.connections
        assert in_component in self.components
        assert out_component in self.components
        assert out_key in out_component.output_keys
        assert in_key in in_component.input_keys

        self.connections.append(((out_component, out_key), (in_component, in_key)))

    def clock(self):
        self.clock_phase = 1 - self.clock_phase

        for component in (c for c in self.components if c.clocked):
            incoming_connections = list(((a,b),(d)) for ((a,b),(c,d)) in self.connections if c == component)

            input_values = {}
            for ((out_component,out_key),in_key) in incoming_connections:
                input_values[in_key] = out_component.output_values[out_key]

            input_values["clock"] = self.clock_phase
            component.update_state(input_values)

    def propagate(self, setup=False):
        # crappy way of doing it but works
        changed = True
        while changed:
            changed = False
            for component in self.components:
                incoming_connections = list(((a,b),(d)) for ((a,b),(c,d)) in self.connections if c == component)

                input_values = {}
                for ((out_component,out_key),in_key) in incoming_connections:
                    input_values[in_key] = out_component.output_values[out_key]

                input_values["clock"] = self.clock_phase
                try:
                    output_values = component.update_values(input_values)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    if setup:
                        changed = True
                        continue
                    else:
                        raise

                for out_key in component.output_keys:
                    assert out_key in output_values, "{} output missing key {}".format(component.name(), out_key)
                    assert output_values[out_key] is not None, "{} {}".format(component.name(), out_key)

                    if output_values[out_key] != component.output_values[out_key]:
                        changed = True

                    component.output_values[out_key] = output_values[out_key]

                assert len(component.output_keys) == len(output_values)
        return

    def step(self):
        self.clock()
        self.propagate()
        self.dump_component_outputs()

    def dump_component_outputs(self):
        for component in self.components:
            print "{}:".format(component.name())
            for out_key in component.output_keys:
                print "    {} = {}".format(out_key, component.output_values[out_key])

class Latch(Component):
    def __init__(self, value, clock_phase):
        super(Latch, self).__init__()

        self.add_input("in")
        self.add_input("freeze")

        self.add_output("out")

        self.enable_clock()

        self.value = value
        self.clock_phase = clock_phase

    def get_state(self):
        return {
            "value": self.value
        }

    def name(self):
        return "latch"

    def update_values(self, input_values):
        return {
            "out": self.value
        }

    def update_state(self, input_values):
        if input_values["clock"] == self.clock_phase and input_values["freeze"] == 0:
            self.value = input_values["in"]

class LatchWithReset(Latch):
    def __init__(self, value, clock_phase):
        super(LatchWithReset, self).__init__(value, clock_phase)

        self.add_input("reset")

        # print self.value
        self.reset_value = value

    def name(self):
        return "latch with reset"

    def update_state(self, input_values):
        if input_values["clock"] == self.clock_phase:
            if input_values["reset"] == 0:
                self.value = input_values["in"]

        # reset on any phase... not sure if this is good or bad
        if input_values["reset"] == 1:
            self.value = self.reset_value

class Incrementer(Component):
    def __init__(self):
        super(Incrementer, self).__init__()

        self.add_input("in")
        self.add_output("out")

    def name(self):
        return "incrementer"

    def update_values(self, input_values):
        return {
            "out": input_values["in"] + 1
        }

class Constant(Component):
    def __init__(self, value):
        super(Constant, self).__init__()

        self.add_output("out")
        self.value = value

    def name(self):
        return "constant"

    def update_values(self, input_values):
        return {
            "out": self.value
        }

class And(Component):
    def __init__(self, num_inputs):
        super(And, self).__init__()

        for i in xrange(num_inputs):
            self.add_input("input_{}".format(i))

        self.add_output("out")

    def name(self):
        return "and"

    def update_values(self, input_values):
        return {
            "out": 1 if all(input_values[k] for k in self.input_keys) else 0
        }

class Or(Component):
    def __init__(self, num_inputs):
        super(Or, self).__init__()

        for i in xrange(num_inputs):
            self.add_input("input_{}".format(i))

        self.add_output("out")

    def name(self):
        return "or"

    def update_values(self, input_values):
        return {
            "out": 1 if any(input_values[k] for k in self.input_keys) else 0
        }

class Multiplexer(Component):
    def __init__(self, num_inputs):
        super(Multiplexer, self).__init__()

        for i in xrange(num_inputs):
            self.add_input("input_{}".format(i))

        self.add_input("control")
        self.add_output("out")

    def name(self):
        return "multiplexer"

    def update_values(self, input_values):
        return {
            "out": input_values["input_{}".format(input_values["control"])]
        }

class DataMemory(Component):
    def __init__(self, memory, clock_phase):
        super(DataMemory, self).__init__()

        self.add_input("write")
        self.add_input("read")
        self.add_input("address")
        self.add_input("write_data")

        self.add_output("read_data")
        self.enable_clock()

        self.memory = {}

        for key, value in enumerate(memory):
            self.memory[key] = value

        self.read_data = 0
        self.clock_phase = clock_phase

    def get_state(self):
        return {
            "read_data": self.read_data,
            "memory": self.memory
        }

    def name(self):
        return "DataMemory"

    def update_values(self, input_values):
        return {
            "read_data": self.read_data
        }

    def update_state(self, input_values):
        if input_values["clock"] == self.clock_phase:
            assert not (input_values["read"] == 1 and input_values["write"] == 1)

            if input_values["read"] == 1:
                if input_values["address"] in self.memory:
                    self.read_data = self.memory[input_values["address"]]
                else:
                    self.read_data = 0

            if input_values["write"] == 1:
                self.memory[input_values["address"]] = input_values["write_data"]

class InstructionMemory(Component):
    def __init__(self, instructions, clock_phase):
        super(InstructionMemory, self).__init__()


        self.add_input("address")
        self.add_output("instruction")
        self.enable_clock()

        self.instructions = instructions
        self.instruction = ("NOOP",0,0,0,0)
        self.clock_phase = clock_phase

    def get_state(self):
        return {
            "instruction": self.instruction,
        }

    def name(self):
        return "InstructionMemory"

    def update_values(self, input_values):
        return {
            "instruction": self.instruction
        }

    def update_state(self, input_values):
        if input_values["clock"] == self.clock_phase:
            self.instruction = self.instructions[input_values["address"]]

class InstructionSplit(Component):
    instruction_keys = ["opcode","reg_read_sel1", "reg_read_sel2", "reg_write_sel", "immediate"]

    def __init__(self):
        super(InstructionSplit, self).__init__()

        self.add_input("instruction")

        for key in InstructionSplit.instruction_keys:
            self.add_output(key)

    def name(self):
        return "instruction split"

    def update_values(self, input_values):
        return dict(zip(InstructionSplit.instruction_keys,input_values["instruction"]))

class RegisterFile(Component):
    def name(self):
        return "register file"

    def __init__(self, num_registers, clock_phase):
        super(RegisterFile, self).__init__()

        # NOTE to make it more obvious
        # TODO remove
        self.registers = list(100 + i for i in xrange(num_registers))
        self.registers[0] = 0

        self.add_input("reg_read_sel1")
        self.add_input("reg_read_sel2")

        self.add_input("write_sel")
        self.add_input("write_enable")
        self.add_input("write_data")

        self.add_output("read_data1")
        self.add_output("read_data2")

        self.enable_clock()

        self.clock_phase = clock_phase

    def get_state(self):
        return {
            "registers": self.registers,
        }

    def update_values(self, input_values):
        # i think we are supposed to store which register to read from in ph0
        return {
            "read_data1": self.registers[input_values["reg_read_sel1"]],
            "read_data2": self.registers[input_values["reg_read_sel2"]]
        }

    def update_state(self, input_values):
        if input_values["clock"] == self.clock_phase:
            if input_values["write_enable"] == 1:
                # disallow writing to R0
                assert input_values["write_sel"] != 0, "Cannot change R0's value"

                self.registers[input_values["write_sel"]] = input_values["write_data"]

class ALU(Component):
    def name(self):
        return "ALU"

    def __init__(self):
        super(ALU, self).__init__()

        self.add_input("operand_a")
        self.add_input("operand_b")
        self.add_input("operation")

        # my ALU will only have 1 output instead of 1 for status and one for result
        self.add_output("out")

        self.add_output("busy")
        self.cycles_left = 0

        self.enable_clock()

    operations = {
        "ADD": lambda ins: ins[0] + ins[1],
        "SUB": lambda ins: ins[0] - ins[1],
        "DIV": lambda ins: 0 if ins[1] == 0 else ins[0] / ins[1], # really this should be raised on ph1?
        "MUL": lambda ins: ins[0] * ins[1],
        "AND": lambda ins: ins[0] & ins[1],
        "OR": lambda ins: ins[0] | ins[1],
        "XOR": lambda ins: ins[0] ^ ins[1],
        "EQ": lambda ins: 1 if ins[0] == ins[1] else 0,
        "NE": lambda ins: 1 if ins[0] != ins[1] else 0,
        "GEZ": lambda ins: 1 if ins[0] >= 0 else 0,
        "GTZ": lambda ins: 1 if ins[0] >  0 else 0,
        "LEZ": lambda ins: 1 if ins[0] <= 0 else 0,
        "LTZ": lambda ins: 1 if ins[0] <  0 else 0,
        "TRUE": lambda _: 1
    }

    operation_delay = {
        "ADD": 0,
        "SUB": 0,
        "DIV": 5,
        "MUL": 5,
        "AND": 0,
        "OR": 0,
        "XOR": 0,
        "EQ": 0,
        "NE": 0,
        "GEZ": 0,
        "GTZ": 0,
        "LEZ": 0,
        "LTZ": 0,
        "TRUE": 0,
    }

    @staticmethod
    def compute(inputs, operation):
        return ALU.operations[operation](inputs)

    # i think real ALUs take both phases, but mine will just use 1 phase? what about division
    def update_values(self, input_values):
        return {
            "busy": 1 if self.cycles_left > 0 else 0,
            "out": 0 if self.cycles_left > 0 else ALU.compute((input_values["operand_a"], input_values["operand_b"]), input_values["operation"])
        }

    # need some sort of busy tag to say to ourselves and others
    # need to let values pass through first, also depending on the operation/status of the operation set the busy flag to true -> keeps divisor same because input latches dont change

    def update_state(self, input_values):
        # on ph1
        if input_values["clock"] == 1:
            if self.cycles_left == 0:
                self.cycles_left = self.operation_delay[input_values["operation"]]
            else:
                self.cycles_left -= 1


class Decoder(Component):
    def name(self):
        return "decoder"

    @staticmethod
    def get_ALU_op(opcode):
        # NOTE this only works because none of them begin with B or end with I
        if opcode[-1:] == "I":
            return opcode[:-1]
        if opcode[:1]  == "B":
            return opcode[1:]

        return opcode

    def __init__(self):
        super(Decoder, self).__init__()

        self.add_input("opcode")

        self.add_output("ALU_op")
        self.add_output("ALU_source") # 0 = read_sel2 1 = immediate

        self.add_output("WB_enable")
        self.add_output("WB_EXEC_or_MEM")

        self.add_output("is_branch")
        self.add_output("branch_addr_or_reg")

        self.add_output("MEM_write")
        self.add_output("MEM_read")

    def update_values(self, input_values):
        opcode = input_values["opcode"]

        output_values = {
            "WB_enable": 0,
            "WB_EXEC_or_MEM": 0,
            "ALU_op": "ADD",
            "ALU_source": 0,
            "is_branch": 0,
            "branch_addr_or_reg": 0,
            "MEM_write": 0,
            "MEM_read":  0
        }

        if  opcode in ["ADD","SUB","DIV","MUL","AND","OR" ,"XOR"]:
            output_values["WB_enable"]  = 1

            output_values["ALU_op"]     = Decoder.get_ALU_op(opcode)

        elif opcode in ["ADDI","SUBI","DIVI","MULI","ANDI","ORI","XORI"]:
            output_values["WB_enable"]  = 1

            output_values["ALU_op"]     = Decoder.get_ALU_op(opcode)
            output_values["ALU_source"] = 1

        elif opcode in ["BEQ","BNE"]:
            output_values["ALU_op"]    = Decoder.get_ALU_op(opcode)

            output_values["is_branch"] = 1

        elif opcode in ["BGEZ","BGTZ","BLEZ","BLTZ"]:
            output_values["ALU_op"]    = Decoder.get_ALU_op(opcode)

            output_values["is_branch"] = 1

        elif opcode in ["J"]:
            output_values["ALU_op"]    = "TRUE"

            output_values["is_branch"] = 1

        elif opcode in ["JR"]:
            output_values["ALU_op"]    = "TRUE"

            output_values["is_branch"] = 1
            output_values["branch_addr_or_reg"] = 1

        elif opcode in ["LOAD"]:
            output_values["WB_enable"]      = 1
            output_values["WB_EXEC_or_MEM"] = 1

            output_values["ALU_op"]     = "ADD"
            output_values["ALU_source"] = 1

            output_values["MEM_read"]  = 1

        elif opcode in ["STOR"]:
            output_values["ALU_op"]     = "ADD"
            output_values["ALU_source"] = 1

            output_values["MEM_write"] = 1

        return output_values

PC    = Latch(0, clock_phase=0)
incPC = Incrementer()
PCmux = Multiplexer(2)

import programs.add_test
fetcher = InstructionMemory(programs.add_test.instructions, clock_phase=1)

instructionLatch = LatchWithReset(("NOOP",0,0,0,0), clock_phase=0)
instructionSplit = InstructionSplit()

decoder = Decoder()
registerFile = RegisterFile(32, clock_phase=0)

# NOTE need to bring through all of register file and decoders outputs into execute phase
decoder_exec_latches = {}
for out_key in ["WB_enable", "WB_EXEC_or_MEM", "ALU_op", "ALU_source", "is_branch", "branch_addr_or_reg", "MEM_write", "MEM_read"]:
    decoder_exec_latches[out_key] = Latch("ADD" if out_key == "ALU_op" else 0, clock_phase=0)
    decoder_exec_latches[out_key].name = functools.partial(lambda x: x, "{} latch Ex".format(out_key))

registerFile_exec_latches = {}
for out_key in ["read_data1","read_data2"]:
    registerFile_exec_latches[out_key] = Latch(0, clock_phase=0)
    registerFile_exec_latches[out_key].name = functools.partial(lambda x: x, "{} latch Ex".format(out_key))

alu = ALU()
alu_source_mux = Multiplexer(2)

branch_addr_mux = Multiplexer(2)
do_branch = And(2)

write_back_sel_ex = Latch(0, clock_phase=0)
immediate_ex = Latch(0, clock_phase=0)

write_back_sel_mem = Latch(0, clock_phase=0)
alu_result_mem = Latch(0, clock_phase=0)
write_data_mem = Latch(0, clock_phase=0)

mem_write_mem = Latch(0, clock_phase=0)
mem_read_mem = Latch(0, clock_phase=0)
wb_enable_mem = Latch(0, clock_phase=0)

data_memory = DataMemory(programs.add_test.memory, clock_phase=1)

write_back_sel_wb = Latch(0, clock_phase=0)

alu_result_wb = Latch(0, clock_phase=0)
mem_read_data_wb = Latch(0, clock_phase=0)
wb_value_mux = Multiplexer(2)

wb_exec_or_mem_mem = Latch(0, clock_phase=0)
wb_exec_or_mem_wb  = Latch(0, clock_phase=0)

wb_enable_wb = Latch(0, clock_phase=0)

freeze_pipeline = Or(1)
freeze_pipeline.name = lambda: "freeze pipeline"

# Setup names for debugging
PC.name               = lambda: "PC"
incPC.name            = lambda: "inc PC"
PCmux.name            = lambda: "PC mux"
instructionLatch.name = lambda: "instruction latch"
alu_source_mux.name   = lambda: "alu source mux"
branch_addr_mux.name  = lambda: "branch address mux"
do_branch.name        = lambda: "do branch"
write_back_sel_ex.name = lambda: "write back sel Ex"
immediate_ex.name = lambda: "immediate Ex"
write_back_sel_mem.name = lambda: "write back sel Mem"
alu_result_mem.name = lambda: "alu result Mem"
write_data_mem.name = lambda: "write data Mem"

mem_write_mem.name = lambda: "mem write Mem"
mem_read_mem.name = lambda: "mem read Mem"
wb_enable_mem.name = lambda: "WB enable Mem"

write_back_sel_wb.name = lambda: "write back sel Wb"
write_back_sel_ex.name = lambda: "write back sel Ex"

alu_result_wb.name = lambda: "alu result Wb"
mem_read_data_wb.name = lambda: "mem read data Wb"
wb_value_mux.name = lambda: "wb value mux"
wb_exec_or_mem_mem.name = lambda: "wb exec or mem Mem"
wb_exec_or_mem_wb.name = lambda: "wb exec or mem Wb"
wb_enable_wb.name = lambda: "wb enable Wb"


# need to pass control signal to registerfile to tell it to write

cco = ComponentConnectionOrchestrator()
cco.add_component(PC)
cco.add_component(incPC)
cco.add_component(PCmux)
cco.add_component(fetcher)
cco.add_component(instructionLatch)
cco.add_component(instructionSplit)
cco.add_component(decoder)
cco.add_component(registerFile)
cco.add_component(write_back_sel_ex)
cco.add_component(immediate_ex)
cco.add_component(alu)
cco.add_component(alu_source_mux)
cco.add_component(branch_addr_mux)
cco.add_component(do_branch)

for component in decoder_exec_latches.values():
    cco.add_component(component)

for component in registerFile_exec_latches.values():
    cco.add_component(component)

cco.add_component(write_back_sel_mem)
cco.add_component(alu_result_mem)
cco.add_component(write_data_mem)
cco.add_component(mem_write_mem)
cco.add_component(mem_read_mem)
cco.add_component(wb_enable_mem)
cco.add_component(wb_exec_or_mem_mem)

cco.add_component(data_memory)

cco.add_component(wb_exec_or_mem_wb)
cco.add_component(write_back_sel_wb)

cco.add_component(alu_result_wb)
cco.add_component(mem_read_data_wb)
cco.add_component(wb_value_mux)
cco.add_component(wb_enable_wb)

cco.add_component(freeze_pipeline)

cco.add_connection((alu,"busy"),(freeze_pipeline,"input_0"))

cco.add_connection((PCmux,"out"),(PC,   "in"))
cco.add_connection((PC,   "out"),(incPC,"in"))

cco.add_connection((incPC,     "out"),(PCmux,"input_0"))
cco.add_connection((branch_addr_mux,"out"),(PCmux,"input_1"))
cco.add_connection((do_branch,  "out"),(PCmux,"control"))

cco.add_connection((PC,   "out"),(fetcher,"address"))

cco.add_connection((fetcher,"instruction"),(instructionLatch,"in"))
cco.add_connection((do_branch, "out"), (instructionLatch, "reset"))

cco.add_connection((instructionLatch, "out"), (instructionSplit, "instruction"))

cco.add_connection((instructionSplit,"opcode"),(decoder,"opcode"))
cco.add_connection((instructionSplit,"reg_read_sel1"),(registerFile,"reg_read_sel1"))
cco.add_connection((instructionSplit,"reg_read_sel2"),(registerFile,"reg_read_sel2"))

for out_key, component in decoder_exec_latches.iteritems():
    cco.add_connection((decoder, out_key),(component, "in"))
    cco.add_connection((freeze_pipeline, "out"),(component, "freeze"))

for out_key, component in registerFile_exec_latches.iteritems():
    cco.add_connection((registerFile, out_key),(component, "in"))
    cco.add_connection((freeze_pipeline, "out"),(component, "freeze"))

cco.add_connection((decoder_exec_latches["ALU_op"], "out"), (alu, "operation"))
cco.add_connection((registerFile_exec_latches["read_data1"], "out"), (alu, "operand_a"))

cco.add_connection((instructionSplit,"immediate"),       (immediate_ex, "in"))

cco.add_connection((registerFile_exec_latches["read_data2"], "out"), (alu_source_mux, "input_0"))
cco.add_connection((immediate_ex,"out"),       (alu_source_mux, "input_1"))

cco.add_connection((decoder_exec_latches["ALU_source"], "out"), (alu_source_mux, "control"))

cco.add_connection((alu_source_mux, "out"), (alu, "operand_b"))

cco.add_connection((immediate_ex,"out"), (branch_addr_mux, "input_0"))
cco.add_connection((registerFile_exec_latches["read_data1"], "out"), (branch_addr_mux, "input_1"))
cco.add_connection((decoder_exec_latches["branch_addr_or_reg"], "out"), (branch_addr_mux, "control"))

cco.add_connection((alu, "out"), (do_branch, "input_0"))
cco.add_connection((decoder_exec_latches["is_branch"], "out"), (do_branch, "input_1"))

cco.add_connection((decoder_exec_latches["MEM_write"],"out"),(mem_write_mem,"in"))
cco.add_connection((decoder_exec_latches["MEM_read"],"out"),(mem_read_mem,"in"))
cco.add_connection((decoder_exec_latches["WB_enable"],"out"),(wb_enable_mem,"in"))

cco.add_connection((alu,"out"),(alu_result_mem,"in"))
cco.add_connection((registerFile_exec_latches["read_data2"],"out"),(write_data_mem,"in"))

cco.add_connection((instructionSplit,"reg_write_sel"),(write_back_sel_ex,"in"))
cco.add_connection((write_back_sel_ex,"out"),(write_back_sel_mem,"in"))
cco.add_connection((write_back_sel_mem,"out"),(write_back_sel_wb,"in"))
cco.add_connection((write_back_sel_wb,"out"),(registerFile,"write_sel"))

cco.add_connection((alu_result_mem,"out"),(data_memory,"address"))
cco.add_connection((write_data_mem,"out"),(data_memory,"write_data"))

cco.add_connection((mem_read_mem,"out"),(data_memory,"read"))
cco.add_connection((mem_write_mem,"out"),(data_memory,"write"))

cco.add_connection((alu_result_mem,"out"),(alu_result_wb,"in"))
cco.add_connection((data_memory,"read_data"),(mem_read_data_wb,"in"))

cco.add_connection((alu_result_wb,"out"),(wb_value_mux,"input_0"))
cco.add_connection((mem_read_data_wb,"out"),(wb_value_mux,"input_1"))
cco.add_connection((wb_exec_or_mem_wb,"out"),(wb_value_mux,"control"))

cco.add_connection((wb_value_mux,"out"),(registerFile,"write_data"))

cco.add_connection((decoder_exec_latches["WB_EXEC_or_MEM"], "out"),(wb_exec_or_mem_mem, "in"))
cco.add_connection((wb_exec_or_mem_mem, "out"),(wb_exec_or_mem_wb, "in"))

cco.add_connection((wb_enable_mem, "out"),(wb_enable_wb, "in"))
cco.add_connection((wb_enable_wb, "out"),(registerFile, "write_enable"))

cco.add_connection((freeze_pipeline,"out"),(PC,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(instructionLatch,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(write_back_sel_ex,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(immediate_ex,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(write_back_sel_mem,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(alu_result_mem,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(write_data_mem,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(mem_write_mem,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(mem_read_mem,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(wb_enable_mem,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(write_back_sel_wb,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(alu_result_wb,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(mem_read_data_wb,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(wb_exec_or_mem_mem,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(wb_exec_or_mem_wb,"freeze"))
cco.add_connection((freeze_pipeline,"out"),(wb_enable_wb,"freeze"))

cco.propagate(True)

cco.generate_nomnoml()
