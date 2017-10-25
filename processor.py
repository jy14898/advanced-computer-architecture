from memory import Memory
import functools

# new system will re-introduce triggering values in terms of what caused the update
# but we will only recieve 1 update per system change per component, so no need to worry about timing
#
# i think that's better??!?!?!
# -a note someone made here:http://slideplayer.com/slide/9255919/
# -https://www.cise.ufl.edu/~mssz/CompOrg/Figure5.6-PipelineControlLines.gif

'''
After branch I need to clear the thing because could be something other than nop?

'''
class Component(object):
    def __init__(self):
        self.input_keys  = []
        self.output_keys = []

        self.output_values = {}

        self.clocked = False

    def name(self):
        return "component"

    # takes set of inputs and produces an output
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

    '''
    if you need to be able to decide which phase to clock on, make it an argument of __self__
    and use that to decide when to trigger
    '''
    def enable_clock(self):
        self.clocked = True

class ComponentConnectionOrchestrator(object):
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

    def propagate(self):
        component_levels = {}
        for component in self.components:
            component_levels[component] = 0

        def recurse(component, path):
            if component in path:
                return

            if component_levels[component] < len(path):
                component_levels[component] = len(path)

            path = list(path)
            path.append(component)

            for out_key in component.output_keys:
                outgoing_connections = list(set(c for ((a,b),(c,d)) in self.connections if (a,b) == (component,out_key)))

                for out_component in outgoing_connections:
                    recurse(out_component, path)

        # Any clocked component is good enough to be a start
        # actually no. this is only working by chance at the moment
        start_component = (c for c in self.components if c.clocked).next()
        recurse(start_component, [])

        levels = {}
        for component, level in component_levels.iteritems():
            if level not in levels:
                levels[level] = []

            levels[level].append(component)

        # print levels

        for level, components in levels.iteritems():
            print level
            for component in components:
                print component.name()
                incoming_connections = list(((a,b),(d)) for ((a,b),(c,d)) in self.connections if c == component)

                input_values = {}
                for ((out_component,out_key),in_key) in incoming_connections:
                    input_values[in_key] = out_component.output_values[out_key]

                input_values["clock"] = self.clock_phase
                output_values = component.update_values(input_values)

                for out_key in component.output_keys:
                    assert out_key in output_values, "{} output missing key {}".format(component.name(), out_key)

                    component.output_values[out_key] = output_values[out_key]

                assert len(component.output_keys) == len(output_values)

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
        self.add_output("out")

        self.enable_clock()

        self.value = value
        self.clock_phase = clock_phase

    def name(self):
        return "latch"

    def update_values(self, input_values):
        return {
            "out": self.value
        }

    def update_state(self, input_values):
        if input_values["clock"] == self.clock_phase:
            self.value = input_values["in"]

class LatchWithReset(Latch):
    def __init__(self, value, clock_phase):
        super(Latch, value, clock_phase).__init__()

        self.add_input("reset")

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
        print input_values
        return {
            "out": 1 if all(input_values[k] for k in self.input_keys) else 0
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

class Fetcher(Component):
    def __init__(self, instructions, clock_phase):
        super(Fetcher, self).__init__()


        self.add_input("address")
        self.add_output("instruction")
        self.enable_clock()

        self.instructions = instructions
        self.instruction = ("NOOP",0,0,0,0)
        self.clock_phase = clock_phase

    def name(self):
        return "fetcher"

    def update_values(self, input_values):
        return {
            "instruction": self.instruction
        }

    def update_state(self, input_values):
        if input_values["clock"] == self.clock_phase:
            self.instruction = self.instructions[input_values["address"]]

class RegisterFile(Component):
    def name(self):
        return "register file"

    def __init__(self, num_registers, clock_phase):
        super(RegisterFile, self).__init__()

        # NOTE to make it more obvious
        # TODO remove
        self.registers = list(100 + i for i in xrange(num_registers))
        self.registers[0] = 0

        self.add_input("instruction")

        self.add_input("write_sel")
        self.add_input("write_enable")
        self.add_input("write_data")

        self.add_output("read_data1")
        self.add_output("read_data2")

        self.enable_clock()

        self.clock_phase = clock_phase

    def update_values(self, input_values):
        # i think we are supposed to store which register to read from in ph0
        return {
            "read_data1": self.registers[input_values["instruction"][1]],
            "read_data2": self.registers[input_values["instruction"][2]]
        }

    def update_state(self, input_values):
        if input_values["clock"] == self.clock_phase:
            pass

#        # define clock high to mean write
#        if input_values["clock"] == 1:
#            # if input_values["write_enable"] == 1:
#            #     # disallow writing to R0
#            #     assert input_values["write_sel"] != 0, "Cannot change R0's value"
#            #
#            #     self.registers[input_values["write_sel"]] = input_values["write_data"]
#            pass
#        # define clock low to mean read
#        else:
#            pass

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

    operations = {
        "ADD": lambda ins: ins[0] + ins[1],
        "SUB": lambda ins: ins[0] - ins[1],
        "DIV": lambda ins: ins[0] / ins[1],
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

    @staticmethod
    def compute(inputs, operation):
        return ALU.operations[operation](inputs)

    # i think real ALUs take both phases, but mine will just use 1 phase? what about division
    def update_values(self, input_values):
        return {
            "out": ALU.compute((input_values["operand_a"], input_values["operand_b"]), input_values["operation"])
        }

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

        self.add_input("instruction")

        self.add_output("ALU_op")
        self.add_output("ALU_source") # 0 = read_sel2 1 = immediate

        self.add_output("WB_enable")
        self.add_output("WB_EXEC_or_MEM")

        self.add_output("is_branch")
        self.add_output("branch_addr_or_reg")

        self.add_output("MEM_write")
        self.add_output("MEM_read")

        self.add_output("immediate")


    def update_values(self, input_values):
        (opcode, _, _, _, imm) = input_values["instruction"]

        output_values = {
            "WB_enable": 0,
            "WB_EXEC_or_MEM": 0,
            "ALU_op": "ADD",
            "ALU_source": 0,
            "is_branch": 0,
            "branch_addr_or_reg": 0,
            "MEM_write": 0,
            "MEM_read":  0,
            # putting these here because cant really put it anywhere else
            # need to add a splitting gate
            "immediate": imm
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

PCmux      = Multiplexer(2)

import programs.add_test
fetcher = Fetcher(programs.add_test.instructions, clock_phase=1)

instructionLatch = Latch(("NOOP",0,0,0,0), clock_phase=0)

decoder = Decoder()
registerFile = RegisterFile(32, clock_phase=0)

# NOTE need to bring through all of register file and decoders outputs into execute phase
decoder_exec_latches = {}
for out_key in ["WB_enable", "WB_EXEC_or_MEM", "ALU_op", "ALU_source", "is_branch", "branch_addr_or_reg", "MEM_write", "MEM_read", "immediate"]:
    decoder_exec_latches[out_key] = Latch("ADD" if out_key == "ALU_op" else 0, clock_phase=0)
    decoder_exec_latches[out_key].name = functools.partial(lambda x: x, "{} latch Ex".format(out_key))

registerFile_exec_latches = {}
for out_key in ["read_data1","read_data2"]:
    registerFile_exec_latches[out_key] = Latch(0, clock_phase=0)
    registerFile_exec_latches[out_key].name = functools.partial(lambda x: x, "{} latch Ex".format(out_key))


alu = ALU()
alu_source_mux = Multiplexer(2)

branch_addr_mux = Multiplexer(2)
# branch_addr_latch = Latch(0, clock_phase=0)

do_branch = And(2)

# Setup names for debugging
PC.name               = lambda: "PC"
incPC.name            = lambda: "inc PC"
PCmux.name            = lambda: "PC mux"
instructionLatch.name = lambda: "instruction latch"
alu_source_mux.name   = lambda: "alu source mux"
branch_addr_mux.name  = lambda: "branch address mux"
do_branch.name  = lambda: "do branch"

cco = ComponentConnectionOrchestrator()
cco.add_component(PC)
cco.add_component(incPC)
cco.add_component(PCmux)
cco.add_component(fetcher)
cco.add_component(instructionLatch)
cco.add_component(decoder)
cco.add_component(registerFile)
cco.add_component(alu)
cco.add_component(alu_source_mux)
cco.add_component(branch_addr_mux)
cco.add_component(do_branch)

for component in decoder_exec_latches.values():
    cco.add_component(component)

for component in registerFile_exec_latches.values():
    cco.add_component(component)

cco.add_connection((PCmux,"out"),(PC,   "in"))
cco.add_connection((PC,   "out"),(incPC,"in"))

cco.add_connection((incPC,     "out"),(PCmux,"input_0"))
cco.add_connection((branch_addr_mux,"out"),(PCmux,"input_1"))
cco.add_connection((do_branch,  "out"),(PCmux,"control"))

cco.add_connection((PC,   "out"),(fetcher,"address"))

cco.add_connection((fetcher,"instruction"),(instructionLatch,"in"))

cco.add_connection((instructionLatch,"out"),(decoder,"instruction"))
cco.add_connection((instructionLatch,"out"),(registerFile,"instruction"))

for out_key, component in decoder_exec_latches.iteritems():
    cco.add_connection((decoder, out_key),(component, "in"))

for out_key, component in registerFile_exec_latches.iteritems():
    cco.add_connection((registerFile, out_key),(component, "in"))

cco.add_connection((decoder_exec_latches["ALU_op"], "out"), (alu, "operation"))
cco.add_connection((registerFile_exec_latches["read_data1"], "out"), (alu, "operand_a"))

cco.add_connection((registerFile_exec_latches["read_data2"], "out"), (alu_source_mux, "input_0"))
cco.add_connection((decoder_exec_latches["immediate"], "out"),       (alu_source_mux, "input_1"))

cco.add_connection((decoder_exec_latches["ALU_source"], "out"), (alu_source_mux, "control"))

cco.add_connection((alu_source_mux, "out"), (alu, "operand_b"))


cco.add_connection((decoder_exec_latches["immediate"], "out"), (branch_addr_mux, "input_0"))
cco.add_connection((registerFile_exec_latches["read_data1"], "out"), (branch_addr_mux, "input_1"))
cco.add_connection((decoder_exec_latches["branch_addr_or_reg"], "out"), (branch_addr_mux, "control"))


cco.add_connection((alu, "out"), (do_branch, "input_0"))
cco.add_connection((decoder_exec_latches["is_branch"], "out"), (do_branch, "input_1"))

cco.propagate()
