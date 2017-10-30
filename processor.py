import functools
import json

from collections import OrderedDict

# -a note someone made here:http://slideplayer.com/slide/9255919/
# -https://www.cise.ufl.edu/~mssz/CompOrg/Figure5.6-PipelineControlLines.gif

class Component(object):
    def __init__(self):
        self.input_keys  = []
        self.output_keys = []

        self.output_values = {}

        self.clocked = False

    '''
    dont actually use this yet
    '''
    def get_state(self):
        return {}

    def update_values(self, input_values):
        raise NotImplementedError('Components must override update_values()!')

    def update_state(self, input_values):
        assert self.clocked == True
        raise NotImplementedError('Components must override update_state()!')

    def add_input(self, key):
        self.input_keys.append(key)

    def add_output(self, key):
        self.output_keys.append(key)
        self.output_values[key] = None # before update

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
        assert isinstance(component, Component)
        assert component not in self.components

        self.components.append(component)

    def add_connection(self, (out_component, out_key), (in_component, in_key)):
        assert ((out_component, out_key), (in_component, in_key)) not in self.connections
        assert in_component in self.components
        assert out_component in self.components
        assert out_key in out_component.output_keys, "{} {}".format(out_key, out_component)
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
                    assert out_key in output_values, "{} output missing key {}".format(component, out_key)
                    assert output_values[out_key] is not None, "{} {}".format(component, out_key)

                    if output_values[out_key] != component.output_values[out_key]:
                        changed = True

                    component.output_values[out_key] = output_values[out_key]

                assert len(component.output_keys) == len(output_values)
        return

    def step(self):
        self.clock()
        self.propagate()
        # self.dump_component_outputs()


    def step_multi(self):
        i = raw_input("Enter e to stop. Enter nothing and press enter to step 1 phase ")
        while i != "e":
            self.step()
            i = raw_input("Enter e to stop. Enter nothing and press enter to step 1 phase ")

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

    def update_state(self, input_values):
        if input_values["clock"] == self.clock_phase:
            if input_values["reset"] == 0:
                self.value = input_values["in"]

        # reset on any phase... not sure if this is good or bad
        if input_values["reset"] == 1:
            self.value = self.reset_value

class PipelineBuffer(Component):
    def __init__(self):
        super(PipelineBuffer, self).__init__()

        self.names = OrderedDict()

        self.add_input("reset")
        self.add_input("freeze")

        self.enable_clock()

    def add_name(self, name, value):
        assert name not in ["reset","freeze"]

        self.add_input(name)
        self.add_output(name)

        self.names[name] = {
            "value": value,
            "default": value
        }

    def update_values(self, input_values):
        return dict((k, v["value"]) for k, v in self.names.iteritems())


    def update_state(self, input_values):
        if input_values["clock"] == 0 and input_values["freeze"] == 0:
            for name in self.names.keys():
                self.names[name]["value"] = input_values[name]

        if input_values["reset"] == 1:
            for name in self.names.keys():
                self.names[name]["value"] = self.names[name]["default"]


class Incrementer(Component):
    def __init__(self):
        super(Incrementer, self).__init__()

        self.add_input("in")
        self.add_output("out")

    def update_values(self, input_values):
        return {
            "out": input_values["in"] + 1
        }

class Constant(Component):
    def __init__(self, value):
        super(Constant, self).__init__()

        self.add_output("out")
        self.value = value

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

    def update_values(self, input_values):
        return dict(zip(InstructionSplit.instruction_keys,input_values["instruction"]))

class RegisterFile(Component):
    def __init__(self, num_registers, clock_phase):
        super(RegisterFile, self).__init__()

        self.registers = list(0 for i in xrange(num_registers))

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
    @staticmethod
    def get_alu_op(opcode):
        # NOTE this only works because none of them begin with B or end with I
        if opcode[-1:] == "I":
            return opcode[:-1]
        if opcode[:1]  == "B":
            return opcode[1:]

        return opcode

    def __init__(self):
        super(Decoder, self).__init__()

        self.add_input("opcode")

        self.add_output("alu_op")
        self.add_output("alu_source") # 0 = read_sel2 1 = immediate

        self.add_output("wb_enable")
        self.add_output("wb_exec_or_mem")

        self.add_output("is_branch")
        self.add_output("branch_imm_or_reg")

        self.add_output("mem_write")
        self.add_output("mem_read")

    def update_values(self, input_values):
        opcode = input_values["opcode"]

        output_values = {
            "wb_enable": 0,
            "wb_exec_or_mem": 0,
            "alu_op": "ADD",
            "alu_source": 0,
            "is_branch": 0,
            "branch_imm_or_reg": 0,
            "mem_write": 0,
            "mem_read":  0
        }

        if  opcode in ["ADD","SUB","DIV","MUL","AND","OR" ,"XOR"]:
            output_values["wb_enable"]  = 1

            output_values["alu_op"]     = Decoder.get_alu_op(opcode)

        elif opcode in ["ADDI","SUBI","DIVI","MULI","ANDI","ORI","XORI"]:
            output_values["wb_enable"]  = 1

            output_values["alu_op"]     = Decoder.get_alu_op(opcode)
            output_values["alu_source"] = 1

        elif opcode in ["BEQ","BNE"]:
            output_values["alu_op"]    = Decoder.get_alu_op(opcode)

            output_values["is_branch"] = 1

        elif opcode in ["BGEZ","BGTZ","BLEZ","BLTZ"]:
            output_values["alu_op"]    = Decoder.get_alu_op(opcode)

            output_values["is_branch"] = 1

        elif opcode in ["J"]:
            output_values["alu_op"]    = "TRUE"

            output_values["is_branch"] = 1

        elif opcode in ["JR"]:
            output_values["alu_op"]    = "TRUE"

            output_values["is_branch"] = 1
            output_values["branch_imm_or_reg"] = 1

        elif opcode in ["LOAD"]:
            output_values["wb_enable"]      = 1
            output_values["wb_exec_or_mem"] = 1

            output_values["alu_op"]     = "ADD"
            output_values["alu_source"] = 1

            output_values["mem_read"]  = 1

        elif opcode in ["STOR"]:
            output_values["alu_op"]     = "ADD"
            output_values["alu_source"] = 1

            output_values["mem_write"] = 1

        return output_values

class Processor(ComponentConnectionOrchestrator):
    def __getitem__(self, key):
        return self.object_lookup[key]

    def get_name(self, component):
        return self.name_lookup[component]

    def __setitem__(self, key, value):
        self.add_component(value)
        self.object_lookup[key] = value
        self.name_lookup[value] = key

    def add_connection(self, (out_name, out_key), (in_name, in_key)):
        out_component = self[out_name]
        in_component  = self[in_name]
        super(Processor, self).add_connection((out_component, out_key), (in_component, in_key))

    def __init__(self, instructions, data):
        super(Processor, self).__init__()

        self.object_lookup = {}
        self.name_lookup = {}

        p = self

        p["pc"] = Latch(0, clock_phase=0)
        p["inc_pc"] = Incrementer()
        p["pc_mux"] = Multiplexer(2)

        p["fetcher"] = InstructionMemory(instructions, clock_phase=1)

        p["decode_buffer"] = PipelineBuffer()
        p["decode_buffer"].add_name("instruction", ("NOOP",0,0,0,0))

        p["instruction_split"] = InstructionSplit()

        p["decoder"] = Decoder()
        p["register_file"] = RegisterFile(32, clock_phase=0)

        p["execute_buffer_reset"] = Constant(0)
        p["execute_buffer"] = PipelineBuffer()
        p["execute_buffer"].add_name("reg_write_sel", 0)
        p["execute_buffer"].add_name("immediate", 0)

        for out_key in ["alu_op", "alu_source", "wb_enable", "wb_exec_or_mem", "is_branch", "branch_imm_or_reg", "mem_write", "mem_read"]:
            p["execute_buffer"].add_name(out_key, "ADD" if out_key == "alu_op" else 0)

        for out_key in ["read_data1","read_data2"]:
            p["execute_buffer"].add_name(out_key, 0)

        p["alu"] = ALU()
        p["alu_source_mux"] = Multiplexer(2)

        p["branch_addr_mux"] = Multiplexer(2)
        p["do_branch"] = And(2)

        p["mem_buffer_reset"] = Constant(0)
        p["mem_buffer"] = PipelineBuffer()
        p["mem_buffer"].add_name("reg_write_sel", 0)
        p["mem_buffer"].add_name("mem_write", 0)
        p["mem_buffer"].add_name("mem_read", 0)
        p["mem_buffer"].add_name("wb_enable", 0)
        p["mem_buffer"].add_name("wb_exec_or_mem", 0)
        p["mem_buffer"].add_name("alu_result", 0)
        p["mem_buffer"].add_name("write_data", 0)

        p["data_memory"] = DataMemory(data, clock_phase=1)

        p["wb_buffer_reset"] = Constant(0)
        p["wb_buffer"] = PipelineBuffer()
        p["wb_buffer"].add_name("reg_write_sel",0)
        p["wb_buffer"].add_name("mem_read_data",0)
        p["wb_buffer"].add_name("wb_enable",0)
        p["wb_buffer"].add_name("wb_exec_or_mem",0)
        p["wb_buffer"].add_name("alu_result",0)

        p["wb_value_mux"] = Multiplexer(2)

        p["freeze_pipeline"] = Or(1)

        ############################# CONNECTIONS #############################

        p.add_connection(("alu","busy"),("freeze_pipeline","input_0"))

        p.add_connection(("pc_mux","out"),("pc",   "in"))
        p.add_connection(("pc",   "out"),("inc_pc","in"))

        p.add_connection(("inc_pc",     "out"),("pc_mux","input_0"))
        p.add_connection(("branch_addr_mux","out"),("pc_mux","input_1"))
        p.add_connection(("do_branch",  "out"),("pc_mux","control"))

        p.add_connection(("pc",   "out"),("fetcher","address"))

        p.add_connection(("fetcher","instruction"),("decode_buffer","instruction"))
        p.add_connection(("do_branch", "out"), ("decode_buffer", "reset"))

        p.add_connection(("decode_buffer", "instruction"), ("instruction_split", "instruction"))

        p.add_connection(("instruction_split","opcode"),("decoder","opcode"))
        p.add_connection(("instruction_split","reg_read_sel1"),("register_file","reg_read_sel1"))
        p.add_connection(("instruction_split","reg_read_sel2"),("register_file","reg_read_sel2"))

        for key in ["wb_enable", "wb_exec_or_mem", "alu_op", "alu_source", "is_branch", "branch_imm_or_reg", "mem_write", "mem_read"]:
            p.add_connection(("decoder", key),("execute_buffer", key))

        for key in ["read_data1","read_data2"]:
            p.add_connection(("register_file", key),("execute_buffer", key))

        p.add_connection(("instruction_split","immediate"),    ("execute_buffer", "immediate"))
        p.add_connection(("instruction_split","reg_write_sel"),("execute_buffer","reg_write_sel"))

        p.add_connection(("execute_buffer_reset", "out"), ("execute_buffer", "reset"))

        p.add_connection(("execute_buffer", "alu_op"), ("alu", "operation"))
        p.add_connection(("execute_buffer", "read_data1"), ("alu", "operand_a"))

        p.add_connection(("execute_buffer","read_data2"), ("alu_source_mux", "input_0"))
        p.add_connection(("execute_buffer","immediate"),  ("alu_source_mux", "input_1"))
        p.add_connection(("execute_buffer","alu_source"), ("alu_source_mux", "control"))
        p.add_connection(("execute_buffer","immediate"), ("branch_addr_mux", "input_0"))
        p.add_connection(("execute_buffer","read_data1"), ("branch_addr_mux", "input_1"))
        p.add_connection(("execute_buffer","branch_imm_or_reg"), ("branch_addr_mux", "control"))
        p.add_connection(("execute_buffer","is_branch"), ("do_branch", "input_1"))
        p.add_connection(("execute_buffer","mem_write"),("mem_buffer","mem_write"))
        p.add_connection(("execute_buffer","mem_read"),("mem_buffer","mem_read"))
        p.add_connection(("execute_buffer","wb_enable"),("mem_buffer","wb_enable"))
        p.add_connection(("execute_buffer","read_data2"),("mem_buffer","write_data"))
        p.add_connection(("execute_buffer","reg_write_sel"),("mem_buffer","reg_write_sel"))
        p.add_connection(("execute_buffer","wb_exec_or_mem"),("mem_buffer", "wb_exec_or_mem"))


        p.add_connection(("alu_source_mux", "out"), ("alu", "operand_b"))
        p.add_connection(("alu", "out"), ("do_branch", "input_0"))
        p.add_connection(("alu","out"),("mem_buffer","alu_result"))

        p.add_connection(("mem_buffer_reset", "out"), ("mem_buffer", "reset"))

        p.add_connection(("mem_buffer","reg_write_sel"),("wb_buffer","reg_write_sel"))
        p.add_connection(("mem_buffer","alu_result"),("data_memory","address"))
        p.add_connection(("mem_buffer","write_data"),("data_memory","write_data"))
        p.add_connection(("mem_buffer","mem_read"),("data_memory","read"))
        p.add_connection(("mem_buffer","mem_write"),("data_memory","write"))
        p.add_connection(("mem_buffer","alu_result"),("wb_buffer","alu_result"))
        p.add_connection(("mem_buffer", "wb_exec_or_mem"),("wb_buffer", "wb_exec_or_mem"))
        p.add_connection(("mem_buffer", "wb_enable"),("wb_buffer", "wb_enable"))

        p.add_connection(("wb_buffer_reset", "out"), ("wb_buffer", "reset"))

        p.add_connection(("data_memory","read_data"),("wb_buffer","mem_read_data"))
        p.add_connection(("wb_buffer","reg_write_sel"),("register_file","write_sel"))
        p.add_connection(("wb_buffer","alu_result"),("wb_value_mux","input_0"))
        p.add_connection(("wb_buffer","mem_read_data"),("wb_value_mux","input_1"))
        p.add_connection(("wb_buffer","wb_exec_or_mem"),("wb_value_mux","control"))
        p.add_connection(("wb_buffer", "wb_enable"),("register_file", "write_enable"))

        p.add_connection(("wb_value_mux","out"),("register_file","write_data"))

        p.add_connection(("freeze_pipeline","out"),("pc","freeze"))
        p.add_connection(("freeze_pipeline","out"),("decode_buffer","freeze"))
        p.add_connection(("freeze_pipeline","out"),("execute_buffer", "freeze"))
        p.add_connection(("freeze_pipeline","out"),("mem_buffer","freeze"))
        p.add_connection(("freeze_pipeline","out"),("wb_buffer","freeze"))

        p.propagate(True)
