import instruction
from memory import Memory
import functools

# new system will re-introduce triggering values in terms of what caused the update
# but we will only recieve 1 update per system change per component, so no need to worry about timing
#
# i think that's better??!?!?!
# -a note someone made here:http://slideplayer.com/slide/9255919/
# -https://www.cise.ufl.edu/~mssz/CompOrg/Figure5.6-PipelineControlLines.gif


class Component(object):
    next_id = 0

    def __init__(self):
        self.id = Component.next_id
        Component.next_id = Component.next_id + 1

        # NOTE: could remove one of these but it's nice to have uniform inputs/outputs
        self.input_keys  = []
        self.output_keys = []
        self.output_values = {}

    def name(self):
        raise NotImplementedError('Components must have a name()!')

    # takes set of inputs and produces an output
    def update(self, input_values, input_values_old):
        raise NotImplementedError('Components must override update()!')

    def add_input(self, key):
        self.input_keys.append(key)

    def add_output(self, key, value):
        self.output_keys.append(key)
        self.output_values[key] = value

class ComponentConnectionOrchestrator(object):
    def __init__(self):
        self.components = {
            #"id": component
        }

        self.connections = [
            # ((id_out, out_key), (id_in, in_key))
        ]

    def add_component(self, component):
        assert component.id not in self.components
        self.components[component.id] = component

    def add_connection(self, (id_out, out_key), (id_in, in_key)):
        assert ((id_out, out_key), (id_in, in_key)) not in self.connections
        self.connections.append(((id_out, out_key), (id_in, in_key)))

    # assume all out_keys are updated
    # could cache this
    def propagate(self, id_out):
        component_levels = {}
        for component_id in self.components:
            component_levels[component_id] = 0

        def recurse(component_id, path):
            if component_id in path:
                return

            if component_levels[component_id] < len(path):
                component_levels[component_id] = len(path)

            path = list(path)
            path.append(component_id)

            for out_key in self.components[component_id].output_keys:
                outgoing_connections = list(set(c for ((a,b),(c,d)) in self.connections if (a,b) == (component_id,out_key)))

                for id_out in outgoing_connections:
                    recurse(id_out, path)

        recurse(id_out, [])

        component_levels = {i: l for i, l in component_levels.iteritems() if l > 0}
        component_levels[id_out] = 0

        output_values_old = {
            #(id_out, out_key): value
        }

        for component_id in component_levels.keys():
            for out_key in self.components[component_id].output_keys:
                output_values_old[(component_id, out_key)] = self.components[component_id].output_values[out_key]

        levels = {}
        for component_id, level in component_levels.iteritems():
            if level not in levels:
                levels[level] = []

            levels[level].append(component_id)

        for level, component_ids in levels.iteritems():
            for component_id in component_ids:
                incoming_connections = list(((a,b),(d)) for ((a,b),(c,d)) in self.connections if c == component_id)

                input_values = {}
                input_values_old = {}
                for ((out_id,out_key),in_key) in incoming_connections:
                    input_values[in_key] = self.components[out_id].output_values[out_key]

                    if (out_id,out_key) in output_values_old:
                        input_values_old[in_key] = output_values_old[(out_id,out_key)]
                    #consider input to be 'constant' for this update as it wasnt changed
                    else:
                        input_values_old[in_key] = input_values[in_key]

                self.components[component_id].update(input_values, input_values_old)

    def dump_component_states(self):
        for component_id, component in self.components.iteritems():
            print "{}:".format(component.name())
            for out_key in component.output_keys:
                print "    {} = {}".format(out_key, component.output_values[out_key])

class Clock(Component):
    def __init__(self):
        super(Clock, self).__init__()

        self.add_output("ph0", 0)
        self.add_output("ph1", 1)

        self.state_ = 0

    def name(self):
        return "clock"

    def step(self):
        self.state_ = 1 - self.state_

    def update(self, input_values, input_values_old):
        self.output_values["ph0"] = self.state_
        self.output_values["ph1"] = 1 - self.state_

class Latch(Component):
    def __init__(self, value):
        super(Latch, self).__init__()

        self.add_input("in")
        self.add_input("clock")

        self.add_output("out", value)

    def name(self):
        return "latch"

    def update(self, input_values, input_values_old):
        if input_values["clock"] == 1 and \
            input_values_old["clock"] != input_values["clock"]:

            self.output_values["out"] = input_values["in"]

class Incrementer(Component):
    def __init__(self):
        super(Incrementer, self).__init__()

        self.add_input("in")
        self.add_output("out", 0)

    def name(self):
        return "incrementer"

    def update(self, input_values, input_values_old):
        self.output_values["out"] = input_values["in"] + 1

class Constant(Component):
    def __init__(self, value=0):
        super(Constant, self).__init__()

        self.add_output("out", value)

    def name(self):
        return "constant"

    def update(self, input_values, input_values_old):
        pass

class Multiplexer(Component):
    def __init__(self, num_inputs):
        super(Multiplexer, self).__init__()

        for i in xrange(num_inputs):
            self.add_input("input_{}".format(i))

        self.add_input("control")
        self.add_output("out", 0)

    def name(self):
        return "multiplexer"

    def update(self, input_values, input_values_old):
        self.output_values["out"] = input_values["input_{}".format(input_values["control"])]

class Fetcher(Component):
    def __init__(self, instructions):
        super(Fetcher, self).__init__()


        self.add_input("address")
        self.add_input("clock")
        self.add_output("instruction", instruction.NOOP())

        self.instructions = instructions

    def name(self):
        return "fetcher"

    def update(self, input_values, input_values_old):
        if input_values["clock"] == 1 and \
            input_values_old["clock"] != input_values["clock"]:

            self.output_values["instruction"] = self.instructions[input_values["address"]]

class RegisterFile:
    def __init__(self, num_registers):
        super(RegisterFile, self).__init__()
        self.name_ = "RegisterFile"

        self.registers = [0]*num_registers

        self.add_input("read_sel1")
        self.add_input("read_sel2")

        self.add_input("write_sel")
        self.add_input("write_enable")
        self.add_input("write_data")

        self.add_input("clock")

        self.add_output("read_data1")
        self.add_output("read_data2")

    def update(self, state):
        # define clock high to mean write
        if state["clock"] == 1:
            if state["write_enable"] == 1:
                # disallow writing to R0
                assert state["write_sel"] != 0, "Cannot change R0's value"

                self.registers[state["write_sel"]] = state["write_data"]

        # define clock low to mean read
        else:
            self.set_output("read_data1", self.registers[state["read_sel1"]])
            self.set_output("read_data2", self.registers[state["read_sel2"]])

# class Decoder:
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.name_ = "Decoder"
#
#         self.add_input("instruction")
#
#         '''
#         next stage needs to know:
#             ALU operation
#             if its a branch or r->r
#
#             ALU source - goes into a tmux to decide which to add ie r0 + r1 or r0 + imm
#         '''
#
#         self.add_output("read_sel1")
#         self.add_output("read_sel2")
#
#         self.add_output("ALU_op")
#
#         self.add_output("WB_enable")
#         self.add_output("WB_sel")
#
#     def clock(self):
#         instr = self.instruction.get_value()
#         print "Decoder recieved instruction {}".format(instr)
#
#         if  isinstance(instr, instruction.ADD) or \
#             isinstance(instr, instruction.SUB) or \
#             isinstance(instr, instruction.DIV) or \
#             isinstance(instr, instruction.MUL) or \
#             isinstance(instr, instruction.AND) or \
#             isinstance(instr, instruction.OR)  or \
#             isinstance(instr, instruction.XOR):
#
#             rd, rv0, rv1 = instr
#
#         if  isinstance(instr, instruction.ADDI) or \
#             isinstance(instr, instruction.SUBI) or \
#             isinstance(instr, instruction.DIVI) or \
#             isinstance(instr, instruction.MULI) or \
#             isinstance(instr, instruction.ANDI) or \
#             isinstance(instr, instruction.ORI)  or \
#             isinstance(instr, instruction.XORI):
#
#             rd, rv0, imm = instr
#
#         if  isinstance(instr, instruction.BEQ) or \
#             isinstance(instr, instruction.BNE):
#
#             ad, rv0, rv1 = instr
#
#         if  isinstance(instr, instruction.BGEZ) or \
#             isinstance(instr, instruction.BGTZ) or \
#             isinstance(instr, instruction.BLEZ) or \
#             isinstance(instr, instruction.BLTZ):
#
#             ad, rv0 = instr
#
#         if  isinstance(instr, instruction.J):
#
#             ad, = instr
#
#         if  isinstance(instr, instruction.JR):
#
#             rv, = instr
#
#         if  isinstance(instr, instruction.LOAD) or \
#             isinstance(instr, instruction.STOR):
#
#             rf, rv, imm = instr
#
#         if  isinstance(instr, instruction.NOOP):
#             pass


clock = Clock()
PC    = Latch(0)
incPC = Incrementer()

PCmux      = Multiplexer(2)
isbranch   = Constant(0) # 1
branchaddr = Constant(1337) # 1337

import programs.add_test
fetcher = Fetcher(programs.add_test.instructions)

cco = ComponentConnectionOrchestrator()
cco.add_component(clock)
cco.add_component(PC)
cco.add_component(incPC)
cco.add_component(PCmux)
cco.add_component(isbranch)
cco.add_component(branchaddr)
cco.add_component(fetcher)

cco.add_connection((clock.id,"ph1"),(PC.id,   "clock"))
cco.add_connection((PCmux.id,"out"),(PC.id,   "in"))

cco.add_connection((PC.id,   "out"),(incPC.id,"in"))

cco.add_connection((incPC.id,     "out"),(PCmux.id,"input_0"))
cco.add_connection((branchaddr.id,"out"),(PCmux.id,"input_1"))
cco.add_connection((isbranch.id,  "out"),(PCmux.id,"control"))

cco.add_connection((PC.id,   "out"),(fetcher.id,"address"))
cco.add_connection((clock.id,"ph0"),(fetcher.id,"clock"))

cco.propagate(clock.id)

def step():
    clock.step()
    cco.propagate(clock.id)
    clock.step()
    cco.propagate(clock.id)

    cco.dump_component_states()
