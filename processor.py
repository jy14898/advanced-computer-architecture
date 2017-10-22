import instruction
from memory import Memory
import functools


'''
a note someone made here:http://slideplayer.com/slide/9255919/
https://www.cise.ufl.edu/~mssz/CompOrg/Figure5.6-PipelineControlLines.gif
'''

'''
instead of having inputs and outputs marked as constant, i move that decision
to add_dependency and add a no_wait flag to say dont wait on this input

I could also solve the issue of setup by once all the inputs are connected,
the component does the first update but doesnt propagate its outputs. This will
basically init the output values

still doesnt fix it tho because we need an initial signal in the case of PC loop
'''

class Component(object):
    def __init__(self):
        self.inputs  = {}
        self.outputs = {}

        self.checked_self = False

    def name(self):
        return self.name_

    def add_dependency(self, source, source_outkey, inkey):
        assert inkey in self.inputs, "Input key doesnt exist"
        assert self.inputs[inkey]["source"] == (None,None), "Only 1 source allowed to resolve 1 input"

        isconstant = source.register(self, source_outkey, inkey)

        self.inputs[inkey]["source"] = (source, source_outkey)
        self.inputs[inkey]["updated"] = 2 if isconstant else 0

        # if all inputs are constant then so is the output...
        # also im not defining what happens when you connect two constant components together in a loop...
        if all(v["updated"] == 2 for v in self.inputs.values()):
            for outkey in self.outputs:
                self.outputs[outkey]["constant"] = True

            # Do the first pass as inputs are constant, then we never need to again
            self.propagate()

    def register(self, destination, outkey, destination_inkey):
        assert outkey in self.outputs, "Tried to register to outkey {} which doesn't exist".format(outkey)
        assert (destination, destination_inkey) not in self.outputs[outkey]["destinations"], "Tried to register {}'s inkey {} to outkey {} twice'".format(destination.name(), destination_inkey, outkey)

        self.outputs[outkey]["destinations"].append((destination, destination_inkey))

        return self.outputs[outkey]["constant"]

    def get_output(self, outkey):
        return self.outputs[outkey]["value"]

    def set_output(self, outkey, value):
        self.outputs[outkey]["value"] = value

    # Just to make sure I havent forgotten any input connections as they're important
    def is_setup(self):
        if self.checked_self == True:
            return True

        for inkey in self.inputs:
            if self.inputs[inkey]["source"] == (None, None):
                return False

        self.checked_self = True
        return True

    def update_input(self, source, source_outkey, inkey):
        assert self.is_setup()

        print "Updating input {} @ {} from {} @ {} with value {}".format(inkey, self.name(), source_outkey, source.name(), source.get_output(source_outkey))

        assert inkey in self.inputs, "Tried to update inkey {} but doesn't exist".format(inkey)
        assert self.inputs[inkey]["source"] == (source, source_outkey), "Wrong source and outkey"
        assert self.inputs[inkey]["updated"] != 1, "Input {} for {} was set twice".format(inkey, self.name())
        assert self.inputs[inkey]["updated"] != 2, "Attempted to change constant input"

        self.inputs[inkey]["updated"] = 1

        if all(v["updated"] == 1 or v["updated"] == 2 for v in self.inputs.values()):
            self.propagate()

    def propagate(self):
        assert self.is_setup()

        state = {}

        for inkey in self.inputs:
            state[inkey] = self.inputs[inkey]["source"][0].get_output(self.inputs[inkey]["source"][1])

        self.update(state)

        # reset non-constant inputs
        for inkey in self.inputs:
            if self.inputs[inkey]["updated"] == 1:
                self.inputs[inkey]["updated"] = 0

        # update non-constant outputs
        for outkey in self.outputs:
            if not self.outputs[outkey]["constant"]:
                for destination, destination_inkey in self.outputs[outkey]["destinations"]:
                    destination.update_input(self, outkey, destination_inkey)

    def update(self, state):
        raise NotImplementedError('Components must override update()!')

    def add_input(self, key):
        self.inputs[key] = {
            "updated": 0,
            "source": (None, None)
        }

    def add_output(self, key, constant=False):
        self.outputs[key] = {
            "value": None,
            "destinations": [],
            "constant": constant
        }

class Constant(Component):
    def __init__(self, value=0):
        super(Constant, self).__init__()

        self.add_output("out", constant=True)
        self.set_output("out", value)

class UserInput(Component):
    def __init__(self):
        super(UserInput, self).__init__()

        self.add_output("out")

    def write(self):
        self.value_ = raw_input("Value: ")
        self.propagate()

    def update(self, state):
        self.set_output("out", self.value_)

class Latch(Component):
    def __init__(self, value, name="latch"):
        super(Latch, self).__init__()

        self.add_input("in")
        self.add_input("clock")

        self.add_output("out")

        self.name_ = name

        self.set_output("out", value)

    def update(self, state):
        if   state["clock"] == 1:
            self.set_output("out", state["in"])
        elif state["clock"] == 0:
            # Keep the output value the same
            pass

class Clock(Component):
    def __init__(self, name="clock"):
        super(Clock, self).__init__()

        self.add_output("ph0")
        self.add_output("ph1")

        self.state_ = 0

        self.name_ = name

    def step(self):
        self.state_ = 1 - self.state_
        self.propagate()

    def update(self, state):
        self.set_output("ph0", self.state_)
        self.set_output("ph1", 1 - self.state_)


class Multiplexer(Component):
    def __init__(self, num_inputs, name="multiplexer"):
        super(Multiplexer, self).__init__()

        for i in xrange(num_inputs):
            self.add_input("input_{}".format(i))

        self.add_input("control")
        self.add_output("out")

        self.name_ = name

    def update(self, state):
        self.set_output("out", state["input_{}".format(state["control"])])

class Incrementer(Component):
    def __init__(self, name="Incrementer"):
        super(Incrementer, self).__init__()

        self.add_input("in")
        self.add_output("out")

        self.name_ = name

    def update(self, state):
        self.set_output("out", state["in"] + 1)


class Display(Component):
    def __init__(self, name="Display"):
        super(Display, self).__init__()

        self.add_input("in")
        self.name_ = name

    def update(self, state):
        print state["in"]

class Fetcher(Component):
    def __init__(self, instructions):
        super(Fetcher, self).__init__()
        self.name_ = "fetcher"

        self.add_input("address")
        self.add_input("clock")
        self.add_output("instruction")

        self.instructions = instructions
        self.set_output("instruction", instruction.NOOP())

    def update(self, state):
        if   state["clock"] == 1:
            self.set_output("instruction", self.instructions[state["address"]])
        elif state["clock"] == 0:
            # Keep the output value the same
            pass

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

class Decoder:
    def __init__(self):
        super(Decoder, self).__init__()
        self.name_ = "Decoder"

        self.add_input("instruction")

        '''
        next stage needs to know:
            ALU operation
            if its a branch or r->r

            ALU source - goes into a tmux to decide which to add ie r0 + r1 or r0 + imm
        '''

        self.add_output("read_sel1")
        self.add_output("read_sel2")

        self.add_output("ALU_op")

        self.add_output("WB_enable")
        self.add_output("WB_sel")

    def clock(self):
        instr = self.instruction.get_value()
        print "Decoder recieved instruction {}".format(instr)

        if  isinstance(instr, instruction.ADD) or \
            isinstance(instr, instruction.SUB) or \
            isinstance(instr, instruction.DIV) or \
            isinstance(instr, instruction.MUL) or \
            isinstance(instr, instruction.AND) or \
            isinstance(instr, instruction.OR)  or \
            isinstance(instr, instruction.XOR):

            rd, rv0, rv1 = instr

        if  isinstance(instr, instruction.ADDI) or \
            isinstance(instr, instruction.SUBI) or \
            isinstance(instr, instruction.DIVI) or \
            isinstance(instr, instruction.MULI) or \
            isinstance(instr, instruction.ANDI) or \
            isinstance(instr, instruction.ORI)  or \
            isinstance(instr, instruction.XORI):

            rd, rv0, imm = instr

        if  isinstance(instr, instruction.BEQ) or \
            isinstance(instr, instruction.BNE):

            ad, rv0, rv1 = instr

        if  isinstance(instr, instruction.BGEZ) or \
            isinstance(instr, instruction.BGTZ) or \
            isinstance(instr, instruction.BLEZ) or \
            isinstance(instr, instruction.BLTZ):

            ad, rv0 = instr

        if  isinstance(instr, instruction.J):

            ad, = instr

        if  isinstance(instr, instruction.JR):

            rv, = instr

        if  isinstance(instr, instruction.LOAD) or \
            isinstance(instr, instruction.STOR):

            rf, rv, imm = instr

        if  isinstance(instr, instruction.NOOP):
            pass

'''
somehow do a post-setup check to verify everything is connected
'''
def test(instructions):
    # Components
    clock             = Clock()
    PC                = Latch(value=0, name="PC")
    incPC             = Incrementer()
    PCmux             = Multiplexer(2, name="PC multiplexer")
    isbranch          = Constant(0) # 1
    branchaddr        = Constant(0) # 1337
    fetcher           = Fetcher(instructions)
    instruction_latch = Latch(value=instruction.NOOP(), name="instruction latch")

    # Connections
    PC.add_dependency(clock, "ph1", "clock")
    PC.add_dependency(PCmux, "out", "in")

    incPC.add_dependency(PC, "out", "in")

    PCmux.add_dependency(incPC,      "out", "input_0")
    PCmux.add_dependency(branchaddr, "out", "input_1")
    PCmux.add_dependency(isbranch,   "out", "control")

    fetcher.add_dependency(PC, "out", "address")
    fetcher.add_dependency(clock, "ph0", "clock")

    instruction_latch.add_dependency(fetcher, "instruction", "in")
    instruction_latch.add_dependency(clock, "ph1", "clock")

    # Need to first propagate signals through the system to init
    # Alternatively we force inputs to be set to changed?
    incPC.propagate()

    def step():
        while True:
            clock.step()
            yield

    return clock, PC, incPC, PCmux, step()
