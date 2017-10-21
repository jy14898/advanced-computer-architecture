import instruction
from memory import Memory
import functools


'''
a note someone made here:http://slideplayer.com/slide/9255919/

you need to carry the write register location through the pipeline as its only relavent
when it leaves the pipeline (well leave as in go back to decode stage)

'''
class Component(object):
    def __init__(self):
        self.inputs  = {}
        self.outputs = {}

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

    def get_output(self, outkey):
        return self.outputs[outkey]["value"]

    def set_output(self, outkey, value):
        self.outputs[outkey]["value"] = value

    def update_input(self, source, source_outkey, inkey):
        assert inkey in self.inputs, "Tried to update inkey {} but doesn't exist".format(inkey)
        assert self.inputs[inkey]["source"] == (source, source_outkey), "Wrong source and outkey"
        assert self.inputs[inkey]["updated"] != 1, "Input was set twice"
        assert self.inputs[inkey]["updated"] != 2, "Attempted to change constant input"

        self.inputs[inkey]["updated"] = 1

        if all(v["updated"] == 1 or v["updated"] == 2 for v in self.inputs.values()):
            self.propagate()

    def propagate(self):
        assert all(v["updated"] == 1 or v["updated"] == 2 for v in self.inputs.values())

        # get all outputs of inputs and whack them into dict
        state = {}

        for inkey in self.inputs:
            state[inkey] = self.inputs[inkey]["source"][0].get_output(self.inputs[inkey]["source"][1])

        self.update(state)

        # update non-constant outputs
        for outkey in self.outputs:
            if not self.outputs[outkey]["constant"]:
                for destination, destination_inkey in self.outputs[outkey]["destinations"]:
                    destination.update_input(self, outkey, destination_inkey)

        # reset non-constant inputs
        for inkey in self.inputs:
            if self.inputs[inkey]["updated"] == 1:
                self.inputs[inkey]["updated"] = 0

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
    def __init__(self):
        super(Constant, self).__init__()

        self.add_output("out_value", constant=True)
        self.outputs["out_value"]["value"] = 1337

class UserInput(Component):
    def __init__(self):
        super(UserInput, self).__init__()

        self.add_output("out_value")

    def write(self):
        num = int(raw_input("Gimme yo digits: "))
        self.value_ = num
        self.propagate()

    def update(self, state):
        self.set_output("out_value", self.value_)

class Latch(Component):
    def __init__(self):
        super(Latch, self).__init__()

        self.add_input("new_value")
        self.add_input("clock")

        self.add_output("out_value")

    name_ = "latch"

    def update(self, state):
        if   state["clock"] == 1:
            self.outputs["out_value"]["value"] = state["new_value"]
        elif state["clock"] == 0:
            # Keep the output value the same
            pass

class Clock(Component):
    name_ = "clock"

    def __init__(self):
        super(Latch, self).__init__()

        self.add_output("ph0")
        self.add_output("ph1")

    def step(self):
        self.value_ = 1 - self.value_
        self.propagate()

    def update(self, state):
        self.set_output("ph0", self.value_)
        self.set_output("ph1", self.1 - self.value_)

class Clocker:
    class Phase:
        def __init__(self):
            self.clks  = []
            self.props = []

            def stepper():
                while True:
                    for prop in self.props:
                        prop()
                    yield
                    for clk in self.clks:
                        clk()
                    yield

            self.stepper = stepper()

        def add_clk(self, clk):
            self.clks.append(clk)

        def add_prop(self, prop):
            self.props.append(prop)

    def __init__(self, num_phases):
        self.num_phases  = num_phases
        self.phases      = []
        self.cycle_count = 0

        for i in xrange(num_phases):
            self.phases.append(Clocker.Phase())

        def stepper():
            while True:
                for idx, phase in enumerate(self.phases):
                    print "Phase {} propagate".format(idx)
                    phase.stepper.next()
                    yield

                    print "Phase {} clk".format(idx)
                    phase.stepper.next()
                    yield

                self.cycle_count += 1

        self.stepper = stepper()

'''
Change to new system:
    All components must now have well defined values every clock - every phase tho?
    This means a lot of "0"s are gonna be thrown around but it makes
        things more safe
'''

class Latch:
    next_value = None

    def __init__(self, value, name="latch"):
        self.current_value = value

        self.name = name

    def set_value(self, value):
        assert value != None, 'Setting {} to None disallowed'.format(self.name)
        assert self.next_value == None, '{} set twice'.format(self.name)

        print "set {} {}".format(self.name, value)

        self.next_value = value

    def switch(self):
        assert self.next_value != None, '{} didn\'t recieve a value to switch to'.format(self.name)

        print "switch {} {} -> {}".format(self.name, self.current_value, self.next_value)

        self.current_value = self.next_value
        self.next_value    = None

    def get_value(self):
        return self.current_value

'''
Not sure how well this will work if we multiplex over different phases
'''
class Multiplexer:
    def __init__(self, num_values, output, name="multiplexer"):
        self.output = output
        self.values = [None]*num_values
        self.control = None

        self.name = name

    def set_value(self, index, value):
        assert value != None, 'Setting {} value to None disallowed'.format(self.name)
        assert self.values[index] == None, '{} value set twice'.format(self.name)

        print "set {}[{}] = {}".format(self.name, index, value)

        self.values[index] = value
        self.check_and_send()

    def set_control(self, index):
        assert index != None, 'Setting {} index to None disallowed'.format(self.name)
        assert self.control == None, '{} control set twice'.format(self.name)

        print "set {} control = {}".format(self.name, index)

        self.control = index
        self.check_and_send()

    def check_and_send(self):
        if all(a is not None for a in self.values) and self.control is not None:
            self.output(self.values[self.control])
            self.control = None
            self.values  = list(None for _ in self.values)


'''
Change everything?

Basically, instead of having a propagate phase, we have a dependency system
    where it will propagate once all dependencies are clocked?

    so for eg you tell all your latches to tell you when they've got a new value
    and then once you've got all your new values, you make your changes

    any changes after that are considered a new 'lot' of values?
'''

def test(instructions):
    clocker = Clocker(num_phases=2)

    fetched_instruction = Latch(value=instruction.NOOP(), name="fetched instruction latch")


    fetcher = Fetcher(instructions, fetched_instruction)
    decoder = Decoder(fetched_instruction, None, None, None)

    '''
    split PC bit into phase 0, IF into phase 1?
    pc will have incremented by 1 tho, by the IF part

    these really need to be checked once things getgoing
    '''
    clocker.phases[1].add_prop(fetcher.clock)
    clocker.phases[1].add_prop(decoder.clock)

    '''
    Doesn't really matter what phase things are on at the moment, but it will later
    '''
    # as the later stages arent set up, always select PC + 1
    clocker.phases[0].add_prop(functools.partial(fetcher.next_PC_mux.set_control, 0))
    clocker.phases[0].add_prop(functools.partial(fetcher.next_PC_mux.set_value, 1, 1337))

    fetcher.PC.set_value(0)
    clocker.phases[0].add_clk(fetcher.PC.switch)

    '''
    I feel like this should be on phase 1 clk?
    '''
    # fetched_instruction.set_value(instruction.NOOP())
    clocker.phases[1].add_clk(fetched_instruction.switch)



    return clocker.stepper

#class Executer:
#    def __init__(self,

'''
decode instruction to generate control signals for each following stage
    wb: load or store control signal
      : whether or not we're writing to register at all

    mem: whether

pass through branch address
pass through the register values for operands
pass through resulting register location

pass through PC? i dont think i need to tho


write back to register file on the latch stage i think? im not really sure
that or make a latch which the register file sometimes reads in?

'''
class Decoder:
    class RegisterFile:
        def __init__(self, num_registers):
            self.registers = [0]*num_registers

        def __getitem__(self, key):
            assert isinstance(key,int)

            return self.registers[key]

        def __setitem__(self, key, value):
            assert isinstance(key,int)
            assert key != 0, 'R0 cannot be changed'

            self.registers[key] = value

    # eventually add latches to set registers?
    def __init__(self, instruction, exec_ctrl, mem_ctrl, wb_ctrl):
        self.instruction = instruction
        self.exec_ctrl = exec_ctrl
        self.mem_ctrl  = mem_ctrl
        self.wb_ctrl   = wb_ctrl

        self.registerFile = Decoder.RegisterFile(32)

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
https://www.cise.ufl.edu/~mssz/CompOrg/Figure5.6-PipelineControlLines.gif
basically mux between PC + 1 or some computed PC

need to mux because conditions for some jumps are decided at execute stage
otherwise stop what doing and redo?
'''
class Fetcher:
    def __init__(self, instructions, instruction):
        self.instructions = instructions
        self.instruction  = instruction

        self.PC = Latch(0, "PC register")
        self.next_PC_mux = Multiplexer(2, self.PC.set_value, "next PC mux")

    def clock(self):
        print "Fetching instruction from address {}".format(self.PC.get_value())

        instruction = self.instructions[self.PC.get_value()]

        print "Fetched instruction {}".format(instruction)

        self.instruction.set_value(instruction)

        self.next_PC_mux.set_value(0, self.PC.get_value() + 1)
