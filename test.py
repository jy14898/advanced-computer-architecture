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

class Latch2(Component):
    def __init__(self):
        super(Latch2, self).__init__()

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
