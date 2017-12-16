import functools
import json
import traceback
import itertools
import copy

from collections import OrderedDict
'''
TODO
somehow enforce that channelends are always written to during update. maybe also enforce only one write? idk
sometimes it doens't make sense to write back to a channel end tho. maybe always set to None before update


Also, technically it'd be fine to set 'new' state for clock during update, but it only the previous state should be read.
then at clock its basically committed?

that or we have internal connections
for now we can recreate the signals needed by state 




TODO 2
 DONE

Decoder
RegisterFile (load registers current values)
Issuer       (Hazards, stall if new register values coming in, issue if good)


TODO 3
Add branch predictor with pre-decode of instruction during fetch
 this can just stall the fetcher atm when it finds an known PC changing instruction
 can also do jumps to fixed addresses here?
 others need at least 1 register load (and comparisons usually)

 listen on common data bus for writes to PC. could put the PC in the register file but i dont think its worth it
 only works for 1 branch atm


 maybe just have it decode type of inst, whether it may jump etc. then it listens on common jump bus
 decode stage can issue a jump
 BR unit can issue a jump
 https://www.google.co.uk/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&ved=0ahUKEwiJy8OsrY3YAhVVFMAKHWocDaoQjRwIBw&url=http%3A%2F%2Fslideplayer.com%2Fslide%2F8969149%2F&psig=AOvVaw2T5DT1JukfNErtC3Y3HVt5&ust=1513473079353861

'''

# TODO use these instead
ph1 = 1
ph2 = 0

class ChannelEnd(object):
    def __init__(self, component):
        self.component = component
        self.rx_value = None
        self.rx_subscribers = set()
        
        self.tx_setter = lambda value: None

    @property
    def rx(self):
        return self.rx_value

    def set_rx(self, value):
        diff = self.rx_value != value
        self.rx_value = value
        
        if diff:
            for cb in self.rx_subscribers:
                cb(self)

    def subscribe_to_rx_change(self, callback):
        self.rx_subscribers.add(callback)

    @property
    def tx(self):
        raise AttributeError("tx can't be read")

    @tx.setter
    def tx(self, value):
        self.tx_setter(value)

class Component(object):
    def __init__(self, channel_end_factory):
        self.channel_end_factory = channel_end_factory

        self.config = {
            "channel_keys": set()
        }

        self.state = {}

    def update_channels(self, clock_phase):
        pass

    def update_state(self, clock_phase):
        return False

    def add_channel(self, key):
        self.config["channel_keys"].add(key)

        # probably not the best way but its easy
        channel_end = self.channel_end_factory(self)
        setattr(self, key, channel_end)
        return channel_end

class ComponentUpdater(object):
    def __init__(self):
        self.dirty_components = set()
        self.clock_phase      = 0

    def channel_end_rx_changed(self, channel_end):
        self.dirty_components.add(channel_end.component)

    def channel_end_factory(self, component):
        channel_end = ChannelEnd(component)
        channel_end.subscribe_to_rx_change(self.channel_end_rx_changed)
        return channel_end

    def clock(self, components):
        # TODO: move clock outside of this into processor
        self.clock_phase = 1 - self.clock_phase

        for component in components.itervalues():
            stateful = component.update_state(self.clock_phase)
            if stateful:
                self.dirty_components.add(component)
                
    def propagate(self):
        while self.dirty_components:
            component = self.dirty_components.pop()
            component.update_channels(self.clock_phase)

class DictBroadcastBus(Component):
    def __init__(self, channel_end_factory):
        super(DictBroadcastBus, self).__init__(channel_end_factory)

        self.bus_channel_counter = 0

    def add_bus_channel(self):
        channel_end = self.add_channel("channel_{}".format(self.bus_channel_counter))
        self.bus_channel_counter += 1
        return channel_end

    def update_channels(self, clock_phase):
        channel_ends = list(getattr(self,"channel_{}".format(i)) for i in xrange(self.bus_channel_counter))

        dicts = (channel_end.rx for channel_end in channel_ends)
        out = dict(itertools.chain.from_iterable(dict_.iteritems() for dict_ in dicts if isinstance(dict_, dict)))

        for channel_end in channel_ends:
            channel_end.tx = out

class DirectionalBus(Component):
    def __init__(self, channel_end_factory):
        super(DirectionalBus, self).__init__(channel_end_factory)
        
        self.dirbus_channel_counter = 0

    def add_dirbus_channel(self):
        channel_end = self.add_channel("channel_{}".format(self.dirbus_channel_counter))
        self.dirbus_channel_counter += 1
        return channel_end

    def update_channels(self, clockphase):
        for i in xrange(1, self.dirbus_channel_counter):
            getattr(self, "channel_{}".format(i - 1)).tx = getattr(self, "channel_{}".format(i)).rx

class ChannelBuffer(Component):
    def __init__(self, channel_end_factory):
        super(ChannelBuffer, self).__init__(channel_end_factory)

        self.add_channel("A")
        self.add_channel("B")

        self.state["A"] = None
        self.state["B"] = None

    def update_channels(self, clock_phase):
        self.A.tx = self.state["B"]
        self.B.tx = self.state["A"]

    def update_state(self, clock_phase):
        if clock_phase == 1:
            self.state["A"] = self.A.rx
            self.state["B"] = self.B.rx
        
            print self.state

            return True
        else:
            return False

# NOTE technically it's InstructionFetchFromCache
class InstructionFetch(Component):
    def __init__(self, channel_end_factory, instructions):
        super(InstructionFetch, self).__init__(channel_end_factory)

        self.add_channel("instruction")
        self.add_channel("stall")

        self.config["instructions"] = instructions

        self.state["instruction"] = None 
        #{
        #    "instruction": ("NOOP",0,0,0,0),
        #    "pc": 0,
        #    "pc_next": 1
        #}

        self.state["pc"] = 0

    def update_channels(self, clock_phase):
        self.instruction.tx = self.state["instruction"]

    def update_state(self, clock_phase):
        if not self.stall.rx and clock_phase == 0: # ph2
            self.state["instruction"] = {
                "instruction": self.config["instructions"][self.state["pc"]],
                "pc": self.state["pc"],
                "pc_next": self.state["pc"] + 1,
            }

            self.state["pc"] = self.state["pc"] + 1

            return True

        return False

class SmartPC(Component):
    def __init__(self, channel_end_factory, instructions):
        super(SmartPC, self).__init__(channel_end_factory)

        self.add_channel("pc")     # this output magically updates by predicting
        self.add_channel("set_pc") # this should only be set on a mispredict 
        self.add_channel("stall")  # this should be set when a buffer is full somewhere

        self.state["pc"] = 0

        # TODO have a history buffer

    def update_channels(self, clock_phase):
        self.instruction_fetch.tx = self.state["pc"]

    def update_state(self, clock_phase):
        if clock_phase == 1: # ph1
            # recieve PC from BUS or something where the predictor was wrong
        else: # ph2
            # predict next PC

        return False


class InstructionOrderBuffer(Component):
    def __init__(self, channel_end_factory, instructions):
        super(InstructionOrderBuffer, self).__init__(channel_end_factory)

        # each stage pushes through the state of each record
        # it either fails and we unwind
        # or succeeds and we push the changes through to all components that care (Predictor, Issue, RF, memory?)


class Decoder(Component):
    def __init__(self, channel_end_factory):
        super(Decoder, self).__init__(channel_end_factory)

        self.add_channel("instruction")
        self.add_channel("decoded_instruction")

    def update_channels(self, clock_phase):
        instruction = self.instruction.rx["instruction"]
        pc          = self.instruction.rx["pc"]
        pc_next     = self.instruction.rx["pc_next"]

        opcode, reg_read_sel1, reg_read_sel2, reg_write_sel, immediate = instruction
        
        is_valid = False
        # TODO decide whether or not to keep DIVI/MULI
        if opcode in ["ADD","SUB","DIV","MUL","AND","OR" ,"XOR", "ADDI","SUBI","DIVI","MULI","ANDI","ORI","XORI"]:
            is_immediate = opcode[-1:] == 'I'

            EU = "ALU"
            EU_control = {
                "op": opcode[:-1] if is_immediate else opcode
            }

            # for simplicity b is always immediate if it is there. this has implicitions for div. 
            EU_data = {
                "a": (True, reg_read_sel1, None),
                "b": (True, reg_read_sel2, None) if not is_immediate else (False, None, immediate) 
            }

            EU_write_reg = reg_write_sel
            is_valid = True

        elif opcode in ["BEQ","BNE", "GEZ", "GTZ", "LEZ", "LTZ"]:
            EU = "BR"
            EU_control = {
                "op": opcode[1:] if opcode in ["BEQ","BNE"] else opcode,
            }

            EU_data = {
                "a": (True, reg_read_sel1, None),
                "pc_true": (False, None, immediate),
                "pc_false": (False, None, pc_next),
            }

            if opcode in ["BEQ", "BNE"]:
                EU_data["b"] = (True, reg_read_sel2, None)
            
            EU_write_reg = "pc"
            is_valid = True

        elif opcode in ["J","JR"]:
            EU = "BR"

            EU_control = {"op":"TRUE"}

            EU_data = {
                "pc_true": (False, None, immediate) if opcode == "J" \ 
                      else (True, reg_read_sel1, None),
            }

            EU_write_reg = "pc"
            is_valid = True
        
        if is_valid:
            self.decoded_instruction.tx = {
                "instruction": instruction,
                "pc": pc,
                "EU": EU,
                "EU_control": EU_control,
                "EU_data": EU_data,
                "EU_write_reg": EU_write_reg,
            }
        else:
            self.decoded_instruction.tx = None


class RegisterFile(Component):
    # Listen on CDB and update entries
    # have a 'instruction passthrough' which updates instruction register values
    
    # write registers on 0, read on 1
    def __init__(self, channel_end_factory, num_registers):
        super(RegisterFile, self).__init__(channel_end_factory)

        # connect to CDB
        self.add_channel("update")

        # connect to decoder
        self.add_channel("decoded_instruction")

        # connect to issuer
        self.add_channel("decoded_instruction_filled")
    
        self.state["registers"] = list(0 for i in xrange(num_registers))


    def update_channels(self, clock_phase):
        if self.decoded_instruction.rx is not None:
            decoded_instruction = self.decoded_instruction.rx.copy()
            decoded_instruction["EU_data"] = dict((k, (True, r, self.state["registers"][r]) if is_reg else (False, None, v)) \
                    for k, (is_reg, r, v) in decoded_instruction["EU_data"].iteritems())
            
            self.decoded_instruction_filled.tx = decoded_instruction
        else:
            self.decoded_instruction_filled.tx = None

        self.update.tx = {}

    def update_state(self, clock_phase):
        # onl write on PH1, not PH2
        if clock_phase == 1 and self.update.rx is not None:
            register_updates = self.update.rx
            
            # probably gonna be a table of indicies to values
            for r, v in register_updates.iteritems():
                if r not in self.state["registers"]:
                    continue

                if r == 0:
                    raise "DONT WRITE TO R0!"
                
                self.state["registers"][r] = v

            return True
        else:
            return False
        

class SimpleInstructionIssuer(Component):
    def __init__(self, channel_end_factory):
        super(SimpleInstructionIssuer, self).__init__(channel_end_factory)

        self.add_channel("decoded_instruction")
        self.add_channel("stall")
        self.add_channel("alu")
        self.add_channel("bru")
        self.add_channel("register_update")

        self.state["EU_states"] = {
            "alu": (False, None), # (inuse, destreg)
        }

    def assign_instruction_eu(self):
        # NOTE this causes stalls until the branch instruction is completed
        if self.state["bru"] == (True, "pc"):
            return (True, None)

        d_instruction = self.decoded_instruction.rx
        if d_instruction is None:
            return (False, None)

        stall = False
        EU    = None

        di_eu_data = d_instruction["EU_data"]

        # read after write
        dep_regs   = (r for (is_reg, r, _) in di_eu_data.itervalues() if is_reg)

        # write after write
        dep_regs   = frozenset(itertools.chain(dep_regs, (d_instruction["EU_write_reg"],)))

        busy_regs  = frozenset(reg for (is_busy, reg) in self.state["EU_states"].itervalues() if is_busy)
        
        if dep_regs.isdisjoint(busy_regs):
            # TODO make list of available ALUs and search through those if the inst is ALU
            if d_instruction["EU"] == "ALU" and self.state["EU_states"]["alu"][0] == False:
                EU = "alu"
            elif d_instruction["EU"] == "BRU" and self.state["EU_states"]["bru"][0] == False:
                EU = "bru"
            else:
                stall = True
        else:
            stall = True

        return (stall, EU)

    def update_channels(self, clock_phase):
        # TODO assign all EU's None
        #      actually that'll cause a bug. we need to store prev value in channelend at start of update?
        self.alu.tx = None
        self.bru.tx = None
        self.stall.tx = False

        if self.decoded_instruction.rx is not None:
            stall, EU = self.assign_instruction_eu()
            
            self.stall.tx = stall
            if not stall:
                getattr(self, EU).tx = {
                    "control": self.decoded_instruction.rx["EU_control"],
                    "data": dict((k,v) for k,(_,_,v) in self.decoded_instruction.rx["EU_data"].iteritems()),
                    "write_reg": self.decoded_instruction.rx["EU_write_reg"]
                }

        self.register_update.tx = {}

    # we can use the same function in update_state because all update_states are essentially called in parallel, so the inputs will be the same
    def update_state(self, clock_phase):
        if clock_phase == 0: # ph2
            stall, EU = self.assign_instruction_eu()

            if not stall and EU is not None:
                self.state["EU_states"][EU] = (True, self.decoded_instruction.rx["EU_write_reg"])
        else: # ph1
            register_updates = self.register_update.rx 

            if register_updates is not None:
                for r, v in register_updates.iteritems():
                    eus = list(eu for eu, (is_busy, dest_reg) in self.state["EU_states"].iteritems() if is_busy and r == dest_reg)
                    if len(eus) > 1:
                        raise "Only one EU should be writing to register {}".format(r)
                    elif len(eus) == 1:
                        self.state["EU_states"][eus[0]] = (False, None)

class BRU(Component):
    def __init__(self, channel_end_factory):
        super(BRU, self).__init__(channel_end_factory)

        self.add_channel("rs")
        self.add_channel("result")

        self.state["current_operation"] = None

    ops = {
        "EQ": lambda ins: True if ins["a"] == ins["b"] else False,
        "NE": lambda ins: True if ins["a"] != ins["b"] else False,
        "GEZ": lambda ins: True if ins["a"] >= 0 else False,
        "GTZ": lambda ins: True if ins["a"] >  0 else False,
        "LEZ": lambda ins: True if ins["a"] <= 0 else False,
        "LTZ": lambda ins: True if ins["a"] <  0 else False,
        "TRUE": lambda _: True
    }

    def get_current_op_result(self):
        co = self.state["current_operation"]
        return co["pc_true"] if BRU.ops[co["op"]](co) else co["pc_false"]

    def update_channels(self, clock_phase):
        if clock_phase == 0 and self.state["current_operation"] is not None:
            self.result.tx = {"pc": self.get_current_op_result()}
        else:
            self.result.tx = {}

    self.rs.tx = {}     
    
    def update_state(self, clock_phase):
        if clock_phase == 1: #ph1
            self.state["current_operation"] = None

            issue = self.rs.rx
            if issue is not None:
                # TODO: throw warning if currently executing?

                print "BR recieved {}".format(self.rs.rx)
                co = {
                    "op": issue["control"]["op"],
                    "a": issue["data"]["a"],
                    "pc_true": issue["data"]["pc_true"],
                    "pc_false": issue["data"]["pc_false"],
                }

                if "b" in issue["data"]:
                    co["b"] = issue["data"]["b"]

                
                self.state["current_operation"] = co

class ALU(Component):
    operations = {
        "ADD":  (1,lambda ins: ins["a"] + ins["b"]),
        "SUB":  (1,lambda ins: ins["a"] - ins["b"]),
        "DIV":  (5,lambda ins: 0 if ins["b"] == 0 else ins["a"] / ins["b"]),
        "MUL":  (5,lambda ins: ins["a"] * ins["b"]),
        "AND":  (1,lambda ins: ins["a"] & ins["b"]),
        "OR":   (1,lambda ins: ins["a"] | ins["b"]),
        "XOR":  (1,lambda ins: ins["a"] ^ ins["b"]),
    }

    def __init__(self, channel_end_factory):
        super(ALU, self).__init__(channel_end_factory)

        self.add_channel("rs")
        self.add_channel("result")

        self.state["cycles_left"] = 0
        self.state["current_operation"] = None

    def get_current_op_result(self):
        co = self.state["current_operation"]
        return ALU.operations[co["op"]][1](co)

    # output answer on clockphase ph2 (0)

    def update_channels(self, clock_phase):
        if self.state["cycles_left"] == 0 and self.state["current_operation"] is not None:
            print "R{}: {}".format(self.state["current_operation"]["dest"], self.get_current_op_result())
            self.result.tx = {
                self.state["current_operation"]["dest"]: self.get_current_op_result()
            }
        else:
            self.result.tx = {} # or None?


        self.rs.tx = {}

    def update_state(self,clock_phase):
        if clock_phase == 1: #ph1
            if self.state["cycles_left"] == 0:
                self.state["current_operation"] = None

            issue = self.rs.rx
            if issue is not None:
                # TODO: throw warning if currently executing?

                print "ALU recieved {}".format(self.rs.rx)
                self.state["cycles_left"] = ALU.operations[issue["control"]["op"]][0]
                self.state["current_operation"] = {
                    "op": issue["control"]["op"],
                    "a": issue["data"]["a"],
                    "b": issue["data"]["b"],
                    "dest": issue["write_reg"],
                }
            
        if clock_phase == 0: # ph2
            if self.state["cycles_left"] > 0:
                self.state["cycles_left"] -= 1

        return True

#    def update_state(self, clock_phase):
   #     if clock_phase == 1:
   #         if self.state["cycles_left"] > 0:
   #             self.state["cycles_left"] -= 1

   #         if self.rs.rx["control"] != None:
   #             instruction = self.rs.rx["instruction"]
   #             
   #             # need to broadcast the operation result with the written register on finish
   #             self.state["current_instruction"] = instruction
   #             self.state["cycles_left"] = ALU.operations[ALU.get_op(instruction[0])][0]
   #         
            # nope
    #        if self.state["current_instruction"] != None and self.state["cycles_left"] == 0:
    #            self.state["result"] = ALU.operations[ALU.get_op(instruction[0])][1](


class Processor(object):
    def __init__(self, instructions, data):
        self.cu = ComponentUpdater()

        self.setup_components(instructions, data)
    
    def step(self):
        self.cu.clock(self.components)
        # maybe subscribe every time its dirtied, it's a set so its fast
        self.cu.propagate()

    def setup_components(self, instructions, data):
        c = self.cu.channel_end_factory

        # directly connect 
        # could technically connect over another medium ie socket for true multprocess
        def connect(a, b):
            a.tx_setter = b.set_rx
            b.tx_setter = a.set_rx

        instruction_fetch    = InstructionFetch(c, instructions)
        instruction_fetch_cb = ChannelBuffer(c)
        connect(instruction_fetch.instruction, instruction_fetch_cb.A)

        decoder = Decoder(c)
        connect(instruction_fetch_cb.B,decoder.instruction)
   
        register_file = RegisterFile(c, 16)
        connect(decoder.decoded_instruction, register_file.decoded_instruction)

        sii = SimpleInstructionIssuer(c)
        connect(register_file.decoded_instruction_filled, sii.decoded_instruction)

        # this is also called common data bus
        write_back_bus = DictBroadcastBus(c)
        connect(write_back_bus.add_bus_channel(), sii.register_update)
        connect(write_back_bus.add_bus_channel(), register_file.update)

        alu = ALU(c)
        connect(write_back_bus.add_bus_channel(), alu.result)
        connect(sii.alu, alu.rs)

        staller = DirectionalBus(c)
        connect(staller.add_dirbus_channel(), instruction_fetch.stall) 
        connect(staller.add_dirbus_channel(), sii.stall) 
        # need a register alias table
        # points to the instruction which holds what the register will eventually be
        # if no instruction, 'points' to the actual register

        # decoded instructions go into RS
        # when issued get their operands resolved
        # execute
        # changes to registers go into ROB
        # changes to memory are done within the EU, ie no need for a ROB

        # nice and hacky
        self.components = dict(i for i in locals().iteritems() if isinstance(i[1],Component))

