import functools
import json
import traceback
import itertools
import copy

from contextlib import contextmanager

from collections import namedtuple
'''
TODO
    make it superscalar. need to make everything a list. start off with RoB providing up to n RoB ids.
    idk if im allowed to predict each individually i guess so?

    eh actually lets make PC predictor better?

Plan:
  x add a halt op
        sets a flag in RoB which stops fetching new things
        eventually it'll be just the HALT instruction left, and nothing will update

  x print out commits/clock
  x finish branch unit
  h add load/store
        Modern machines use Load/store queue. similar to RoB, when decoded add entry, update as we go along, apply once instruction committed in RoB
        I think data bypass allowed between the queues? ie if you load from an address in the store queue, it's faster
        for now just do loads and stores in order?
        have a queue that the LSU loads its things into. Can immediately return if this queue is not full.
        then later on resolve the loads?
        
        we can go even simpler
        1 load/store unit. only allow 1 load/store instruction in RS at any time. if another tries to come in, stall

  x make branch predictor smarter
    multiple fetch/issue

    Make JR fire off at issue stage?
    maybe do exceptions
    write some programs - got a few now. maybe add a sort and something else?

    GUI? at least output instructions per cycle

      ok. don't predict exceptions

      
thought experiment we need to think about:
    RS gets full. stalls decode. decode had a jump but it doesnt send because stall (or does it?). the jump may clear RS and not require stall
'''

class ChannelEnd(object):
    def __init__(self, component):
        self.component = component
        self.rx_value = None
        self.tx_value = None

        self.pair = None

    @property
    def rx(self):
        return self.rx_value

    def set_rx(self, value):
        diff = self.rx_value != value
        
        # TODO optimise this
        if diff:
            self.rx_value = copy.deepcopy(value)
        
        return diff

    @property
    def tx(self):
        raise AttributeError("tx can't be read")

    @tx.setter
    def tx(self, value):
        self.tx_value = value

    def send(self):
        v = self.tx_value
        self.tx_value = None

        return self.pair.set_rx(v)


class Component(object):
    def __init__(self):
        self.config = {
            "channel_keys": set()
        }

        self._state      = {}
        self._state_next = {}

        self.print_list = []

    @contextmanager
    def Printer(self):
        # dont really need this whole thing anymore
        # other than for resetting i guess

        self.print_list = []

        def prints(s):
            self.print_list.append(s)

        yield prints


    @contextmanager
    def State(self):
        state = copy.deepcopy(self._state)
        yield state
        self._state_next = state
    
    def has_new_state(self):
        return self._state != self._state_next

    def commit_state(self):
        self._state = self._state_next

    def update(self):
        pass

    # NOTE DEPRECIATED
    def update_state(self):
        return self.commit_state()

    def add_channel(self, key):
        self.config["channel_keys"].add(key)

        channel_end = ChannelEnd(self)
        setattr(self, key, channel_end)
        return channel_end

class ComponentUpdater(object):
    def clock(self, components):
        dirty_components = set(component for component in components.itervalues() if component.has_new_state())
        
        for component in dirty_components:
            component.commit_state()

        return dirty_components
                
    def propagate(self, dirty_components):
        updated_components = set()

        dirty_components = copy.copy(dirty_components)
        while dirty_components:
            component = dirty_components.pop()
            
            updated_components.add(component)
            
            component.update()

            channels = (getattr(component, key) for key in component.config["channel_keys"])
            
            affected_components = (channel.pair.component for channel in channels if channel.pair is not None and channel.send())

            dirty_components.update(affected_components)

        return updated_components

class DictBroadcastBus(Component):
    def __init__(self):
        super(DictBroadcastBus, self).__init__()

        self.bus_channel_counter = 0

    def add_bus_channel(self):
        channel_end = self.add_channel("channel_{}".format(self.bus_channel_counter))
        self.bus_channel_counter += 1
        return channel_end

    def update(self):
        channel_ends = list(getattr(self,"channel_{}".format(i)) for i in xrange(self.bus_channel_counter))

        dicts = (channel_end.rx for channel_end in channel_ends if isinstance(channel_end.rx, dict))
        out = dict(itertools.chain.from_iterable(dict_.iteritems() for dict_ in dicts))

        for channel_end in channel_ends:
            channel_end.tx = out

class DirectionalBus(Component):
    def __init__(self):
        super(DirectionalBus, self).__init__()
        
        self.dirbus_channel_counter = 0

    def add_dirbus_channel(self):
        channel_end = self.add_channel("channel_{}".format(self.dirbus_channel_counter))
        self.dirbus_channel_counter += 1
        return channel_end

    def update(self):
        for i in xrange(1, self.dirbus_channel_counter):
            getattr(self, "channel_{}".format(i - 1)).tx = getattr(self, "channel_{}".format(i)).rx

class ChannelBuffer(Component):
    def __init__(self):
        super(ChannelBuffer, self).__init__()

        self.add_channel("A")
        self.add_channel("B")

        self._state["A"] = None
        self._state["B"] = None

    def update(self):
        with self.State() as state:
            self.A.tx, self.B.tx = state["B"], state["A"]
            state["A"], state["B"] = self.B.rx, self.A.rx
'''
random joe thought intergection.

for the roll-backable efficient dict
when you access an element, return an object which is a proxy of that element
have proxy types for dict, list etc. then we can record changes that way
'''

class PCPredictor(Component):
    def __init__(self, bht_size):
        super(PCPredictor, self).__init__()

        self.add_channel("pc")    # this output magically updates by predicting

        self.add_channel("rob")   # this should only be set on a mispredict 
        self.add_channel("update_predictor") # set this before rob

        self.add_channel("stall") # this should be set when a buffer is full somewhere

        self._state["pc"] = 0

        self._state["bht"] = list((0, 0, 0) for _ in xrange(bht_size))
        #(pc, counter, target)

    def update(self):
        with self.State() as state, self.Printer() as prints:
            prints("set pc.tx to state[\"pc\"]: {}".format(state["pc"]))
            self.pc.tx = state["pc"]

            if self.update_predictor.rx is not None:
                for pc, target in self.update_predictor.rx:
                    index = pc % len(state["bht"])
                    entry = state["bht"][index]
                    if entry[0] == pc:
                        if pc + 1 != target:
                            state["bht"][index] = (pc, min(max(entry[1] + 1,0),3), target)  
                        else:
                            state["bht"][index] = (pc, min(max(entry[1] - 1,0),3), entry[2])  
                    else:
                        if pc + 1 != target:
                            state["bht"][index] = (pc, 2, target)
                        else:
                            # use the old target address.
                            state["bht"][index] = (pc, 1, entry[2]) 


            # if there's an update to the PC, take it
            if self.rob.rx is not None:
                prints("ROB said to change next PC to {}".format(self.rob.rx))
                state["pc"] = self.rob.rx

            # predict next
            else: 
                if not self.stall.rx:
                    index = state["pc"] % len(state["bht"])
                    entry = state["bht"][index]
                    if entry[0] == state["pc"] and entry[1] > 1:
                        state["pc"] = entry[2]
                    else:
                        state["pc"] = state["pc"] + 1

                    prints("predicted {}".format(state["pc"]))
                else:
                    prints("stalled, so no change in pc")

class InstructionFetch(Component):
    def __init__(self, instructions):
        super(InstructionFetch, self).__init__()

        self.add_channel("pc")
        self.add_channel("stall")
        self.add_channel("rob")

        self.add_channel("rob_cancel")

        self.add_channel("instruction")

        self.config["instructions"] = instructions
        self._state = {
            "pc": None,
            "instruction": None,
            "rob_id": None,
        }

        self.default_instruction = ("NOOP",0,0,0,0)
        
    def update(self):
        with self.State() as state, self.Printer() as prints:
            if state["instruction"] is not None:
                self.instruction.tx = {
                    "instruction": state["instruction"],
                    "pc": state["pc"],
                    "pc_add_one": state["pc"] + 1, # technically this would be calculated here, so we can reuse it
                    "rob_id": state["rob_id"],
                }

            # have this after the output so that it's one behind
            if self.pc.rx is not None and not self.stall.rx and self.rob.rx is not None:
                try:
                    instruction = self.config["instructions"][self.pc.rx]
                except IndexError:
                    instruction = self.default_instruction

                state.update({
                    "pc": self.pc.rx,
                    "instruction": instruction,
                    "rob_id": self.rob.rx,
                })

                prints("fetch pc={} complete: {}".format(state["pc"], state["instruction"]))

                self.rob.tx = {
                    "pc": state["pc"],
                    "_instruction": state["instruction"],
                }
            else:
                pass
                #state.update({
                #    "pc": None,
                #    "instruction": None,
                #    "rob_id": None,
                #})

            if self.stall.rx or self.rob.rx is None:
                self.stall.tx = True

            cancel = self.rob_cancel.rx
            if cancel is not None and "keys" in cancel and state["rob_id"] in cancel["keys"]: 
                state["instruction"] = None
                prints("Fetched instruction was canceled by RoB. Sending bubble")


class ReorderBuffer(Component):
    record_fields = ["instruction_pc", "dest", "dest_value", "pc_value", "exception_id_value", "exception_pc_value", "_instruction", "type"]

    def __init__(self, max_size, num_commit):
        super(ReorderBuffer, self).__init__()
        
        
        self.config["max_size"] = max_size # 20
        self.config["num_commit"] = num_commit # 20
        self._state["buffer"]   = list(dict((f, None) for f in ReorderBuffer.record_fields) for _ in xrange(self.config["max_size"]))
        self._state["start"]    = 0
        self._state["size"]     = 0
        self._state["halt"]     = False

        self._state["_num_instructions"] = 0
    
        self.add_channel("pc_predictor")
        self.add_channel("update_predictor")
        self.add_channel("fetch")
        self.add_channel("decode")
        self.add_channel("rat")

        self.add_channel("rf_write")

        self.add_channel("update_bus")

        self.add_channel("cancel")


    # https://classes.cs.uchicago.edu/archive/2016/fall/22200-1/lecture_notes/li-cmsc22200-aut16-lecture9.pdf
    # https://courses.cs.washington.edu/courses/cse471/07sp/lectures/Lecture4.pdf
    # slide 4
    
    def get_next_free(self, state):
        return (state["start"] + state["size"]) % self.config["max_size"]
    
    def get_naming(self, state):
        # the oldest reference to the dest is the entry that is assigned to it
        # TODO eventually i need to 'sub-key' the ROB indicies, so that exceptions can also be renamed?

        mapping = {}
        for key in self.key_iterator(state):
            dest = state["buffer"][key]["dest"]
            if dest is not None:
                mapping[dest] = key

        return mapping

    def update_entry(self, state, rob_id, update):
        if rob_id in self.key_iterator(state):
            entry = state["buffer"][rob_id]
            entry.update(update)
            
            flush = False

            if entry["type"] == "HALT":
                state["halt"] = True
                flush = True

            if entry["pc_value"] is not None:
                flush = True

                if state["size"] < self.config["max_size"]:
                    next_id = (rob_id + 1) % self.config["max_size"]
                    next_id = next_id if next_id in self.key_iterator(state) else None
                    
                    if next_id is not None:
                        next_entry = state["buffer"][next_id]
                        if next_entry["instruction_pc"] == entry["pc_value"]:
                            # don't flush if we predicted correctly
                            flush = False

            if flush:
                keys = list(self.key_iterator(state))
                pos  = keys.index(rob_id)
                flushed_keys = list(k for i, k in enumerate(keys) if i > pos)

                # make the queue finish at pos
                state["size"] = pos + 1

                self.cancel.tx = {
                    "keys": flushed_keys
                }
                
                # update PC
                # TODO add the pc -> pc_next mapping here so that predictor can learn from it
                self.pc_predictor.tx = entry["pc_value"]
                
                return True

    def update(self):
        with self.State() as state, self.Printer() as prints:
            commit_registers = {}

            # TODO do this properly
            update_pcpred = []
            for _ in xrange(self.config["num_commit"]): 
                if state["size"] > 0:
                    commit_key = state["start"]
                    commit_entry = state["buffer"][commit_key]
                   
                    # TODO check for exceptions
                    if commit_entry["pc_value"] is not None and (commit_entry["dest"] is None or commit_entry["dest_value"] is not None):
                        # TODO remove assumption that they're RF destinations
                        if commit_entry["dest"] is not None:
                            commit_registers[commit_entry["dest"]] = commit_entry["dest_value"]

                        update_pcpred.append((commit_entry["instruction_pc"],commit_entry["pc_value"]))

                        prints("Committed {}".format(commit_key))

                        state["start"] = (state["start"] + 1) % self.config["max_size"]
                        state["size"] -= 1

                        state["_num_instructions"] += 1
                else:
                    break

            self.update_predictor.tx = update_pcpred
            self.rf_write.tx = commit_registers


            self.fetch.tx = None if state["size"] >= self.config["max_size"] or state["halt"] \
                        else self.get_next_free(state)
        
            if self.fetch.rx is not None and not state["halt"]:
                rob_id = self.get_next_free(state)
                entry = state["buffer"][rob_id]

                for key in entry:
                    entry[key] = None
                    
                entry["instruction_pc"] = self.fetch.rx["pc"]
                entry["_instruction"]   = self.fetch.rx["_instruction"]
                
                state["size"] += 1
            
            updates = {}
            if self.decode.rx is not None:
                rob_id = self.decode.rx["rob_id"]
                update = ((k,v) for k, v in self.decode.rx.iteritems() if k in ReorderBuffer.record_fields)
                updates[rob_id] = update

            if self.update_bus.rx is not None:
                for rob_id, unfiltered_update in self.update_bus.rx.iteritems():
                    update = ((k,v) for k, v in unfiltered_update.iteritems() if k in ReorderBuffer.record_fields)
                    updates[rob_id] = update

            for key in self.key_iterator(state):
                if key in updates:
                    flushed = self.update_entry(state, key, updates[key])

                    # if we flushed, then the later entries don't exist anymore
                    if flushed:
                        break

            if self.rat.rx is not None:
                reply = {}

                # dict keyed by rob. then inside dict is list of keys
                # that's a lie now. now it's just None
                for rob_id, _ in self.rat.rx.iteritems():
                    subreply = reply[rob_id] = {}
                    keys = list(self.key_iterator(state))

                    if rob_id in keys:
                        prev_keys = keys[0:keys.index(rob_id)]

                        for key in prev_keys:
                            dest = state["buffer"][key]["dest"]
                            if dest is not None:
                                # give it both the renaming and the current value
                                subreply[dest] = (key, state["buffer"][key]["dest_value"])

                self.rat.tx = reply

            for k in self.key_iterator(state):
                entry = state["buffer"][k]
                prints("{}: pc={} dest={} value={} pc_next={}".format(k, entry["instruction_pc"], entry["dest"], entry["dest_value"], entry["pc_value"]))

            prints("Register renaming:")
            for k,v in self.get_naming(state).iteritems():
                prints("  R{}: {}".format(k, "RoB{}".format(v)))

    def key_iterator(self, state):
        return ((state["start"] + i ) % self.config["max_size"] for i in xrange(state["size"]))


'''
still need to decide structures fortuff

things like ALU MAYBE return a value for a register
but ALWAYS return a value for PC


maybe add 'horizontal' indicies to ROB. so for eg when they finish on the bus
they write to [rob_id]_pc
              [rob_id]_dest?
           or [rob_id]_exception_id 
              and also exception_pc ? idk whether this needs to be renamed or if we can just check the above when committing and set it then
                                      can have it be set for every instruction, and just make sure that the jump to address location for exceptions
                                      has a special command to copy hmmmm. idk, we need to copy both at the same time.
                                      this would benefit from a stack i think
                                      almost requires one

                                      OR once again, don't set it for every instruction. and have a reset exception instruction which essentialy throws a no exception exception
                                      when finishing, EUs will say 'finished' with its values and if it had an exception

           so then we can rename our exception register too

note that you cant read your own PC, but you can read exception_pc
'''# TODO add a EXCEP instruction which allows us to manually throw or reset exceptions
'''



'''
reg_prefix = 0
rob_prefix = 1
class Decoder(Component):
    def __init__(self):
        super(Decoder, self).__init__()

        self.add_channel("instruction")
        self.add_channel("decoded_instruction")

        self.add_channel("rob")

        self.add_channel("rob_cancel")

    def update(self):
        if self.instruction.rx is not None:
            instruction = self.instruction.rx["instruction"]
            pc          = self.instruction.rx["pc"]
            pc_add_one  = self.instruction.rx["pc_add_one"]
            rob_id      = self.instruction.rx["rob_id"]

            opcode, reg_read_sel1, reg_read_sel2, reg_write_sel, immediate = instruction
            
            EU = None

            if opcode in ["ADD","SUB","DIV","MUL","AND","OR" ,"XOR", "ADDI","SUBI","DIVI","MULI","ANDI","ORI","XORI"]:
                is_immediate = opcode[-1:] == 'I'

                EU = "ALU"
                EU_control = {
                    "op": opcode[:-1] if is_immediate else opcode
                }

                # for simplicity b is always immediate if it is there. this has implicitions for div. 
                EU_data = {
                    "a": ((reg_prefix, reg_read_sel1), None),
                    "b": ((reg_prefix, reg_read_sel2), None) if not is_immediate else (None, immediate),
                    "pc_add_one": (None, pc_add_one),
                }

                EU_write_reg = reg_write_sel

            elif opcode in ["BEQ","BNE", "BGEZ", "BGTZ", "BLEZ", "BLTZ"]:
                EU = "BRU"
                EU_control = {
                    "op": opcode[1:] if opcode in ["BEQ","BNE"] else opcode,
                }

                EU_data = {
                    "a": ((reg_prefix, reg_read_sel1), None),
                    "pc_true": (None, immediate),
                    "pc_false": (None, pc_add_one),
                }

                if opcode in ["BEQ", "BNE"]:
                    EU_data["b"] = ((reg_prefix, reg_read_sel2), None)
                
                EU_write_reg = None 

            elif opcode in ["J"]:
                self.rob.tx = {
                    "rob_id": rob_id,
                    "pc_value": immediate
                }

            elif opcode in ["NOOP"]:
                self.rob.tx = {
                    "rob_id": rob_id,
                    "pc_value": pc_add_one
                }

            elif opcode in ["JR"]:
                EU = "BR"
                EU_control = {"op":"TRUE"}
                EU_data = {
                    "pc_true": ((reg_prefix, reg_read_sel1), None),
                }

                EU_write_reg = None

            elif opcode in ["HALT"]:
                self.rob.tx = {
                    "rob_id": rob_id,
                    "type": "HALT",
                }

            elif opcode in ["LOAD", "STOR"]:
                EU = "LSU"
                EU_control = { "op": opcode, }
                EU_data = {
                    "addr_a": (None, immediate),
                    "addr_b": ((reg_prefix, reg_read_sel1), None),
                    "pc_add_one": (None, pc_add_one),
                }

                if opcode in ["STOR"]:
                    EU_data["data"] = ((reg_prefix, reg_read_sel2), None)
                    EU_write_reg = None

                if opcode in ["LOAD"]:
                    EU_write_reg = reg_write_sel

            if EU is not None:
                cancel = self.rob_cancel.rx
                if cancel is None or "keys" not in cancel or rob_id not in cancel["keys"]: 
                    self.decoded_instruction.tx = {
                        "rob_id": rob_id,
                        "pc": pc,
                        "EU": EU,
                        "EU_control": EU_control,
                        "EU_data": EU_data,
                    }
                    
                    self.rob.tx = {
                        "rob_id": rob_id,
                        "dest": EU_write_reg, # TODO support other types of dests?
                    }

class InstructionRegisterLoader(Component):
    def __init__(self):
        super(InstructionRegisterLoader, self).__init__()

        self.add_channel("rat")
        self.add_channel("rf_read")

        self.add_channel("ins_in")
        self.add_channel("ins_out")

    def update(self):
        with self.Printer() as prints:
            if self.ins_in.rx is not None:
                rob_id = self.ins_in.rx["rob_id"]
                self.rat.tx = {rob_id: None}
                mapping = self.rat.rx

                if mapping is not None and rob_id in mapping:
                    renames = mapping[rob_id]

                    # i shouldnt actualy have to deep copy, but i dont trust my logic elsewhere
                    instruction = copy.deepcopy(self.ins_in.rx)

                    prints(instruction["EU_data"])
                    for k, (source, value) in instruction["EU_data"].iteritems():
                        if source is not None and source[0] == reg_prefix:
                            if source[1] in renames:
                                renamed = renames[source[1]]
                                instruction["EU_data"][k] = (renamed[0], renamed[1])
                            else:
                                instruction["EU_data"][k] = (None, self.rf_read.rx[source[1]])

                    self.ins_out.tx = instruction

class ReservationStation(Component):
    def __init__(self, num_slots):
        super(ReservationStation, self).__init__()

        self.add_channel("instruction")
        self.add_channel("stall")
        
        self.add_channel("rob_cancel")  # RoB cancel (read only)
        self.add_channel("results") # Results from EUs (read only?)

        # define the semantics to be if named, registers (or rob entries) will either always be written to or instruction get flushed 
        
        # make these config?
        self.eu_counter = 0
        self.eu_channels = {}

        self.config["num_slots"] = num_slots

        self._state = {
            "slots": [],
        }

    def add_eu(self, eu_type):
        key = "eu_{}_{}".format(eu_type, self.eu_counter)
        channel_end = self.add_channel(key)

        if eu_type in self.eu_channels:
            self.eu_channels[eu_type].append(channel_end)
        else:
            self.eu_channels[eu_type] = [channel_end]

        self.eu_counter += 1

        return channel_end

    def update(self):
        with self.State() as state, self.Printer() as prints:
            cancel = self.rob_cancel.rx
            if cancel is not None and "keys" in cancel:
                state["slots"][:] = list(slot for slot in state["slots"] if slot["rob_id"] not in cancel["keys"])

            # NOTE quick fix for full RoB
            if self.instruction.rx is not None and self.instruction.rx["rob_id"] not in (s["rob_id"] for s in state["slots"]):
                # check if it's LOAD/STORE, if we have one pls stall
                # simplification to keep load/stores in order
                lsu_stall = self.instruction.rx["EU"] == "LSU" and any(slot["EU"] == "LSU" for slot in state["slots"])

                if len(state["slots"]) < self.config["num_slots"] and not lsu_stall:
                    state["slots"].append(self.instruction.rx)
                else:
                    self.stall.tx = True
            
            if self.results.rx is not None:
                # iterate along all slots and fill in
                for slot in state["slots"]:
                    for key, (source, value) in slot["EU_data"].iteritems():
                        if source is not None:
                            if source in self.results.rx:
                                slot["EU_data"][key] = (source, self.results.rx[source]["dest_value"])
                        else:
                            # we should never need to update rf or immediates
                            pass

            # get ready slot keys into a form matching eu_channels
            ready_slot_keys = dict((k, []) for k in self.eu_channels)
            for key, slot in enumerate(state["slots"]):
                if all(value is not None for _, value in slot["EU_data"].itervalues()):
                    if slot["EU"] in ready_slot_keys:
                        ready_slot_keys[slot["EU"]].append(key)

            # map eus to slots
            dispatched_slot_keys = []
            for eu_type, eu_channels in self.eu_channels.iteritems():
                free_eus = list(ec for ec in eu_channels if ec.rx) # ec.rx == True when it's not busy
                
                for eu_channel, slot_key in zip(free_eus, ready_slot_keys[eu_type]):
                    slot = state["slots"][slot_key]
                    eu_channel.tx = {
                        "control": slot["EU_control"],
                        "data": dict((key, value) for key, (_, value) in slot["EU_data"].iteritems()),
                        "dest": slot["rob_id"],
                    }
                    dispatched_slot_keys.append(slot_key)

            state["slots"][:] = (v for k,v in enumerate(state["slots"]) if k not in dispatched_slot_keys)

            for v in state["slots"]:
                tick  = u'\u2713'
                cross = u'\u2717'
                data  = u", ".join(cross if val is None else tick for _, val in v["EU_data"].itervalues())
                prints(u"rob_id={} EU={} data={}".format(v["rob_id"],v["EU"],data))

class RegisterFile(Component):
    def __init__(self, num_registers):
        super(RegisterFile, self).__init__()

        self.add_channel("set")
        self.add_channel("get")
    
        self._state["registers"] = list(0 for i in xrange(num_registers))

    def update(self):
        with self.State() as state:
            if self.set.rx is not None:
                for r, v in self.set.rx.iteritems():
                    if r == 0:
                        raise "DONT WRITE TO R0!"
                    
                    state["registers"][r] = v

            self.get.tx = dict(enumerate(state["registers"]))

class LSU(Component):
    def __init__(self, data):
        super(LSU, self).__init__()

        self.add_channel("dispatcher")
        self.add_channel("result")

        self.add_channel("rob_cancel")

        self._state["cycles_left"] = 0
        self._state["current_op"] = None

        # TODO should really make this a dict
        self._state["memory"] = copy.deepcopy(data)

    def update(self):
        with self.Printer() as prints, self.State() as state:
            cancel = self.rob_cancel.rx
            if state["current_op"] is not None and cancel is not None \
                    and "keys" in cancel and state["current_op"]["dest"] in cancel["keys"]: 
                state["cycles_left"] = 0
                state["current_op"]  = None
                prints("Current OP was canceled by RoB")

            if state["cycles_left"] > 0:
                state["cycles_left"] -= 1

            if state["cycles_left"] == 0 and state["current_op"] is not None:
                op = state["current_op"]
                if op["control"]["op"] == "LOAD":
                    self.result.tx = {
                        op["dest"]: {
                            "pc_value": op["data"]["pc_add_one"],
                            "dest_value": state["memory"][op["data"]["addr_a"] + op["data"]["addr_b"]],
                        }
                    }
                else:
                    state["memory"][op["data"]["addr_a"] + op["data"]["addr_b"]] = op["data"]["data"]
                    self.result.tx = {
                        op["dest"]: {
                            "pc_value": op["data"]["pc_add_one"],
                        }
                    }

                state["current_op"] = None
            
            if state["current_op"] is None:
                self.dispatcher.tx = True

            op = self.dispatcher.rx
            prints("Recieved {}".format(op))

            if op is not None and state["current_op"] is None:
                state["current_op"] = op
                state["cycles_left"] = 1 if op["control"]["op"] == "LOAD" else 6

            prints("Currently executing {}".format(state["current_op"]))


class BRU(Component):
    def __init__(self):
        super(BRU, self).__init__()

        self.add_channel("dispatcher")
        self.add_channel("result")

        self.add_channel("rob_cancel")

        self._state["current_op"] = None

    ops = {
        "EQ":   lambda ins: True if ins["a"] == ins["b"] else False,
        "NE":   lambda ins: True if ins["a"] != ins["b"] else False,
        "GEZ":  lambda ins: True if ins["a"] >= 0 else False,
        "GTZ":  lambda ins: True if ins["a"] >  0 else False,
        "LEZ":  lambda ins: True if ins["a"] <= 0 else False,
        "LTZ":  lambda ins: True if ins["a"] <  0 else False,
        "TRUE": lambda _: True
    }

    def update(self):
        with self.Printer() as prints, self.State() as state:
            cancel = self.rob_cancel.rx
            if state["current_op"] is not None and cancel is not None \
                    and "keys" in cancel and state["current_op"]["dest"] in cancel["keys"]: 
                state["current_op"]  = None
                prints("Current OP was canceled by RoB")

            if state["current_op"] is not None:
                op = state["current_op"]
                self.result.tx = {
                    op["dest"]: {
                        "pc_value": op["data"]["pc_true"] if BRU.ops[op["control"]["op"][1:]](op["data"]) else op["data"]["pc_false"],
                    }
                }
                state["current_op"] = None
            
            if state["current_op"] is None:
                self.dispatcher.tx = True

            op = self.dispatcher.rx
            prints("Recieved {}".format(op))

            if op is not None and state["current_op"] is None:
                state["current_op"] = op

            prints("Currently executing {}".format(state["current_op"]))

class ALU(Component):
    operations = {
        "ADD":  (1,lambda ins: ins["a"] + ins["b"]),
        "SUB":  (1,lambda ins: ins["a"] - ins["b"]),
        "DIV":  (5,lambda ins: 0 if ins["b"] == 0 else ins["a"] // ins["b"]),
        "MUL":  (5,lambda ins: ins["a"] * ins["b"]),
        "AND":  (1,lambda ins: ins["a"] & ins["b"]),
        "OR":   (1,lambda ins: ins["a"] | ins["b"]),
        "XOR":  (1,lambda ins: ins["a"] ^ ins["b"]),
    }

    def __init__(self):
        super(ALU, self).__init__()

        self.add_channel("dispatcher")
        self.add_channel("result")

        self.add_channel("rob_cancel")

        self._state["cycles_left"] = 0
        self._state["current_op"] = None

    def update(self):
        with self.Printer() as prints, self.State() as state:
            cancel = self.rob_cancel.rx
            if state["current_op"] is not None and cancel is not None \
                    and "keys" in cancel and state["current_op"]["dest"] in cancel["keys"]: 
                state["cycles_left"] = 0
                state["current_op"]  = None
                prints("Current OP was canceled by RoB")

            if state["cycles_left"] > 0:
                state["cycles_left"] -= 1

            if state["cycles_left"] == 0 and state["current_op"] is not None:
                op = state["current_op"]
                self.result.tx = {
                    op["dest"]: {
                        "pc_value": op["data"]["pc_add_one"],
                        "dest_value": ALU.operations[op["control"]["op"]][1](op["data"]),
                    }
                }
                state["current_op"] = None
            
            if state["current_op"] is None:
                self.dispatcher.tx = True

            op = self.dispatcher.rx
            prints("Recieved {}".format(op))

            if op is not None and state["current_op"] is None:
                state["current_op"] = op
                state["cycles_left"] = ALU.operations[op["control"]["op"]][0]

            prints("Currently executing {}".format(state["current_op"]))

class Processor(object):
    def __init__(self, instructions, data):
        self.cu = ComponentUpdater()

        self.setup_components(instructions, data)

        try:
            u = self.cu.propagate(set(self.components.itervalues())) 
            self.output(u)
        except:
            self.output(self.components)
            raise

    def step(self):
        dirty = self.cu.clock(self.components)
        try:
            updated = self.cu.propagate(dirty)
            self.output(updated)
        except:
            self.output(set(self.components.itervalues()))
            raise

    def run_until_done(self):
        cycle_count = 0

        try:
            while True:
                dirty = self.cu.clock(self.components)
                updated = self.cu.propagate(dirty)

                cycle_count += 1

                if not updated:
                    break
        finally:
            instruction_count = self.components["rob"]._state_next["_num_instructions"]
            print "{} instructions committed after {} cycles.".format(instruction_count, cycle_count)
            print "{:.2f} instructions per cycle".format(float(instruction_count)/cycle_count)


    def step_quiet(self):
        dirty = self.cu.clock(self.components)
        updated = self.cu.propagate(dirty)

    def output(self, updated):
        for component_name, component in self.components.iteritems():
            if component in updated:
                print "{} updated".format(component_name)
                for s in component.print_list:
                    print u"    {}".format(s)

    def setup_components(self, instructions, data):
        def connect(a, b):
            assert a.pair is None and b.pair is None
            a.pair, b.pair = b, a

        pc_predictor      = PCPredictor(64)
        instruction_fetch = InstructionFetch(instructions)
        staller           = DirectionalBus()
        decode            = Decoder()
        irl               = InstructionRegisterLoader()
        register_file     = RegisterFile(16)
        
        rs = ReservationStation(10)

        alus              = list(ALU() for _ in xrange(4))
        brus              = list(BRU() for _ in xrange(1))
        lsu               = LSU(data) 
        
        rob               = ReorderBuffer(128, 4)
        rob_cancel_bus    = DictBroadcastBus()

        results_bus       = DictBroadcastBus()

        connect(staller.add_dirbus_channel(), pc_predictor.stall) 
        connect(staller.add_dirbus_channel(), instruction_fetch.stall) 
        connect(staller.add_dirbus_channel(), rs.stall) 

        connect(rob.pc_predictor, pc_predictor.rob)
        connect(rob.update_predictor, pc_predictor.update_predictor)
        connect(rob.fetch       , instruction_fetch.rob)
        connect(rob.decode      , decode.rob)
        
        connect(rob_cancel_bus.add_bus_channel(), rob.cancel)
        connect(rob_cancel_bus.add_bus_channel(), instruction_fetch.rob_cancel)
        connect(rob_cancel_bus.add_bus_channel(), decode.rob_cancel)
        connect(rob_cancel_bus.add_bus_channel(), rs.rob_cancel)

        for alu in alus:
            connect(rob_cancel_bus.add_bus_channel(), alu.rob_cancel)
            connect(rs.add_eu("ALU"), alu.dispatcher)
            connect(results_bus.add_bus_channel(), alu.result)
        alu = None

        for bru in brus:
            connect(rob_cancel_bus.add_bus_channel(), bru.rob_cancel)
            connect(rs.add_eu("BRU"), bru.dispatcher)
            connect(results_bus.add_bus_channel(), bru.result)
        bru = None

        connect(rob_cancel_bus.add_bus_channel(), lsu.rob_cancel)
        connect(rs.add_eu("LSU"), lsu.dispatcher)
        connect(results_bus.add_bus_channel(), lsu.result)

        connect(irl.rat, rob.rat)

        connect(irl.rf_read , register_file.get)
        connect(rob.rf_write, register_file.set)

        connect(pc_predictor.pc, instruction_fetch.pc)
        connect(instruction_fetch.instruction, decode.instruction)
        connect(decode.decoded_instruction, irl.ins_in)
        connect(rs.instruction, irl.ins_out)
        

        connect(results_bus.add_bus_channel(), rob.update_bus)
        connect(results_bus.add_bus_channel(), rs.results)


        # nice and hacky
        self.components = dict(i for i in locals().iteritems() if isinstance(i[1],Component))
        self.components.update(("alu_{}".format(i),alu) for i,alu in enumerate(alus))
        self.components.update(("bru_{}".format(i),bru) for i,bru in enumerate(brus))

