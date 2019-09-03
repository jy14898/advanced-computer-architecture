to start the processor enter in the terminal:

python -B -i processor.py

This will bring up a python prompt. From here you can construct a processor:

p = Processor(instructions, data)

to get instructions and data, you need to import a program. Do this like so:

import programs.fibb

and then execute

instructions = programs.fibb.program["instructions"]
data         = programs.fibb.program["data"]

Now when you create a processor, it will have the program instructions and memory loaded

to single step through, type 

p.step() 

or

p.step_quiet()

to run the program to end, or for n cycles, call:

p.run_until_done( optional_number_of_cycles )

to view component state, you can access like so:

p.components["register_file"]._state (or _state_next)
