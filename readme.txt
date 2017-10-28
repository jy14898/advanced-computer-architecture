to start the processor enter in the terminal:

python -B -i processor.py

This will bring up a python prompt. From here you can either type:

cco.step()

to step a single phase change of the clock. Or you can type:

cco.step_multi()

to step multiple cycles. To inspect the values of registers, type:

registerFile.registers

To inspect the values of data memory, type:

data_memory.memory

The processor currently runs only one program (programs/add_test.py) by default.
