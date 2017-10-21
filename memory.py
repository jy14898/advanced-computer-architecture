
class Memory:
    storage = {}
    def get(self, index):
        if index not in storage:
            storage[index] = 0

        return storage[index]

    def set(self, index, value):
        storage[index] = value

