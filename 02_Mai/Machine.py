import simpy


class Machine(simpy.Resource):
    def __init__(self, env, name):
        super().__init__(env, capacity=1)
        self.name = name