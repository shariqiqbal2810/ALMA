import numpy as np

class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))


class FlatDecayFlatSchedule():

    def __init__(self,
                 start_val,
                 finish_val,
                 start_time,
                 end_time,
                 decay="exp"):

        self.start_val = start_val
        self.finish_val = finish_val
        self.start_time = start_time
        self.end_time = end_time
        self.decay = decay

        if self.decay == "exp":
            self.exp_scaling = (self.start_time - self.end_time) / np.log(self.finish_val)
        elif self.decay == "linear":
            self.delta = (self.start_val - self.finish_val) / (self.end_time - self.start_time)
        else:
            raise Exception("Decay option not recognized")

    def eval(self, T):
        if T >= self.start_time and T <= self.end_time:
            if self.decay == "exp":
                return (np.exp((self.start_time - T) / self.exp_scaling) - self.finish_val) * self.start_val + self.finish_val
            elif self.decay == "linear":
                return self.start_val - self.delta * (T - self.start_time)
        elif T < self.start_time:
            return self.start_val
        else:
            return self.finish_val
