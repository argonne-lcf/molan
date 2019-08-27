

def cycle_fn(epoch, cycle_length=15, const_epochs=3, start=0.0, stop=1.0):
    cur_epoch = epoch % cycle_length
    grow_length = (cycle_length - const_epochs)
    growing = cur_epoch < grow_length
    rate = (stop - start)
    if growing:
        t = cur_epoch / float(grow_length)
        return rate * t + start
    else:
        return stop


class KLCylicAnnealer:

    @classmethod
    def from_config(cls, config):
        return cls(config.kl_cycle_length, config.kl_cycles_constant, 0.0, 1.0)

    def __init__(self, cycle_length, const_epochs, w_start=0.0, w_end=1.0):
        self.cycle_length = cycle_length
        self.const_epochs = const_epochs
        self.w_start = w_start
        self.w_end = w_end

    def __call__(self, i):
        return cycle_fn(i, self.cycle_length, self.const_epochs,
                        self.w_start, self.w_end)


class WAnnealer:

    @classmethod
    def from_config(cls, config):
        return cls(config.y_start, config.y_end, config.y_w_start, config.y_w_end)

    def __init__(self, start, end, w_start, w_end):
        self.i_start = start
        self.i_end = end
        self.w_start = w_start
        self.w_end = w_end
        self.inc = (self.w_end - self.w_start) / (self.i_end - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        w = min(self.w_start + k * self.inc, self.w_end)
        return w
