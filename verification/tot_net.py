import os, copy
import numpy as np
from maraboupy import Marabou, MarabouCore, MarabouUtils

default_outdir = './logs'
default_timeout = 0

class TOTNet:
    '''
    Class representing SCAD TOT Marabou network
    '''
    def __init__(self, network_path, lbs=None, ubs=None, marabou_verbosity=0, marabou_logdir=None):
        self.__original_nnet = Marabou.read_nnet(network_path)
        self.network = copy.deepcopy(self.__original_nnet)
        if not(lbs is None and ubs is None):
            assert(len(lbs) == len(ubs))
            assert(len(lbs) == self.get_num_inputs())
            self.set_lower_bounds(lbs)
            self.set_upper_bounds(ubs)
        self.__marabou_verbosity = marabou_verbosity
        self.__marabou_logfile = os.path.join(marabou_logdir, 'marabou.log') if marabou_logdir else None

    def get_num_inputs(self):
        return len(self.network.inputVars[0])
    
    def get_num_outputs(self):
        return len(self.network.outputVars[0])
    
    def set_lower_bounds(self, scaled_values):
        assert(len(scaled_values) == self.get_num_inputs())
        for x,v in enumerate(scaled_values):
            self.set_input_lower_bound(x, v)

    def set_upper_bounds(self, scaled_values):
        assert(len(scaled_values) == self.get_num_inputs())
        for x,v in enumerate(scaled_values):
            self.set_input_upper_bound(x, v)
    
    def get_input_var(self, x_index):
        assert(x_index < self.get_num_inputs())
        return self.network.inputVars[0][x_index]
    
    def get_output_var(self, x_index):
        assert(x_index < self.get_num_inputs())
        return self.network.outputVars[0][x_index]

    def set_input_lower_bound(self, x_index, scaled_value):
        variable = self.get_input_var(x_index)
        self.network.setLowerBound(variable, scaled_value)
    
    def set_input_upper_bound(self, x_index, scaled_value):
        variable = self.get_input_var(x_index)
        self.network.setUpperBound(variable, scaled_value)
    
    def set_expected_category(self, y_index):
        n_outputs = self.get_num_outputs()
        assert(y_index < n_outputs)
        other_ys = [y for y in range(n_outputs) if y != y_index]
        for other_y in other_ys:
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
            eq.addAddend(1, self.get_output_var(other_y))
            eq.addAddend(-1, self.get_output_var(y_index))
            eq.setScalar(0)
            self.network.addEquation(eq)
    
    def get_input_lower_bound(self, x):
        return self.network.lowerBounds[self.get_input_var(x)]
    
    def get_input_upper_bound(self, x):
        return self.network.upperBounds[self.get_input_var(x)]

    def get_lower_bounds(self):
        n = self.get_num_inputs()
        return [self.get_input_lower_bound(x) for x in range(n)]
    
    def get_lower_bounds(self):
        n = self.get_num_inputs()
        return [self.get_input_upper_bound(x) for x in range(n)]

    def get_bounds(self):
        return self.get_lower_bounds(), self.get_upper_bounds()

    def solve(self, timeout=default_timeout):
        options = Marabou.createOptions(timeoutInSeconds=timeout, verbosity=self.__marabou_verbosity)
        vals, stats = self.network.solve(verbose=bool(self.__marabou_verbosity), options=options)
        assignment = ([], [])
        if len(vals) > 0:
            for i in range(self.get_num_inputs()):
                assignment[0].append(vals[self.get_input_var(i)])
            for i in range(self.get_num_outputs()):
                assignment[1].append(vals[self.get_output_var(i)])
        return assignment, stats
    
    def find_counterexample(self, lbs, ubs, y, timeout=default_timeout):
        assert(len(lbs) == self.get_num_inputs())
        assert(y < self.get_num_outputs())
        other_ys = [oy for oy in range(self.get_num_outputs()) if oy != y]
        for oy in other_ys:
            self.reset()
            self.set_lower_bounds(lbs)
            self.set_upper_bounds(ubs)
            self.set_expected_category(oy)
            vals, stats = self.solve(timeout=timeout)
            if len(vals[0]) > 0 or len(vals[1]) > 0:
                return (vals, stats)
        return None

    def check_prediction(self, inputs, y):
        pred = self.evaluate(inputs)
        pred_val = max(pred)
        pred_idxs = [i for i,v in enumerate(pred) if v == pred_val]
        return len(pred_idxs) == 1 and pred_idxs[0] == y

    def evaluate(self, input_vals):
        options = Marabou.createOptions(verbosity=bool(self.__marabou_verbosity))
        return self.network.evaluate([input_vals], options=options)[0]

    def reset(self):
        self.network = copy.deepcopy(self.__original_nnet)
