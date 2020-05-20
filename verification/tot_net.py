from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions
import numpy as np

class TOTNet:
    '''
    Class representing SCAD TOT Marabou network
    '''
    def __init__(self, network_path, property_path='', lbs=None, ubs=None):
        self.network_path = network_path
        self.load_query(property_path)
        if not(lbs is None and ubs is None):
            assert(len(lbs) == len(ubs))
            assert(len(lbs) == self.ipq.getNumInputVariables())
            for i in range(len(lbs)):
                self.set_input_lower_bound(i, lbs[i])
                self.set_input_upper_bound(i, ubs[i])

    def load_query(self, property_path=''):
        self.ipq = MarabouCore.InputQuery()
        MarabouCore.createInputQuery(self.ipq, self.network_path, property_path)
    
    def set_lower_bounds(self, scaled_values):
        assert(len(scaled_values) == self.ipq.getNumInputVariables())
        for x,v in enumerate(scaled_values):
            self.set_input_lower_bound(x, v)

    def set_upper_bounds(self, scaled_values):
        assert(len(scaled_values) == self.ipq.getNumInputVariables())
        for x,v in enumerate(scaled_values):
            self.set_input_upper_bound(x, v)

    def set_input_lower_bound(self, x_index, scaled_value):
        assert(x_index < self.ipq.getNumInputVariables())
        variable = self.ipq.inputVariableByIndex(x_index)
        self.ipq.setLowerBound(variable, scaled_value)
    
    def set_input_upper_bound(self, x_index, scaled_value):
        assert(x_index < self.ipq.getNumInputVariables())
        variable = self.ipq.inputVariableByIndex(x_index)
        self.ipq.setUpperBound(variable, scaled_value)
    
    def adjust_input_upper_bound(self, x_index, adjustment):
        assert(x_index < self.ipq.getNumInputVariables())
        variable = self.ipq.inputVariableByIndex(x_index)
        value = self.ipq.getUpperBound(variable)
        self.ipq.setUpperBound(variable, value+adjustment)
    
    def adjust_input_lower_bound(self, x_index, adjustment):
        assert(x_index < self.ipq.getNumInputVariables())
        variable = self.ipq.inputVariableByIndex(x_index)
        value = self.ipq.getLowerBound(variable)
        self.ipq.setLowerBound(variable, value+adjustment)
    
    def set_expected_category(self, y_index):
        n_outputs = self.ipq.getNumOutputVariables()
        assert(y_index < n_outputs)
        other_cats_y = [y for y in range(n_outputs) if y != y_index]
        for other_y in other_cats_y:
            eq = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq.addAddend(1, self.ipq.outputVariableByIndex(other_y))
            eq.addAddend(-1, self.ipq.outputVariableByIndex(y_index))
            eq.setScalar(0)
            self.ipq.addEquation(eq)
    
    def get_lower_bounds(self):
        n = self.ipq.getNumInputVariables()
        return [self.ipq.getLowerBound(self.ipq.inputVariableByIndex(x)) for x in range(n)]
    
    def get_upper_bounds(self):
        n = self.ipq.getNumInputVariables()
        return [self.ipq.getUpperBound(self.ipq.inputVariableByIndex(x)) for x in range(n)]

    def get_bounds(self):
        return self.get_lower_bounds(), self.get_upper_bounds()

    def solve(self, output_path='', timeout=0):
        options = createOptions(timeoutInSeconds=timeout)
        vals, stats = MarabouCore.solve(self.ipq, options, output_path)
        assignment = ([], [])
        if len(vals) > 0:
            for i in range(self.ipq.getNumInputVariables()):
                assignment[0].append(vals[self.ipq.inputVariableByIndex(i)])
                # assignment.append(f'input {i} = {vals[self.ipq.inputVariableByIndex(i)]}')
            for i in range(self.ipq.getNumOutputVariables()):
                assignment[1].append(vals[self.ipq.outputVariableByIndex(i)])
                # assignment.append(f'output {i} = {vals[self.ipq.outputVariableByIndex(i)]}')
        return [assignment, stats]

    def reset_query(self):
        self.load_query()
