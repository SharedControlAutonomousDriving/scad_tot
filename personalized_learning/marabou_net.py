import enum, sys, os, copy
import numpy as np
import tensorflow as tf
from numbers import Number
from typing import Dict, Iterable, List, Tuple, TypeVar, Union
from itertools import product
from utils import softargmax, get_file_extension

# TODO: remove sys.path.append after maraboupy pip package is available.
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Marabou')))
from maraboupy import Marabou, MarabouCore, MarabouUtils

default_outdir = '../logs'
default_timeout = 0

class MarabouTimeoutError(Exception):
    '''Exception raised when marabou timeout occurs.

    Args:
        stats (object): marabou Statistics object
        message (str, optional): explanation of error
    '''

    def __init__(self, stats:object, message:str=''):
        self.stats = stats
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'MarabouTimeoutError: {self.message}'

OneHotFeatureDefinition = Tuple[int]
'''Defines a onehot encoded feature as a list of input indexes

Syntax: (INDEX1, INDEX2, ...)

Example: A onehot encoded feature at indexes 3 and 4
(3, 4)
'''

OrdinalFeatureDefinition = Tuple[int, Tuple[Number]]
'''Defines an ordinal categorical feature as the input index and possible values.

Syntax: (INDEX, (VAL1, VAL2, ...))

Example: An ordinal categorical feature at index 5 with values 2.0 and 3.0
(5, (2.0, 3.0))
'''

CategoricalFeatureDefinition = Dict[str, Union[OneHotFeatureDefinition, OrdinalFeatureDefinition]]
'''Defines a 'onehot' or 'ordinal' categorical feature. Syntax is {'type':TYPE, 'definition':DEFINITION}.

OneHot Syntax: {'type':'onehot', 'definition':(INDEX1, INDEX2, ...)}
Ordinal Syntax: {'type':'ordinal', 'definition':(INDEX, (VAL1, VAL2, ...))}

Onehot Example:
One two-class onehot feature at x[3] and x[4].
{'type':'onehot', 'definition':(3, 4)}

Ordinal Example:
A two-class ordinal feature at x[5] with values=(2.0, 3.0).
{'type':'ordinal', 'definition':(5, (2.0, 3.0))}
'''

CategoricalFeatureAssignment = Tuple[int, Number]
'''Defines an assignment for a single categorical feature (can also be a single member of a onehot).

Syntax: (FEATURE_INDEX, FEATURE_VALUE)

Example: x[5]=3.0
(5, 3.0)
'''

CategoricalCombination = Tuple[CategoricalFeatureAssignment]
'''Defines a combination of assigned values for a model's categorical features.

Syntax: ((INDEX1, VAL1), (INDEX2, VAL2), (INDEX3, VAL3), ...)

Example: Case where categorical features are assigned as x[3]=1, x[4]=0, x[5]=3.0.
((3, 1), (4, 0), (5, 2.0))
'''

AllowedMisclassificationDefinition = Tuple[int, int]
'''Defines a change in classification that should be allowed (ignored) during verification

Syntax: (ACTUAL_CLASS, ALLOWED_CLASS)

Example: Allow the 0 class to be predicted as 2
(0, 2)
'''

Counterexample = Dict[np.array, np.array]
'''Defines a counterexample (discovered by marabou)

Syntax: (INPUTS, OUTPUTS)
'''

class CategoricalFeatureTypes(str, enum.Enum):
    '''Types of categorical features'''
    ONEHOT='onehot'
    ORDINAL='ordinal'

class CategoricalFeatures:
    '''Defines categorical features (and optionally excludes combinations)

    Args:
        definitions (List[CategoricalFeatureDefinition]): definitions of categorical features
        exclusions (List[CategoricalCombination]): excluded list of feature combinations (ignored during verification)
    
    Returns:
        CategoricalFeatures: the CategoricalFeatures instance
    '''
    def __init__(self,
        definitions:List[CategoricalFeatureDefinition],
        exclusions:List[CategoricalCombination]=[]
        ):
        self.__definitions = definitions
        onehot_defs = self.get_definitions(feature_type=CategoricalFeatureTypes.ONEHOT)
        ordinal_defs = self.get_definitions(feature_type=CategoricalFeatureTypes.ORDINAL)
        self.__excluded_combos = [sorted(c) for c in exclusions]
        self.__combos = self.__find_combos(onehot_defs, ordinal_defs)

    def __find_combos(self, onehot_defs:List[OneHotFeatureDefinition], ordinal_defs:List[OrdinalFeatureDefinition]) -> List[CategoricalFeatureAssignment]:
        '''Finds all possible combinations of values for the defined categorical features.

        Args:
            onehot_defs (List[OneHotFeatureDefinition]): List of onehot feature definitions
            ordinal_defs (List[OrdinalFeatureDefinition]): List of ordinal feature definitions

        Returns:
            List[CategoricalFeatureAssignment]: List of all possible assignments for categorical features
        '''
        # convert onehot definitions to CategoricalFeatureAssignment syntax
        onehot_defs = [(x, (0, 1)) for d in onehot_defs for x in d]
        # combine onehots and ordinals into a single list and find combinations
        categorical_fields = onehot_defs + ordinal_defs
        combos = [sorted(c) for c in tuple(product(*[tuple(product((f,),fvals)) for f,fvals in categorical_fields]))]
        # remove any invalid onehot defs created while finding combinations, and return list.
        return [c for c in combos if sum([1 for o in onehot_defs if len(list(filter(lambda f:f[0] in o and f[1] > 0, c))) != 1]) == 0]
    
    def get_definitions(self, feature_type:CategoricalFeatureTypes=None) -> List[Union[OneHotFeatureDefinition, OrdinalFeatureDefinition]]:
        '''Gets the categorical feature definitions (optionally for a single feature_type using get_definitions)

        Args:
            feature_type (CategoricalFeatureTypes, optional): Gets categorical feature definitions of the specified feature_type. Defaults to None.

        Returns:
            List[Union[OneHotFeatureDefinition, OrdinalFeatureDefinition]]: list of onehot and/or ordinal definitions.
        '''
        if feature_type == CategoricalFeatureTypes.ONEHOT:
            return [d['definition'] for d in self.__definitions if d['type'] == CategoricalFeatureTypes.ONEHOT]
        elif feature_type == CategoricalFeatureTypes.ORDINAL:
            return [d['definition'] for d in self.__definitions if d['type'] == CategoricalFeatureTypes.ORDINAL]
        return self.__definitions
    definitions = property(get_definitions)

    def get_indexes(self, feature_type:CategoricalFeatureTypes=None) -> List[int]:
        '''Get a list of input indexes for categorical features (optionally for a single feature_type using get_indexes).

        Args:
            feature_type (CategoricalFeatureTypes, optional): Get input indexes of the specified type (onehot or ordinal). Defaults to None.

        Returns:
            List[int]: list of input indexes for the onehot and/or ordinal categorical features.
        '''
        onehot_indexes = list(set([x for d in self.get_definitions(feature_type=CategoricalFeatureTypes.ONEHOT) for x in d]))
        ordinal_indexes = list(set([d[0] for d in self.get_definitions(feature_type=CategoricalFeatureTypes.ORDINAL)]))
        if feature_type == CategoricalFeatureTypes.ONEHOT:
            return onehot_indexes
        elif feature_type == CategoricalFeatureTypes.ORDINAL:
            return ordinal_indexes
        return list(set(onehot_indexes + ordinal_indexes))
    indexes = property(get_indexes)

    def get_combos(self, all_combos:bool=False) -> List[CategoricalFeatureAssignment]:
        '''Returns a list of categorical feature combinations (optionally include the excluded combinations using get_combos)

        Args:
            all_combos (bool, optional): When true, also includes the combinations in the 'exclusions' list. Defaults to False.

        Returns:
            List[CategoricalFeatureAssignment]: List of possible combinations of categorical features and their values.
        '''
        if all_combos:
            return self.__combos
        return [c for c in self.__combos if c not in self.__excluded_combos]
    combos = property(get_combos)

class AllowedMisclassifications:
    '''Defines misclassifications allowed during verification (for Targeted Robustness)
    
    Args:
        definitions (List[AllowedMisclassificationDefinition]): Definitions for the allowed misclassifications
    
    Returns:
        AllowedMisclassifications: AllowedMisclassifications object instance
    '''

    def __init__(self, definitions:List[AllowedMisclassificationDefinition]):
        self.__definitions = definitions
    
    def get_allowed_classes(self, y:int=None) -> Tuple[int]:
        '''Gets a list of allowed classes (optionally for a single original class using get_allowed_classes)

        Args:
            y (int, optional): Original class to get allowed misclassifications for. Defaults to None.

        Returns:
            tuple(int): Tuple containing a list of allowed class indexes (excluding the original class)
        '''
        if y is None:
            return tuple([y_allowed for _, y_allowed in self.__definitions])
        return tuple([y_allowed for y_orig, y_allowed in self.__definitions if y_orig == y])
    allowed_classes = property(get_allowed_classes)

class MarabouNet:
    '''Class representing Marabou network

    Args:
        network_path (str): path to network (nnet, pb, or onnx)
        network_options (dict): options passed to MarabouNetwork constructor
        categorical_features (CategoricalFeatures): object defining the network's categorical features
        ignored_features (List[int]): features which are not perturbed during verification.
        marabou_verbosity (int): amount of marabou logging
        marabou_logdir (str): directory where marabou logs should be stored
        marabou_options (dict): options passed to Marabou.marabouOptions

    Raises:
        MarabouTimeoutError: raised when a marabou query times out

    Returns:
        object: the MarabouNet instance
    '''
    def __init__(self,
        network_path:str,
        network_options:dict=dict(),
        categorical_features:CategoricalFeatures=None,
        ignored_features:List[int]=[],
        marabou_verbosity:int=0,
        marabou_logdir:str='./',
        marabou_options:dict=dict(),
        timeout:int=0
        ):
        self._network_path = network_path
        self._network_options = network_options
        self._categorical_features = CategoricalFeatures([]) if categorical_features is None else categorical_features
        self._ignored_features = ignored_features
        self._marabou_verbosity = marabou_verbosity
        self._marabou_options = marabou_options
        self._marabou_logfile = os.path.join(marabou_logdir, 'marabou.log') if marabou_logdir else None
        self._marabou_timeout = timeout
        self.network = self._load_network()

    @property
    def ignored_features(self) -> tuple:
        '''ignored_features property

        Returns:
            tuple: indexes of features which won't be perturbed.
        '''
        return tuple(self._ignored_features)

    @property
    def num_inputs(self) -> int:
        '''num_inputs property

        Returns:
            int: number of inputs in network
        '''
        return self.network.inputVars[0].flatten().shape[0]
    
    @property
    def num_outputs(self) -> int:
        '''num_outputs property

        Returns:
            int: number of outputs in network
        '''
        return self.network.outputVars[0].flatten().shape[0]
    
    def get_input_var(self, x_index:int) -> int:
        '''gets marabou variable for one of the network's inputs

        Args:
            x_index (int): the input index

        Returns:
            int: variable number for the input
        '''
        assert(x_index < self.num_inputs)
        return self.network.inputVars[0].flatten()[x_index]
    
    def get_output_var(self, x_index:int) -> int:
        '''gets marabou variable for one of the network's outputs

        Args:
            x_index (int): the output index

        Returns:
            int: variable number for the output
        '''
        assert(x_index < self.num_inputs)
        return self.network.outputVars[0].flatten()[x_index]
    
    def set_lower_bounds(self, scaled_values:Iterable[Number]):
        '''Sets lower bounds for each of the network's inputs

        Args:
            scaled_values (Iterable[Number]): Iterable object containing the lower bounds
        '''
        assert len(scaled_values) == self.num_inputs, 'number of lower bound vals must be equal to num_inputs'
        for x,v in enumerate(scaled_values):
            if v is not None:
                self.set_input_lower_bound(x, v)

    def set_upper_bounds(self, scaled_values:Iterable[Number]):
        '''Sets upper bounds for each of the network's inputs

        Args:
            scaled_values (Iterable[Number]): Iterable object containing the upper bounds
        '''
        assert len(scaled_values) == self.num_inputs, 'number of upper bound vals must be equal to num_inputs'
        for x,v in enumerate(scaled_values):
            if v is not None:
                self.set_input_upper_bound(x, v)

    def set_input_lower_bound(self, x_index:int, scaled_value:Number):
        '''Sets the lower bound for one of the network's inputs

        Args:
            x_index (int): the input's index
            scaled_value (Number): the lower bound
        '''
        variable = self.get_input_var(x_index)
        self.network.setLowerBound(variable, scaled_value)
    
    def set_input_upper_bound(self, x_index:int, scaled_value:Number):
        '''Sets the upper bound for one of the network's inputs

        Args:
            x_index (int): the input's index
            scaled_value (Number): the upper bound
        '''
        variable = self.get_input_var(x_index)
        self.network.setUpperBound(variable, scaled_value)
    
    def set_expected_category(self, y_index:int, allowed_misclassifications:AllowedMisclassifications=None):
        '''Sets up the Marabou output query

        Args:
            y_index (int): The index of the expected label
            allowed_misclassifications (AllowedMisclassifications, optional): Additional labels which are allowed. Defaults to None.
        '''        
        n_outputs = self.num_outputs
        assert(y_index < n_outputs)
        allowed_classes = tuple() if allowed_misclassifications is None else allowed_misclassifications.get_allowed_classes(y=y_index)
        other_ys = [y for y in range(n_outputs) if (y != y_index) and (y not in allowed_classes)]
        for other_y in other_ys:
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
            eq.addAddend(1, self.get_output_var(other_y))
            eq.addAddend(-1, self.get_output_var(y_index))
            eq.setScalar(0)
            self.network.addEquation(eq)

    def solve(self) -> Tuple[Tuple[List[Number], List[Number]], MarabouCore.Statistics]:
        '''Solves the input query encoded in the MarabouNetwork object

        Returns:
            Tuple[Tuple[List[Number], List[Number]], MarabouCore.Statistics]: tuple containing a counterexample and marabou statistics.
        '''
        options = {'timeoutInSeconds': self._marabou_timeout, 'verbosity': self._marabou_verbosity, **self._marabou_options}
        vals, stats = self.network.solve(verbose=bool(self._marabou_verbosity), options=Marabou.createOptions(**options))
        assignment = ([], [])
        if len(vals) > 0:
            for i in range(self.num_inputs):
                assignment[0].append(vals[self.get_input_var(i)])
            for i in range(self.num_outputs):
                assignment[1].append(vals[self.get_output_var(i)])
        return assignment, stats
    
    def find_counterexample(self, lbs:Iterable[Number], ubs:Iterable[Number], y:int, allowed_misclassifications:AllowedMisclassifications=None) -> Tuple[int, Counterexample]:
        '''Finds any counterexamples within the specified bounds where the network's output is not 'y' or one of the 'allowed_misclassifications'.

        Args:
            lbs (Iterable[Number]): Lower bounds for each input
            ubs (Iterable[Number]): Upper bounds for each input
            y (int): The expected (actual) label.
            allowed_misclassifications (AllowedMisclassifications, optional): other labels which are allowed. Defaults to None.

        Raises:
            MarabouTimeoutError: raised if marabou exceeds the specified timeout.

        Returns:
            Tuple[int, Tuple[np.array, np.array]]: Tuple containg the predicted label and counterexample.
        '''
        assert len(lbs) == len(ubs) == self.num_inputs, 'lbs and ubs must be same size a num_inputs'
        assert y < self.num_outputs, 'y must be less that num_outputs'

        def _solve_for_output(y, _lbs:Iterable[Number], _ubs:Iterable[Number]) -> Counterexample:
            '''helper function to solve for a given output'''
            self.reset()
            self.set_lower_bounds(_lbs)
            self.set_upper_bounds(_ubs)
            self.set_expected_category(y, allowed_misclassifications=allowed_misclassifications)
            result = self.solve()
            # TODO: look into returning & saving statistics
            vals, stats = result
            if stats.hasTimedOut():
                raise MarabouTimeoutError(stats, f'Marabou query exceeded timeout of {self._marabou_timeout}.')
            inputs, outputs = vals
            return (np.array(inputs), np.array(outputs))

        y_idxs = [oy for oy in range(self.num_outputs) if oy != y]
        for y_idx in y_idxs:
            if len(self._categorical_features.combos) > 0:
                # for networks with categorical features, solve a query for each possible combination.
                for combo in self._categorical_features.combos:
                    combo = {f:fval for f,fval in combo}
                    _lbs = [combo.get(x, v) for x,v in enumerate(lbs)]
                    _ubs = [combo.get(x, v) for x,v in enumerate(ubs)]
                    cex = _solve_for_output(y_idx, _lbs, _ubs)
                    cex_inputs, cex_outputs = cex
                    if np.any(cex_inputs) or np.any(cex_outputs):
                        return y_idx, cex
            else:
                # for networks without categorical features, just a single query per output.
                cex = _solve_for_output(y_idx, lbs, ubs)
                cex_inputs, cex_outputs = cex
                if np.any(cex_inputs) or np.any(cex_outputs):
                    return y_idx, cex
        return y, None

    def check_prediction(self, x:Iterable[Number], y:int) -> bool:
        '''Checks the prediction of a given input against the specified label.

        Args:
            x (Iterable[Number]): the input (x)
            y (int): the expected label.

        Returns:
            bool: true if prediction is same as y, or false otherwise.
        '''
        pred = self.evaluate(x)
        pred_val = max(pred)
        pred_idxs = [i for i,v in enumerate(pred) if v == pred_val]
        return len(pred_idxs) == 1 and pred_idxs[0] == y

    def evaluate(self, x:Iterable[Number]) -> Iterable[Number]:
        '''Makes a prediction

        Args:
            x (Iterable[Number]): The input (x)

        Returns:
            Iterable[Number]: The network's prediction.
        '''
        options = Marabou.createOptions(verbosity=bool(self._marabou_verbosity > 1))
        return self.network.evaluate([x], options=options)[0]

    def reset(self):
        '''Reloads the network (and drops any previous queries or modifications)
        '''
        self.network = self._load_network()
    
    def _load_network(self) -> Marabou.MarabouNetwork:
        '''loads the network as a MarabouNetwork object

        Returns:
            Marabou.MarabouNetwork: the MarabouNetwork object instance.
        '''
        valid_exts = ('.nnet', '', '.pb', '.onnx')
        ext = get_file_extension(self._network_path)
        assert ext in valid_exts, 'Model must be in nnet, pb, or onnx format'
        if ext == '.nnet':
            return Marabou.read_nnet(self._network_path, **self._network_options)
        elif ext in ('', '.pb'):
            return Marabou.read_tf(self._network_path, **self._network_options)
        elif ext == '.onnx':
            return Marabou.read_onnx(self._network_path, **self._network_options)
        return None

class safescad_Net(MarabouNet):
    def __init__(self,
        network_path:str,
        network_options:dict=dict(),
        marabou_verbosity:int=0,
        marabou_logdir:str='./',
        marabou_options:dict=dict(),
        timeout:int=0
        ):
        # categorical_features = CategoricalFeatures([
        #     {'type': CategoricalFeatureTypes.ORDINAL, 'definition':(22, (0.1844895800457542, -5.615958110066326, -2.715734265010286, -8.516181955122367))}, # RightLaneType
        #     {'type': CategoricalFeatureTypes.ORDINAL, 'definition':(24, (-0.1623221645152569, 5.48060923307527, 11.123540630665795, -5.805253562105784))}   # LeftLaneType
        #     ])
        # ignored_features = [22, 24]
        #
        # super().__init__(
        #     network_path=network_path,
        #     network_options=network_options,
        #     categorical_features=categorical_features,
        #     ignored_features=ignored_features,
        #     marabou_verbosity=marabou_verbosity,
        #     marabou_logdir=marabou_logdir,
        #     marabou_options=marabou_options,
        #     timeout=timeout
        #     )

        super().__init__(
            network_path=network_path,
            network_options=network_options,
            marabou_verbosity=marabou_verbosity,
            marabou_logdir=marabou_logdir,
            marabou_options=marabou_options,
            timeout=timeout
            )