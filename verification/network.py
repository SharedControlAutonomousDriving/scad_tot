import enum
from typing import Dict, List, Tuple, Union
from itertools import product

OneHotFeatureDefinition = Tuple[int]
'''Defines a onehot encoded feature as a list of input indexes

Syntax: (INDEX1, INDEX2, ...)

Example: A onehot encoded feature at indexes 3 and 4
(3, 4)
'''

OrdinalFeatureDefinition = Tuple[int, Tuple[Union[int, float]]]
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

CategoricalFeatureAssignment = Tuple[int, Union[int, float]]
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
    '''Defines misclassifications allowed during verification (e.g. TargetedRobustness)
    
    Args:
        definitions (List[AllowedMisclassificationDefinition]): Definitions for the allowed misclassifications
    
    Returns:
        AllowedMisclassifications: the AllowedMisclassifications object instance
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
