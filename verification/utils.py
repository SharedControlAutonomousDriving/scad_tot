
from decimal import Decimal

def print_heading(text):
    print(('-'*40) + f'\n{text}\n')

def count_decimal_places(float):
    return abs(Decimal(str(float)).as_tuple().exponent)

class TOTUtils:
    _features = ('FixationDuration', 'FixationSeq', 'FixationStart', 'FixationX', 
                 'FixationY', 'GazeDirectionLeftZ', 'GazeDirectionRightZ', 'PupilLeft', 
                 'PupilRight', 'InterpolatedGazeX', 'InterpolatedGazeY', 'AutoThrottle', 
                 'AutoWheel', 'CurrentThrottle', 'CurrentWheel', 'Distance3D', 'MPH', 
                 'ManualBrake', 'ManualThrottle', 'ManualWheel', 'RangeW', 'RightLaneDist', 
                 'RightLaneType', 'LeftLaneDist', 'LeftLaneType')
    _categories = ('fast', 'normal', 'slow')

    @staticmethod
    def get_feature_names():
        return list(_features)

    @staticmethod
    def get_feature_name(input_x):
        return _features[input_x]
    
    @staticmethod
    def get_feature_index(feature_name):
        return TOTUtils._features.index(feature_name)
    
    @staticmethod
    def get_category_index(label):
        return TOTUtils._categories.index(label)
    
    @staticmethod
    def get_majority_category_index(output_sample):
        return output_sample.index(max(output_sample))

    @staticmethod
    def get_scaled_value(scikit_scaler,  col_idx, actual_value):
        '''
        returns the scaled value of a single column from
        '''
        dummy = [[0] * len(scikit_scaler.scale_)]
        dummy[0][col_idx] = actual_value
        return scikit_scaler.transform(dummy)[0][col_idx]

    @staticmethod
    def get_actual_value(scikit_scaler, col_idx, scaled_value):
        '''
        returns the actual value of a single column
        '''
        dummy = [[0] * len(scikit_scaler.scale_)]
        dummy[0][col_idx] = scaled_value
        return scikit_scaler.inverse_transform(dummy)[0][col_idx] 

    @staticmethod
    def get_scaled_values(scikit_scaler, actual_values):
        '''
        returns the scaled values
        '''
        assert(len(scikit_scaler.scale_) == len(actual_values))
        return scikit_scaler.transform(actual_values)[0]

    @staticmethod
    def get_actual_values(scikit_scaler, scaled_values):
        '''
        returns the scaled value of a single column from
        '''
        assert(len(scikit_scaler.scale_) == len(scaled_values))
        return scikit_scaler.inverse_transform(scaled_values)[0]
