#!./venv/bin/python3

import os, pickle, typing
import numpy as np
import pandas as pd
from verification.tot_net import TOTNet
from verification.network import CategoricalFeatureDefinition, CategoricalFeatures, CategoricalFeatureTypes, AllowedMisclassifications
from verification.utils import _set_tf_log_level, ms_since_1970, _ms_to_human, create_dirpath, _parse_onehot_features, _parse_ordinal_features, _parse_allowed_misclassifications, count_decimal_places
from abc import ABCMeta, abstractmethod

_set_tf_log_level()

DEFAULTS = dict(
    e_min=0.0,
    e_max=1.0,
    e_interval=0.0001,
    timeout=0,
    verbosity=1,
    marabou_verbosity=0
    )

# ============================================================
# _BaseRobustness
# ============================================================
class _BaseRobustness(metaclass=ABCMeta):
    def __init__(
        self,
        network_path:str='',
        network_options:dict=dict(),
        X:np.array=np.array([]),
        Y:np.array=np.array([]),
        categorical_features:CategoricalFeatures=None,
        allowed_misclassifications:AllowedMisclassifications=None,
        e_min:float=DEFAULTS['e_min'],
        e_max:float=DEFAULTS['e_max'],
        e_interval:float=DEFAULTS['e_interval'],
        timeout:int=DEFAULTS['timeout'],
        verbosity:int=DEFAULTS['verbosity'],
        marabou_options:dict=dict(),
        marabou_verbosity:int=0
        ):
        self._network_path = network_path
        self._network_options = network_options
        self._X = X
        self._Y = Y
        self._categorical_features = categorical_features
        self._allowed_misclassifications = allowed_misclassifications
        self._e_min = e_min
        self._e_max = e_max
        self._e_interval = e_interval
        self._timeout = timeout
        self._verbosity = verbosity
        self._marabou_options = marabou_options
        self._marabou_verbosity = marabou_verbosity
        self._net = TOTNet(self._network_path,
            network_options=self._network_options,
            categorical_features=self._categorical_features,
            marabou_options=self._marabou_options,
            marabou_verbosity=self._marabou_verbosity
            )
        self._results = []
        self._counterexamples = []
    
    @property
    def net(self) -> TOTNet:
        return self._net

    @property
    def X(self) -> np.array:
        return self._X

    @property
    def Y(self) -> np.array:
        return self._Y
    
    @property
    def dataset(self) -> typing.Tuple[np.array, np.array]:
        return (self.X, self.Y)

    @property
    def results(self) -> pd.DataFrame:
        return self._results

    @results.setter
    def results(self, results:pd.DataFrame):
        self._results = results
    
    def get_counterexample(self, x_index) -> dict:
        return self._counterexamples[x_index]

    def get_counterexamples(self, include_outputs=True):
        if include_outputs:
            return self._counterexamples
        return [c[0] for c in self._counterexamples]
    counterexamples = property(get_counterexamples)

    @counterexamples.setter
    def counterexamples(self, counterexamples:dict):
        self._counterexamples = counterexamples
    
    @abstractmethod
    def _find_counterexample(self, x:np.array, y:np.array, epsilon:float) -> tuple:
        pass
    
    def _get_predicted_label(self, actual_y, outputs:typing.List[float]) -> int:
        maxval = max(outputs)
        for i in enumerate(outputs):
            if i != actual_y and i >= maxval:
                return i
        return actual_y

    def _find_epsilon(self, x:np.array, y:np.array, x_index=None) -> dict:
        lower, upper, interval = self._e_min, self._e_max, self._e_interval
        actual_label = np.argmax(y)
        predicted_label = actual_label
        epsilon = self._e_max
        counterexample = None
        while ((upper - lower) > interval):
            guess = lower + (upper - lower) / 2.0
            pred, cex = self._find_counterexample(x, y, guess)
            if self._verbosity > 1:
                print(f's{x_index}@epsilon={guess}: class={actual_label}, pred_class={pred}')
            if cex is None:
                # correct prediction
                lower = guess
            else:
                # incorrect prediction
                upper = guess
                epsilon = guess
                counterexample = cex
                predicted_label = pred
        return epsilon, predicted_label, counterexample
    
    def _find_epsilon_v2(self, x:np.array, y:np.array, x_index=None):
        dplaces = count_decimal_places(self._e_interval)
        epsilons = [round(e, dplaces+1) for e in np.arange(self._e_min, self._e_max, self._e_interval)]
        e, l, m, h = 0, 0, 0, len(epsilons) - 1
        counterexample, predicted_label, epsilon = None, np.argmax(y), self._e_max
        while l < h:
            m = (h + l) // 2
            e = epsilons[m]
            pred, cex = self._find_counterexample(x, y, e)
            if cex:
                h = m - 1
                counterexample = cex
                predicted_label = pred
                epsilon = e
            else:
                l = m + 1
        return epsilon, predicted_label, counterexample
    
    def analyze(self, results_outpath:str='', counterexamples_outpath:str='', dataset_outpath:str=''):
        results, counterexamples = [], []
        X, Y = self.X, self.Y
        for i in range(X.shape[0]):
            x, y = X[i], Y[i]
            start = ms_since_1970()
            # epsilon, pred_label, counterexample = self._find_epsilon(x, y, x_index=i)
            epsilon, pred_label, counterexample = self._find_epsilon_v2(x, y, x_index=i)
            actual_label = np.argmax(y)
            duration = ms_since_1970() - start
            if self._verbosity > 0:
                print(f's{i}, class:{actual_label}, pred:{pred_label}, epsilon:{epsilon} ({duration} ms)')
            results.append({
                'x':i,
                'class': actual_label,
                'pred_class': pred_label,
                'epsilon':epsilon,
                'time':duration
                })
            # TODO: convert counterexamples to numpy arrays
            counterexamples.append(counterexample)
        
        self.results = pd.DataFrame(results)
        self.counterexamples = counterexamples
        if results_outpath:
            create_dirpath(results_outpath)
            self.results.to_csv(results_outpath)
        if counterexamples_outpath:
            create_dirpath(counterexamples_outpath)
            pickle.dump(self.counterexamples, open(counterexamples_outpath, 'wb'))
        if dataset_outpath:
            create_dirpath(dataset_outpath)
            pickle.dump(self.dataset, open(dataset_outpath, 'wb'))
        if self._verbosity > 0:
            print(f'completed analysis of {self.X.shape[0]} samples in {_ms_to_human(self.results["time"].sum())}')
        return self
    
    def load_results(self, results_path:str='', counterexamples_path:str='', dataset_path:str=''):
        if results_path:
            self.results = pd.read_csv(results_path, index_col=0)
        if counterexamples_path:
            self.counterexamples = pickle.load(open(counterexamples_path, 'rb'))
        if dataset_path:
            self._X, self._Y = pickle.load(open(dataset_path, 'rb'))

# ============================================================
# LocalRobustness
# ============================================================
class LocalRobustness(_BaseRobustness):
    def __init__(
        self,
        network_path:str='',
        network_options:dict=dict(),
        X:np.array=np.array([]),
        Y:np.array=np.array([]),
        categorical_features:CategoricalFeatures=None,
        allowed_misclassifications:AllowedMisclassifications=None,
        e_min:float=DEFAULTS['e_min'],
        e_max:float=DEFAULTS['e_max'],
        e_interval:float=DEFAULTS['e_interval'],
        timeout:int=DEFAULTS['timeout'],
        verbosity:int=DEFAULTS['verbosity'],
        marabou_options:dict=dict(),
        marabou_verbosity:int=DEFAULTS['marabou_verbosity']
        ):
        super().__init__(
            network_path=network_path,
            network_options=network_options,
            X=X,
            Y=Y,
            categorical_features=categorical_features,
            allowed_misclassifications=allowed_misclassifications,
            e_min=e_min,
            e_max=e_max,
            e_interval=e_interval,
            timeout=timeout,
            verbosity=verbosity,
            marabou_options=marabou_options,
            marabou_verbosity=marabou_verbosity
            )

    def _find_counterexample(self, x:np.array, y:np.array, epsilon:float) -> tuple:
        lbs = x-epsilon
        ubs = x+epsilon
        y_idx = np.argmax(y)
        return self.net.find_counterexample(lbs, ubs, y_idx,
            allowed_misclassifications=self._allowed_misclassifications,
            timeout=self._timeout
            )

def test_local_robustness(
    network_path:str='',
    X:np.array=np.array([]),
    Y:np.array=np.array([]),
    e_min:float=DEFAULTS['e_min'],
    e_max:float=DEFAULTS['e_max'],
    e_interval:float=DEFAULTS['e_interval'],
    categorical_features=[],
    allowed_misclassifications=[],
    timeout:int=DEFAULTS['timeout'],
    out_dir:str='./results'
    ):
    results_outpath = os.path.join(out_dir, './results.csv'),
    counterexamples_outpath = os.path.join(out_dir, './counterexamples.p'),

    lr = LocalRobustness(
        network_path=network_path,
        X=X,
        Y=Y,
        e_min=e_min,
        e_max=e_max,
        e_interval=e_interval,
        categorical_features=categorical_features,
        allowed_misclassifications=allowed_misclassifications,
        timeout=timeout
        )
    
    # TODO: pickle entire LocalRobustness object (or as much as possible)
    lr.analyze(
        results_outpath=results_outpath,
        counterexamples_outpath=counterexamples_outpath
        )

def _main(
    network_path:str,
    dataset_path:str,
    chunksize:int=None,
    e_min:float=DEFAULTS['e_min'],
    e_max:float=DEFAULTS['e_max'],
    e_interval:float=DEFAULTS['e_interval'],
    timeout:int=DEFAULTS['timeout'],
    onehot_features:typing.List[str]=[],
    ordinal_features:typing.List[str]=[],
    allowed_misclassifications:typing.List[str]=[],
    out_dir:str='./results'
    ):

    X, Y = pickle.load(open(dataset_path, 'rb'))
    # TODO: Add support for excluded feature combinations
    onehot_features = [{'type':CategoricalFeatureTypes.ONEHOT, 'definition':d} for d in _parse_onehot_features(onehot_features)]
    ordinal_features = [{'type':CategoricalFeatureTypes.ORDINAL, 'definition':d} for d in _parse_ordinal_features(ordinal_features)]
    categorical_features = CategoricalFeatures(onehot_features + ordinal_features)
    allowed_misclassifications = _parse_allowed_misclassifications(allowed_misclassifications)
    # TODO: Handle Chunksize

    test_local_robustness(
        network_path=network_path,
        X=X,
        Y=Y,
        e_min=e_min,
        e_max=e_max,
        e_interval=e_interval,
        timeout=timeout,
        categorical_features=categorical_features,
        allowed_misclassifications=allowed_misclassifications,
        out_dir=out_dir
        )

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-n', '--networkpath',
        required=True,
        help='Path to network')
    parser.add_argument('-d', '--datasetpath',
        required=True,
        help='Path to pickle containing dataset (X, Y)')
    parser.add_argument('-c', '--chunksize',
        type=int,
        default=None,
        help='Analyze dataset in chunks of chunksize.')
    parser.add_argument('-en', '--emin',
        default=DEFAULTS['e_min'],
        help='Minimum epsilon')
    parser.add_argument('-ex', '--emax',
        default=DEFAULTS['e_max'],
        help='Maximum epsilon')
    parser.add_argument('-ei', '--eint',
        default=DEFAULTS['e_interval'],
        help='Interval between epsilons (precision)')
    parser.add_argument('-t', '--timeout',
        default=DEFAULTS['timeout'],
        help='Timeout for marabou queries')
    parser.add_argument('-o', '--onehotfeatures',
        nargs='*',
        default=[],
        help='Onehot encoded features. Onehot indexes separated by comma; Multiple onehots separated by space. (e.g. 1,2,3 7,8)')
    parser.add_argument('-a', '--allowedmisclassifications',
        nargs='*',
        default=[],
        help='Allowed misclassifications (for targeted queries). Actual and allowed class separated by comma; Multiple items separated by space. (e.g. 4,3 2,1)')
    parser.add_argument('-o', '--outdir',
        default='./results',
        help='Output directory for results')
    args = parser.parse_args()

    _main(
        network_path=args.networkpath,
        dataset_path=args.datasetpath,
        chunksize=args.chunksize,
        e_min=args.emin,
        e_max=args.emax,
        e_interval=args.eint,
        timeout=args.timeout,
        onehot_features=args.onehotfeatures,
        allowed_misclassifications=args.allowedmisclassifications,
        out_dir=args.outdir
        )
