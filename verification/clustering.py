import os, pickle, argparse
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn_extra.cluster import KMedoids
from scipy.spatial import distance
from matplotlib import pyplot as plt
from utils import create_logger, ms_since_1970, tohex, TOTUtils

algos = {'KMeans': KMeans, 'KModes': KModes, 'KPrototypes': KPrototypes, 'KMedoids': KMedoids}
metrics = {'euclidean': distance.euclidean, 'manhattan': distance.cityblock}
init_centroid_choices = ('rand', 'first', 'none')
default_outdir = './logs/clustering'
logger = create_logger('clustering', logdir=default_outdir)
logperf = lambda s: f'({ms_since_1970() - s}ms)'

class LGKClustering:
    '''
    Label Guided K* Clustering
    performs KMeans, KModes, or KPrototypes and divides each cluster into smaller regions that contain a single label
    '''
    def __init__(self, algo='KMeans', metric='euclidean'):
        assert algo in algos, f'algo must be one of ({algos.keys()})'
        assert metric is 'euclidean' if algo is not 'KMedoids' else metric in metrics, 'unsupported metric/algo combination'
        self.__algorithm, self.__fn, self.__metric, self.__distfn = algo, algos[algo], metric, metrics[metric]

    def fit(self, X, y, init_centroid='rand', categorical=[]):
        X_count = X.shape[0]
        assert X_count == y.shape[0], 'X & y must have same number of items'
        assert X_count == len(np.unique(X, axis=0)), 'X must have no duplicates'
        assert init_centroid in init_centroid_choices, 'init_centroid mode must be valid'
        assert not categorical if self.__algorithm is not 'KPrototypes' else True, 'categorical only used by KPrototypes'

        self.__categories = np.unique(y, axis=0)
        get_centroids = (lambda X,y: LGKUtils.get_initial_centroids(X,y, rand=(init_centroid == 'rand'))) if init_centroid != 'none' else (lambda X,y: None)
        logger.info(f'finding regions...')
        start = ms_since_1970()
        remaining, regions = [(X, y)], []
        while len(remaining) > 0:
            X, y = remaining.pop(0)
            n = np.unique(y).shape[0]
            model_params = dict(n_clusters=n)
            if init_centroid:
                model_params['init'] = get_centroids(X, y)
            fit_params = dict()
            model_data = X
            if self.__algorithm == 'KPrototypes':
                fit_params = dict(fit_params, categorical=categorical)
            if self.__algorithm == 'KMedoids':
                del model_params['init']
                model_params = dict(model_params, metric=self.__metric)
            model = self.__fn(**model_params).fit(model_data, **fit_params)
            centroids = model.cluster_centers_ if self.__algorithm == 'KMeans' else model.cluster_centroids_
            yhat = model.predict(model_data)
            for c in np.unique(yhat):
                xis = np.where(yhat == c)[0]
                Xc, yc = X[xis], y[xis]
                if len(np.unique(yc, axis=0)) == 1:
                    regions.append(LGKRegion(centroids[c], Xc, yc))
                else:
                    remaining.append((Xc, yc))
        # assert sum total of region sizes equals num rows in X
        assert(X_count == sum([r.X.shape[0] for r in regions]))
        logger.info(f'found {len(regions)} regions {logperf(start)}')
        self.__regions = regions
        return self
    
    def predict(self, x, y=None):
        regions = self.get_regions(category=y)
        distances = {i:self.__distfn(x, r.centroid) for i,r in enumerate(regions)}
        region = regions[min(distances, key=distances.get)]
        return region
    
    def get_regions(self, category=None, sort=False, sortrev=True):
        regions = self.__regions
        if category is not None:
            assert(category in self.__categories)
            regions = [r for r in regions if r.category == category]
        if sort:
            regions = sorted(regions, key=lambda r:(r.X.shape[0], r.density), reverse=sortrev)
        return regions
    
    def get_categories(self):
        return self.__categories

class LGKRegion:
    '''
    LG Region
    represents a label-guided 'region' which contains inputs of a single label
    '''
    def __init__(self, centroid, X, y, metric='euclidean'):
        assert X.shape[0] == y.shape[0], 'X and y must have same number of items'
        assert len(np.unique(y, axis=0)) == 1, 'all labels in y must be the same'
        assert metric in metrics, 'unsupported metric'
        self.__metric, self.__distfn = metric, metrics[metric]
        self.centroid, self.X, self.y, self.category, self.n = centroid, X, y, y[0], X.shape[0]
        self.radii = [self.__distfn(x, self.centroid) for x in X]
        self.radius = max(self.radii)
        self.density = (self.n / self.radius) if self.radius > 0 else 0

class LGKUtils:
    @staticmethod
    def find_region(lgkc, x, category, metric='euclidean'):
        assert metric in metrics, 'unsupported metric'
        return next([r for r in lgkc.get_regions(category=category) if metrics[metric](x, r.centroid) < r.radius])

    @staticmethod
    def to_categorical(y, n_cats):
        return np.array([[int(yi==i) for i in range(n_cats)] for yi in y])

    @staticmethod
    def get_initial_centroids(X, y, rand=True):
        return np.array([X[np.random.choice(cy) if rand else 0] for cy in [[i for i,yi in enumerate(y) if yi == c] for c in np.unique(y, axis=0)]])
    
    @staticmethod
    def find_input_index(x, X):
        idxs = np.where((X == x).all(axis=1))[0]
        return (idxs[0] if len(idxs) else None)
    
    @staticmethod
    def get_input_class(x, X, y):
        assert(X.shape[0] == y.shape[0])
        idx = LGKUtils.find_input_index(x, X)
        return (y[idx] if idx is not None else None)
    
    @staticmethod
    def save(lgkm, outdir=default_outdir):
        savepath = os.path.join(outdir, 'lgkm.pkl')
        pickle.dump(lgkm, open(savepath, 'wb'))
        logger.info(f'saved to {savepath}')
    
    @staticmethod
    def load(path):
        lgkm = pickle.load(open(path, 'rb'))
        return lgkm
    
    @staticmethod
    def print_regions(lgkm, sort=True):
        regions = lgkm.get_regions(sort=sort)
        logger.info(f'{len(regions)} regions:\n' + '\n'.join([f'y={r.category}, n={len(r.X)}, d={round(r.density, 2)}' for r in regions]))
    
    @staticmethod
    def print_summary(lgkm):
        pass
    
    @staticmethod
    def pair_plot_regions(lgkm, save=False, show=True, inc_x=True, outdir=default_outdir, palette='rainbow_r'):
        logger.info('plotting regions...')
        regions = lgkm.get_regions()
        n_cats = len(lgkm.get_categories())
        X = ([x for r in regions for x in r.X] if inc_x else []) + [r.centroid for r in regions]
        y = ([y for r in regions for y in r.y] if inc_x else []) + [n_cats+r.category for r in regions]
        df = pd.DataFrame(X, columns=TOTUtils.get_feature_names())
        df['y'] = y
        colors = [tohex(r,g,b) for r,g,b in sns.color_palette('rainbow_r', n_cats)]
        palette = {i:colors[i if i < n_cats else i-n_cats] for i in range(n_cats*(2 if inc_x else 1))}
        markers = ['o' if i<n_cats else 'D' for i in range(n_cats*(2 if inc_x else 1))]
        g = sns.pairplot(df, hue='y', corner=True, palette=palette, markers=markers, plot_kws=dict(alpha=0.5, s=10))
        g = g.add_legend({i:l for i,l in enumerate(TOTUtils.get_category_names())})
        if save:
            savepath = os.path.join(outdir, 'lgkm.png')
            g.savefig(savepath, dpi=300)
            logger.info(f'regions plot saved to {savepath}')
        if show:
            plt.show()
    
    @staticmethod
    def tsne_plot_regions(lgkm, save=False):
        pass

    @staticmethod
    def reduce_classes(y, metaclasses=[(0,1), (2,), (3,4)]):
        yprime = y.copy()
        for mc,classes in enumerate(metaclasses):
            for c in classes:
                yprime[y==c] = mc
        return yprime

    @staticmethod
    def load_dataset(csvfile, n_outputs):
        logger.info(f'reading dataset from {csvfile}...')
        # read input and outputs into separate dataframes
        df = pd.read_csv(csvfile, index_col=0).drop_duplicates()
        output_cols = df.columns.tolist()[-n_outputs:]
        output_df, input_df = df[output_cols], df.drop(output_cols, axis=1)
        # drop any duplicate inputs from both dataframes
        dupes = [i for i,d in enumerate(input_df.duplicated()) if d]
        input_df = input_df.drop(input_df.index[dupes], axis=0)
        output_df = output_df.drop(output_df.index[dupes], axis=0)
        # convert to numpy arrays
        X = input_df.values
        y = np.array([output_cols.index(c) for c in (output_df[output_cols] == 1).idxmax(1)])
        return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('-n', '--noutputs', required=True)
    parser.add_argument('-i', '--initcentroid', default='rand', nargs='?', choices=init_centroid_choices)
    parser.add_argument('-o', '--outdir', default=default_outdir)
    # parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-sr', '--saveregions', action='store_true')
    parser.add_argument('-sl', '--savelogs', action='store_true')
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    args = parser.parse_args()
    # configure logger
    for handler in logger.handlers[:]: logger.removeHandler(handler)  
    logger = create_logger('clustering', to_file=args.savelogs, logdir=args.outdir)
    # read dataset, and start clustering
    X, y = LGKUtils.load_dataset(args.file, args.noutputs)
    lgkm = LGKClustering().fit(X, y, init_centroid=args.initcentroid)
    # print regions
    if args.verbosity > 0: LGKUtils.print_regions(lgkm)
    # generate plot png, and save regions
    # if args.plot: LGKUtils.plot_regions(lgkm, save=True, show=False, outdir=args.outdir)
    if args.saveregions: LGKUtils.save(lgkm, outdir=args.outdir)
