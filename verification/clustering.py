import os, pickle, argparse
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from utils import create_logger, ms_since_1970

default_outdir = './logs/clustering'
logger = create_logger('clustering', logdir=default_outdir)
logperf = lambda s: f'({ms_since_1970() - s}ms)'

class LGKMeansRegion:
    '''
    Label Guided KMeans Region
    represents a 'region' of a KMeans cluster which contains inputs of a single label
    '''
    def __init__(self, centroid, X, y):
        assert(X.shape[0] == y.shape[0])
        assert(len(np.unique(y, axis=0)) == 1)
        self.centroid, self.X, self.y, self.category = centroid, X, y, y[0]

class LGKMeans:
    '''
    Label Guided KMeans Clustering
    performs KMeans and divides each cluster into smaller regions that contain a single label
    '''
    def __init__(self):
        pass

    def fit(self, X, y, randinit=True):
        n_inputs = X.shape[0]
        assert(n_inputs == y.shape[0]) # X & y must have same number of items
        assert(n_inputs == len(np.unique(X, axis=0))) # X must have no duplicates
        remaining, regions = [(X, y)], []
        start = ms_since_1970()
        logger.info(f'finding regions from |X|={X.shape[0]}')
        while len(remaining) > 0:
            X, y = remaining.pop(0)
            ics = LGKMeansUtils.get_initial_centroids(X, y, rand=randinit)
            model = KMeans(n_clusters=len(ics), init=ics).fit(X)
            yhat = model.predict(X)
            for c in np.unique(yhat):
                centroid = model.cluster_centers_[c]
                xis = np.where(yhat == c)[0]
                Xc, yc = X[xis], y[xis]
                if len(np.unique(yc)) == 1:
                    regions.append(LGKMeansRegion(centroid, Xc, yc))
                else:
                    remaining.append((Xc, yc))
        # assert sum total of region sizes equals num rows in X
        assert(n_inputs == sum([r.X.shape[0] for r in regions]))
        logger.info(f'found {len(regions)} regions {logperf(start)}')
        self.__regions = regions
        return self
    
    def get_regions(self):
        return self.__regions

class LGKMeansUtils:
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
        idx = LGKMeansUtils.find_input_index(x, X)
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
    def print_regions(lgkm):
        logger.info(f'{len(lgkm.get_regions())} regions:\n' + '\n'.join([f'y={r.category}, |X|={len(r.X)}' for r in lgkm.get_regions()]))
    
    @staticmethod
    def plot_regions(lgkm, save=False, outdir=default_outdir):
        logger.info('plotting regions...')
        df = pd.DataFrame([r.centroid for r in lgkm.get_regions()])
        df['y'] = [r.category for r in lgkm.get_regions()]
        plot = sns.pairplot(df, hue='y', corner=True)
        if save:
            savepath = os.path.join(outdir, 'lgkm.png')
            plt.savefig(savepath, dpi=300)
            logger.info(f'regions plot saved to {savepath}')
        return plot

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
    parser.add_argument('-r', '--randinit', action='store_true')
    parser.add_argument('-o', '--outdir', default=default_outdir)
    parser.add_argument('-sr', '--saveregions', action='store_true')
    parser.add_argument('-sp', '--saveplot', action='store_true')
    parser.add_argument('-sl', '--savelogs', action='store_true')
    args = parser.parse_args()
    # configure logger
    for handler in logger.handlers[:]: logger.removeHandler(handler)  
    logger = create_logger('clustering', to_file=args.savelogs, logdir=args.outdir)
    # read dataset, and start clustering
    X, y = LGKMeansUtils.load_dataset(args.file, args.noutputs)
    lgkm = LGKMeans().fit(X, y, randinit=args.randinit)
    # print regions
    LGKMeansUtils.print_regions(lgkm)
    # generate plot png, and save regions
    if args.saveplot: LGKMeansUtils.plot_regions(lgkm, args.saveplot, args.outdir)
    if args.saveregions: LGKMeansUtils.save(lgkm, args.outdir)
