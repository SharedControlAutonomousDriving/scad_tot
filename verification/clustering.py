import os, pickle, math, random
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from scipy.spatial import distance
from utils import create_dirpath, create_logger, ms_since_1970
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

default_outdir = './logs/clustering'
logger = create_logger('clustering', logdir=default_outdir)

init_centroid_choices = ('rand', 'first', 'kmeans++')

class LabelGuidedKMeans:
    '''
    Class represeting a set of label-guided k-means regions

    Properties
        regions      : list of all LabelGuidedKMeansRegion objects
        categories   : 1D of 2D array of unique categories
        n_categories : number of unique categories (labels)
    
    Functions
        fit            : generates regions from the input data (performs clustering)
        predict        : finds a matching region for a given x and y
        get_regions    : getter function for regions with optional sorting & filtering
        get_categories : getter function for categories with optional onehot encoding
    '''
    def __init__(self):
        ''' 
        initializer, sets up object
        '''
        self._regions = []

    def fit(self, X, Y, init_centroid='rand', weighted_column=None):
        '''
        fits the LGKMeans model to the input data to generate regions

        Parameters
            X             : np.array of inputs
            Y             : np.array of integer labels OR np.array of one-hot labels
            init_centroid : string (rand, first, kmeans++)                    
                                rand     - chooses a random point of each label
                                first    - chooses the first point of each label
                                kmeans++ - uses native 'kmeans++' algorithm (ignoring labels)
            weighted_column : int - the index of the column to use for KMeans sample_weights
        
        Return
            LabelGuidedKMeans object
        '''
        assert len(X.shape) == 2, 'Expected a 2D numpy array (n, width)'
        assert X.shape[0] == Y.shape[0], 'X & Y must have same number of items'
        assert X.shape[0] == np.unique(X, axis=0).shape[0], 'X must have no duplicates'
        assert init_centroid in init_centroid_choices, f'init_centroid mode must be one of {init_centroid_choices}'

        start_time = ms_since_1970()
        self._X, self._Y = X.copy(), np.array([LabelGuidedKMeansUtils.from_categorical(y) for y in Y])
        # convert categories to array of integers if onehot encoded
        self._categories = np.unique(self._Y, axis=0)
        logger.info(f'running label-guided k-means on {self._X.shape[0]} inputs of {self._categories.shape[0]} labels')

        # Use data from weighted_column if provided. Otherwise, set all weights equally.
        weights = self._X[:, weighted_column] if weighted_column != None else np.ones((self._X.shape[0],))
        remaining, regions = [(self._X, self._Y, weights)], []
        while len(remaining) > 0:
            # get data to work on
            X, Y, weights = remaining.pop(0)
            n = np.unique(Y, axis=0).shape[0]
            # setup KMeans params and get initial centroids
            model_params = dict(n_clusters=n)
            if init_centroid != 'kmeans++':
                model_params['init'] = LabelGuidedKMeansUtils.get_initial_centroids(X, Y, rand=(init_centroid=='rand'))
            # create kmeans clusters, get the centroids, and count labels in each cluster
            model = KMeans(**model_params).fit(X, Y, sample_weight=weights)
            centroids = model.cluster_centers_
            Yhat = model.predict(X)
            # create kmeans clusters, get the centroids, and count labels in each cluster
            for c in np.unique(Yhat, axis=0):
                xis = np.where(Yhat == c)[0]
                Xc, Yc, wc = X[xis], Y[xis], weights[xis]
                if len(np.unique(Yc, axis=0)) == 1:
                    # cluster only contained a single label, so save it as a 'region'
                    regions.append(LabelGuidedKMeansRegion(centroids[c], Xc, Yc, self._categories.shape[0]))
                else:
                    # cluster contained two or more labels, so repeat KMeans on the cluster.
                    remaining.append((Xc, Yc, wc))

        # sanity check the regions
        assert self._X.shape[0] == sum([r.X.shape[0] for r in regions]), 'sum total of region sizes should equal num rows in X'
        assert all([np.unique(r.Y, axis=0).shape[0] == 1 for r in regions]), 'all points in each region should have the same label'

        logger.info(f'completed in {ms_since_1970() - start_time} ms')
        self._regions = regions
        return self
    
    def predict(self, x, y=None):
        '''
        finds a region

        Parameters
            x             : np array (a single input)
            y             : a single label (optional)
            init_centroid : string (rand, first, none)
        
        Return
            The closest LabelGuidedKMeansRegion to x with a matching y
        '''
        regions = self.get_regions(category=y)
        distances = {i:distance.euclidean(x, r.centroid) for i,r in enumerate(regions)}
        region = regions[min(distances, key=distances.get)]
        return region
    
    def get_regions(self, category=None, sort=False, sortrev=True):
        '''
        getter for regions

        Parameters
            category : (integer or one-hot encoded) return only categories for a given category
            sort     : bool (if true, returns in ascending sorted order)
            sortrev  : bool (reverses sort order)
        
        Return
            list of LabelGuidedKMeansRegion objects
        '''
        regions = self._regions
        # convert category to integer if one-hot encoded
        if category is not None:
            category = category if isinstance(category, (int, np.integer)) else np.argmax(category)
            assert category in self.categories, f'regions with category {category} do not exist'
            regions = [r for r in regions if r.category == category]
        if sort:
            regions = sorted(regions, key=lambda r:(r.X.shape[0], r.density), reverse=sortrev)
        return regions
    regions = property(get_regions)
    
    def get_categories(self, onehot=False):
        '''
        custom getter for 'categories'

        Parameters
            onehot : if true, converts categories to one-hot encoding
        
        Return
            np.array of integers OR np.array of one-hot encoded values
        '''
        return np.array([LabelGuidedKMeansUtils.to_categorical(c, self._categories.shape[0]) for c in self._categories]) if onehot else self._categories
    categories = property(get_categories)

    @property
    def n_categories(self): return self.categories.shape[0]

class LabelGuidedKMeansRegion:
    '''
    Class represeting a label-guided k-means region

    Properties
        centroid : np array, the center of the region
        radius   : float, the region's radius
        n        : integer, number of inputs in the region
        density  : float, region's density (n / radius)
        category : label (y) for the region
        X        : returns the region's inputs
        Y        : returns the region's labels
    
    Functions
        get_category : getter for category with optional onehot encoding
        get_X        : getter for X with optional sorting
        get_Y        : getter for Y with optional onehot encoding
    '''
    def __init__(self, centroid, X, Y, ncats):
        '''
        initializer - sets up object and calculates n, radius, and density
        
        Parameters
            centroid : np array, represents center of region
            X        : np array, inputs in region
            Y        : np array, labels for inputs
            ncats    : total number of categories (for one-hot encoding)
        '''
        assert X.shape[0] == Y.shape[0], 'X and Y must have same number of items'
        assert np.unique(Y, axis=0).shape[0] == 1, 'all labels in Y must be the same'

        self._centroid = centroid
        self._X = X
        self._Y = Y
        self._category = Y[0]
        self._n = self._X.shape[0]
        self._radius = max([distance.euclidean(x, self.centroid) for x in X])
        self._density = (self.n / self.radius) if self.radius > 0 else 0
        self._ncats = ncats

    @property
    def centroid(self): return self._centroid

    @property
    def density(self): return self._density

    @property
    def radius(self): return self._radius
    
    @property
    def n(self): return self._n

    def get_category(self, onehot=False):
        '''
        custom getter for category

        Parameters
            onehot : when true, returns category in onehot encoding
        
        Return
            integer OR one-hot encoded np.array
        '''
        return LabelGuidedKMeansUtils.to_categorical(self._category, self._ncats) if onehot else self._category
    category = property(get_category)

    def get_X(self, sort=False, sortrev=False):
        '''
        custom getter for 'X'

        Parameters
            sort    : bool, returns points sorted by dist from centroid (smallest to largest)
            sortrev : bool, reverses order of sorting (largest to smallest)
        
        Return
            np.array of all original inputs (x) in the region
        '''
        return np.array(sorted(self._X, key=lambda x: distance.euclidean(x, self._centroid), reverse=sortrev)) if sort else self._X
    X = property(get_X)

    def get_Y(self, onehot=False):
        '''
        custom getter for 'Y'

        Parameters
            onehot : bool, returns Y in onehot encoding when true
        
        Return
            1d np.array of int labels OR 2D np.array of onehot labels
        '''
        return np.array([LabelGuidedKMeansUtils.to_categorical(y, self._ncats) for y in self._Y]) if onehot else self._Y
    Y = property(get_Y)

class LabelGuidedKMeansUtils:
    @staticmethod
    def get_initial_centroids(X, Y, rand=True):
        '''
        helper function for getting the initial centroids used in KMeans

        Parameters
            X    : np array input data
            Y    : np array of labels for input data
            rand : bool (if true choose a random item, else just the first)
        
        Return
            np.array of initial centroids (array of inputs from X)
        '''
        # if labels are onehot, convert to integers
        Y = Y if len(Y.shape) == 1 else np.array([LabelGuidedKMeansUtils.from_categorical(y) for y in Y])
        initial_centroids = []
        for yuniq in np.unique(Y, axis=0):
            yuniq_idxs = [i for i,y in enumerate(Y) if y == yuniq]
            ic = X[np.random.choice(yuniq_idxs) if rand else 0]
            initial_centroids.append(ic)
        return np.array(initial_centroids)

    @staticmethod
    def find_original_point(region, X_orig, nearest=True):
        '''
        finds the matching original point from a region

        Parameters
            region  : LabelGuidedKMeansRegion object
            X_orig  : np.array, dataset's original inputs (X)
            nearest : dataset's original inputs (X)
        
        Return
            np.array, or None
        '''
        X = region.get_X(sort=True, sortrev=(not nearest))
        X_orig = X_orig.reshape((X_orig.shape[0], -1))
        for x in X:
            if x in X_orig:
                return x
        return None

    @staticmethod
    def find_region(lgkm, x, category=None):
        '''
        finds a given region for a given input

        Parameters
            lgkm     : LabelGuidedKMeans object
            x        : the input
            category : the input's category (optional)
        
        Return
            LabelGuidedKMeansRegion object
        '''
        return next(iter([r for r in lgkm.get_regions(category=category) if distance.euclidean(x, r.centroid) <= r.radius]))
    
    @staticmethod
    def filter_regions(lgkm, modelpath):
        '''
        filters out regions whose centroid is not correctly predicted by the supplied network

        Parameters
            lgkm      : LabekGuidedKMeans object
            modelpath : path to the h5 or pb model
        
        Return
            list of correctly predicted regions
        '''
        regions = lgkm.get_regions()
        model = load_model(modelpath)
        predictions = model.predict([r.centroid for r in regions])
        return [r for i,r in enumerate(regions) if LabelGuidedKMeansUtils.validate_prediction(r.category, predictions[i])]
    
    @staticmethod
    def to_categorical(y, n_cats):
        '''
        converts a given y value to categorical (one-hot)

        Parameters
            y      : integer
            n_cats : number of categories
        
        Return
            np.array (onehot encoded label)
        '''
        return np.array([int(y == i) for i in range(n_cats)])
    
    @staticmethod
    def from_categorical(y):
        '''
        converts a given categorical (one-hot) value to integer

        Parameters
            y : one-hot encoded array
        
        Return
            integer label
        '''
        return np.argmax(y)

    @staticmethod
    def remove_outliers(X, Y, tolerance):
        '''
        removes outliers with abs(zscore(X)) < tolerance from the dataset
        zscore => (X – μ) / σ

        Parameters
            X : np.array of inputs
            Y : np.array of outputs
        
        Return
            np.array(X), np.array(Y)
        '''
        assert X.shape[0] == Y.shape[0], 'X and Y must have same number of rows'
        idxs = np.where((np.abs(stats.zscore(X)) < tolerance).all(axis=1))[0]
        return X[idxs], Y[idxs]
    
    @staticmethod
    def load_dataset(csvfile, n_outputs=5, index_col=0):
        '''
        loads the dataset from csv
        columns in format [index, x0, x1...xN, y0...yN]

        Parameters
            csvfile   : path to csv file
            n_outputs : int, number of outputs in dataset
            index_col : int, column number of index column (pass None if no index column)
        
        Return
            tuple (X, Y) where X and Y are np.array
        '''
        logger.info(f'reading dataset from {csvfile}...')
        # load data from csv
        df = pd.read_csv(csvfile, index_col=index_col)
        # drop duplicate rows
        df = df.drop_duplicates()
        # separate inputs and outputs
        X = df.iloc[:, 0:df.shape[1] - n_outputs].to_numpy()
        Y = df.iloc[:, -n_outputs:].to_numpy()
        # drop X duplicates
        X, idxs = np.unique(X, axis=0, return_index=True)
        Y = Y[idxs]
        return X, Y

    @staticmethod
    def validate_prediction(y, pred):
        '''
        returns true if the prediction equals the expected label
        (note: returns false if the max value appears in multiple outputs (e.g. [3,1,9,4,9]))
        
        Parameters
            y    : expected label (integer or one-hot)
            pred : network's outputs
        '''
        # convert y to integer if categorical is supplied.
        y = y if isinstance(y, (int, np.integer)) else LabelGuidedKMeansUtils.from_categorical(y)
        maxidxs = np.argwhere(pred == np.amax(pred)).reshape(-1)
        return (maxidxs.shape[0] == 1) and maxidxs[0] == y
    
    @staticmethod
    def reduce_classes(Y, metaclasses=[(0,1), (2,), (3,4)]):
        '''
        reduces the original labels to combined 'metalabels'

        Parameters
            Y           : 1D or 2D numpy array of labels
            metaclasses : representation of the desired classes
        Return
            np.array of metalabels

        Example:
            [(0,1), (2,), (3,4)] will combine labels 0/1 and 4/4 into two classes instead of 4
        '''
        Yprime = np.array([LabelGuidedKMeansUtils.from_categorical(y) for y in Y]) if len(Y.shape) == 2 else Y.copy()
        for mc,classes in enumerate(metaclasses):
            for c in classes:
                Yprime[Y==c] = mc
        return Yprime

    @staticmethod
    def print_regions(lgkm, sort=False, sortrev=True):
        '''
        prints all regions
        
        Parameters
            lgkm    : LabelGuidedKMeans object
            sort    : bool (if true, returns in ascending sorted order)
            sortrev : bool (reverses sort order)
        '''
        regions = lgkm.get_regions(sort=sort)
        stringify_region = lambda r: ', '.join([f'{p}={getattr(r, p)}' for p in ('category', 'n', 'radius', 'density')])
        print(f'{len(regions)} regions:\n' + '\n'.join([stringify_region(r) for r in regions]))
    
    @staticmethod
    def print_summary(lgkm, boundaries=[10, 100, 1000], modelpath=''):
        '''
        prints a summary of the LGKMeans region sizes

        Parameters
            lgkm       : LabelGuidedKMeans object
            boundaries : list of integers (specifies which boundaries to print)
            modelpath  : if supplied, centroids will be checked against network
        '''
        regions = lgkm.get_regions()
        lines = [
            '%d regions from %d inputs' % (len(regions), sum([r.n for r in regions])),
            'n == 1: %d' % sum([1 for r in regions if r.n == 1]),
            'n > 1: %d' % sum([1 for r in regions if r.n > 1])
        ]
        lines.extend(['n >= %d: %d' % (n, sum([1 for r in regions if r.n >= n])) for n in boundaries])
        if modelpath:
            filtered = LabelGuidedKMeansUtils.filter_regions(lgkm, modelpath)
            nregions, nfiltered = len(regions), len(filtered)
            lines.append('%d of %d centroids are valid (%f)' % (nfiltered, nregions, 100*nfiltered/nregions))
        summary = '\n'.join(lines)
        print(summary)
    
    @staticmethod
    def serialize_regions(lgkm, sort=False, sortrev=True, include_data=False):
        '''
        serializes regions from LabelGuidedKMeans object into a list of dictionaries

        Parameters
            lgkm         : LabelGuidedKMeans object
            sort         : bool (sorts in ascending order by radius & density)
            sortrev      : bool (reverses order of sort)
            include_data : bool (if true, X and Y are included)
        
        Return
            list of dictionaries
        '''
        props = ['centroid', 'category', 'density', 'radius', 'n', *(['X', 'Y'] if include_data else [])]
        return [{getattr(r, p) for p in props} for r in lgkm.get_regions(sort=sort, sortrev=sortrev)]
    
    @staticmethod
    def save(lgkm, outpath='./lgkm.p', serialize=False, include_data=False):
        '''
        saves a LabelGuidedKMeans object (or list of dicts) to a pickle

        Parameters
            lgkm         : LabelGuidedKMeans object
            outpath      : the output filepath
            serialized   : bool (if true, serializes lgkm to list of dicts)
            include_data : bool (if true, saves X and Y - much larger file size)
        '''
        data = lgkm if not serialize else LabelGuidedKMeansUtils.serialize_regions(lgkm, include_data=include_data)
        create_dirpath(os.path.dirname(os.path.abspath(outpath)))
        pickle.dump(data, open(outpath, 'wb'))
        print(f'saved to {outpath}')
    
    @staticmethod
    def load(path):
        '''
        loads a pickled object

        Parameters
            path : string (path to pickle)

        Return
            LabelGuidedKMeans object (or list of dictionaries)
        '''
        lgkm = pickle.load(open(path, 'rb'))
        return lgkm
    
    @staticmethod
    def save_regions_csv(lgkm, sort=True, sortrev=True, outpath='./lgkm.csv'):
        '''
        saves a LabelGuidedKMeans object's regions to CSV

        Parameters
            lgkm    : LabelGuidedKMeans object
            sort    : bool (sorts in ascending order by radius & density)
            sortrev : bool (reverses order of sort)
            outpath : the output filepath
        '''
        regions = lgkm.get_regions(sort=sort, sortrev=sortrev)
        n_features = regions[0].centroid.shape[0]
        header = ','.join([f'cx{i}' for i in range(n_features)] + ['radius', 'n', 'density', 'category'])
        rows = []
        for r in regions:
            rows.append(','.join([str(x) for x in r.centroid] + [str(v) for v in (r.radius, r.n, r.density, r.category)]))
        create_dirpath(outpath)
        with open(outpath, 'w') as f:
            f.writelines('\n'.join([header] + rows))
            print(f'saved regions to {outpath}')
