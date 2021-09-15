import os, sys
import numpy as np
from scriptify import scriptify
from data_preprocessing import get_data
from clustering import LabelGuidedKMeans, LabelGuidedKMeansUtils
from robustness import verify_regions

def print_marabou_query(lbs, ubs, correct_label, file_path):
    incorrect_labels = list(filter(lambda l: l != correct_label, [0,1,2,3,4]))
    names = ['a','b','c','d']

    for i in range(0, len(incorrect_labels)):
        fpath = f'{file_path}{names[i]}.txt'
        with open(fpath, 'w') as f1:
            for j in range(0, lbs.size):
                f1.write(f'x{j} >= {lbs[j]}\n')
                f1.write(f'x{j} <= {ubs[j]}\n')
            f1.write(f'+y{incorrect_labels[i]} -y{correct_label} >= 0')

if __name__ == '__main__':

    @scriptify
    def script(marabou_path='../../Marabou',
               experiment='safescad',
               conf_name='default'):
        sys.path.append(os.path.abspath(marabou_path))

        data_dir = f'./data/{experiment}/{conf_name}'
        model_dir = f'./models/{experiment}/{conf_name}'
        robust_regions_dir = f'./robust_regions/{experiment}/{conf_name}'
        marabou_query_dir = f'./marabou_queries/{experiment}/{conf_name}'

        if not os.path.exists(robust_regions_dir):
            os.makedirs(robust_regions_dir)
        if not os.path.exists(marabou_query_dir):
            os.makedirs(marabou_query_dir)

        X_train_enc, y_train_enc, _, _, _, _, _, _ = get_data(experiment, None, data_dir)

        # drop duplicates
        X_train_enc, idxs = np.unique(X_train_enc, axis=0, return_index=True)
        y_train_enc = y_train_enc[idxs]

        X, Y = LabelGuidedKMeansUtils.remove_outliers(X_train_enc, y_train_enc, tolerance=10)

        print(f'dropped {X_train_enc.shape[0] - X.shape[0]} outliers from {X_train_enc.shape[0]} inputs')
        print(X.shape, Y.shape)

        # ## Generate Label Guided K-Means Regions
        init = 150  # Initial number of clusters for k-means. Verified experimentally for SafeScadthrough elbow method.

        # init_centroid dictates the mechanism which the algorithm should follow for initial centroid generation.
        # The options are 'rand','mean','first','k-means++'. Check clustering.py for further details.

        lgkmc = LabelGuidedKMeans().fit(X, Y, init_nclusters=init, init_centroid='mean')
        LabelGuidedKMeansUtils.print_summary(lgkmc)

        # save regions
        LabelGuidedKMeansUtils.save(lgkmc, outpath=f'{robust_regions_dir}/lgkm.p')

        # ## Verify Regions

        # load the regions
        lgkmc = LabelGuidedKMeansUtils.load(f'{robust_regions_dir}/lgkm.p')
        regions = lgkmc.get_regions(sort=True)

        # load model
        model_path = f'{model_dir}/model_base.nnet'

        # verify the regions (they will be automatically saved to out_dir)
        # note: you'll need the Gurobi Optimizer installed (https://github.com/NeuralNetworkVerification/Marabou#use-lp-relaxation)
        #       if solveWithMILP=true

        regions = verify_regions(
            network_path=model_path,
            experiment=experiment,
            network_options=dict(),
            marabou_options=dict(solveWithMILP=False, milpTightening='none'),
            regions=regions,
            e_min=0.0001,
            e_max=10.0,
            e_interval=0.00001,
            radius_padding=0,
            timeout=0,
            marabou_verbosity=0,
            allowed_misclassifications=None,
            out_dir=f'{robust_regions_dir}'
        )

        num_features = X_train_enc.shape[1]
        # print marabou queries
        for index, row in regions.iterrows():
            l2_radii = row['radius']
            linf_radii = row['epsilon']
            centroids_row = []
            for i in range(0, num_features):
                centroids_row.append(row[f'cx{i}'])
            centroids = np.array(centroids_row)
            category = row['category']
            lbs = centroids - linf_radii
            ubs = centroids + linf_radii
            fpath = f'{marabou_query_dir}/q{index}'
            print_marabou_query(lbs, ubs, category, fpath)




