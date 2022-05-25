import pandas as pd
import numpy as np
import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple
import copy
from sklearn.datasets import make_circles

import sys
sys.path.append('../../utils')
from common_utils import set_seed

@dataclass
class ArtifactFeatures():
    ratio: float
    factor: float
    division: float
    shuffle: float
    means: Tuple[int]
    data_size: int = 20000
    nclasses: int = 2
    std: float = 0.1
    rand_m: float = 0
    rand_s: float = 1
    
    def introduce_bias_feature(self, class_gauss, ntotal):
        n = ntotal//self.nclasses  # number of examples per class.
        features = np.random.rand(ntotal)*self.rand_s - self.rand_m
        classes = range(self.nclasses)
        for i in classes:
            heur_feats = list(np.random.normal(self.means[i], self.std, len(class_gauss[i][0])))
            anti_feats = list(np.random.normal(self.means[not i], self.std, len(class_gauss[i][1])))

            for cf, cind in zip(heur_feats+anti_feats, class_gauss[i][0]+class_gauss[i][1]):
                features[cind] = cf
        assert len(features) == ntotal
        return np.expand_dims(features, axis=1)

    def introduce_anti_feature(self, class_gauss, ntotal):
        n = ntotal//self.nclasses  # number of examples per class.
        features = np.random.rand(ntotal)*self.rand_s - self.rand_m
        classes = range(self.nclasses)
        for i in classes:
            heur_feats = list(np.random.normal(self.means[not i], self.std, len(class_gauss[i][0])))
            anti_feats = list(np.random.normal(self.means[i], self.std, len(class_gauss[i][1])))

            for cf, cind in zip(heur_feats+anti_feats, class_gauss[i][0]+class_gauss[i][1]):
                features[cind] = cf
        assert len(features) == ntotal
        return np.expand_dims(features, axis=1)

    def introduce_no_bias_feature(self, ntotal):
        n = ntotal//self.nclasses  # number of examples per class.
        features = np.random.rand(ntotal)*self.rand_s - self.rand_m
        return np.expand_dims(features, axis=1)
    
    def introduce_aligned_feature(self, ntotal):
        n = ntotal//self.nclasses  # number of examples per class.
        features = np.random.rand(ntotal)*self.rand_s - self.rand_m
        features_b, features_nb = copy.deepcopy(features), copy.deepcopy(features)
        classes = range(self.nclasses)
        for i in classes:
            heur_feats = list(np.random.normal(self.means[i], self.std, n))
            anti_feats = list(np.random.normal(self.means[not i], self.std, n))

            for cf, cind in zip(heur_feats, range(n*i, n*(i+1))):
                features_b[cind] = cf
            for cf, cind in zip(anti_feats, range(n*i, n*(i+1))):
                features_nb[cind] = cf

        assert len(features) == len(features_b) == len(features_nb) == ntotal
        return np.expand_dims(features, axis=1), np.expand_dims(features_b, axis=1), np.expand_dims(features_nb, axis=1)

    def perform_label_shuffling(self, labels):
        to_assign = np.random.rand(len(labels))
        bin_ranges = np.linspace(0, 1*self.shuffle, num=2+1)
        labels_ = np.digitize(to_assign, bins=bin_ranges)-1
        labels_[labels_ == 2] = np.array(labels)[labels_ == 2]
        print(f'Perc flips: {np.sum(labels_ != np.array(labels)).item()/len(labels)}')

        return labels_

    def get_dataset(self, artifact="bias", aligned=False, seed=42, evaluation=False):
        self.artifact, self.aligned = artifact, aligned
        set_seed(seed)
        if aligned:
            return self.return_aligned_dataset()
        return self.return_dataset(seed, evaluation)
    
    def return_dataset(self, seed=42, evaluation=False):
        half = self.data_size//self.nclasses
        art_circles = {}

        if self.artifact == "bias" or self.artifact == "anti":
            class_gauss = [[None, None], [None, None]]
            artifacts_class0 = np.random.choice(range(half), int(self.ratio*half), replace=False)
            class_gauss[0][0] = list(np.random.choice(artifacts_class0, math.ceil(self.ratio*half*self.division), replace=False))
            class_gauss[0][1] = list(set(artifacts_class0).difference(set(class_gauss[0][0])))
            assert len(set(class_gauss[0][0])) + len(class_gauss[0][1]) == int(self.ratio*half)
            
            artifacts_class1 = np.random.choice(range(half, self.data_size), int(self.ratio*half), replace=False)
            class_gauss[1][0] = list(np.random.choice(artifacts_class1, math.ceil(self.ratio*half*self.division), replace=False))
            class_gauss[1][1] = list(set(artifacts_class1).difference(set(class_gauss[1][0])))
            assert len(set(class_gauss[1][1])) + len(class_gauss[1][0]) == int(self.ratio*half)
        
        base_dir = f'models/factor={self.factor}_division={self.division}_shuffle={self.shuffle}_size={self.data_size}'
        artifacted_circles_file = f'{base_dir}/dataset.csv' if not evaluation else f'{base_dir}/eval_set.csv'

        if os.path.exists(artifacted_circles_file):
            print(f"Reading {self.division} artifacted circles data from {artifacted_circles_file}.")
            art_circles = pd.read_csv(artifacted_circles_file)
        else:
            os.makedirs(base_dir, exist_ok=True)
            if self.factor == 0.9:
                circles_file = f"data/circles/flipped_circles_sep_factor_{self.factor}_{self.data_size}_seed={seed}.csv"
            else:
                circles_file = f"data/circles/circles_sep_factor_{self.factor}_{self.data_size}_seed={seed}.csv"
            
            print(f"Creating {self.division} artifacts for circles data in {circles_file}.")
            if os.path.exists(circles_file):
                art_circles = pd.read_csv(circles_file)
            else:
                x, y = (make_circles(n_samples=self.data_size,
                                     shuffle=False, 
                                     noise=0.2, 
                                     random_state=seed, 
                                     factor=self.factor))
                df = pd.DataFrame(x)
                df['label'] = y
                df.rename(columns={0:"x", 1:"y"}, inplace=True)

                art_circles = df
                df.to_csv(circles_file, index_label=circles_file)

            ntotal = len(art_circles)
            
            if self.artifact == "bias":
                cheats = self.introduce_bias_feature(class_gauss, ntotal)
            elif self.artifact == "anti":
                cheats = self.introduce_anti_feature(class_gauss, ntotal)
            elif self.artifact == "none":
                cheats = self.introduce_no_bias_feature(ntotal)

            art_circles['c1'] = cheats

            labels = art_circles['label']
            labels_ = self.perform_label_shuffling(labels)
            art_circles['label'] = labels_

            art_circles.to_csv(artifacted_circles_file)
            print(f"Written to {artifacted_circles_file}")

        return art_circles

    def return_aligned_dataset(self):
        art_circles_bias, art_circles_anti, art_circles_none = {}, {}, {}
        
        none_circles_dir = f'data/circles/aligned_none/'
        biased_circles_dir = f'data/circles/aligned_bias/'
        anti_bias_circles_dir = f'data/circles/aligned_anti/'
        for dir_ in [none_circles_dir, biased_circles_dir, anti_bias_circles_dir]:
            os.makedirs(dir_, exist_ok=True)

        common_suffix = f'factor_{self.factor}_means_{self.means[0]}_{self.means[1]}_{self.data_size}.csv'
        none_circles_file, biased_circles_file, anti_bias_circles_file = \
        none_circles_dir+common_suffix, biased_circles_dir+common_suffix, anti_bias_circles_dir+common_suffix

        if os.path.exists(none_circles_file) and os.path.exists(biased_circles_file) and os.path.exists(anti_bias_circles_file):
            art_circles_bias = pd.read_csv(biased_circles_file)
            art_circles_anti = pd.read_csv(anti_bias_circles_file)
            art_circles_none = pd.read_csv(none_circles_file)
        else:
            circles_file = f"data/circles/circles_sep_factor_{self.factor}_{self.data_size}_seed=42.csv"

            art_circles_bias = pd.read_csv(circles_file)
            art_circles_anti = pd.read_csv(circles_file)
            art_circles_none = pd.read_csv(circles_file)

            ntotal = len(art_circles_bias)
            no_bias, biased, anti_bias = self.introduce_aligned_feature(ntotal)

            art_circles_bias['c1'] = biased
            art_circles_anti['c1'] = anti_bias
            art_circles_none['c1'] = no_bias

            art_circles_bias.to_csv(biased_circles_file)
            art_circles_anti.to_csv(anti_bias_circles_file)
            art_circles_none.to_csv(none_circles_file)

        return art_circles_bias, art_circles_anti, art_circles_none