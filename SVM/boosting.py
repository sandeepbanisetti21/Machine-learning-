import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod
from operator import itemgetter


class Boosting(Classifier):
    # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs  # set of weak classifiers to be considered
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T

        self.clfs_picked = []  # list of classifiers h_t for t=0,...,T-1
        self.betas = []  # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:
        '''
        Inputs:
        - features: the features of all test examples

        Returns:
        - the prediction (-1 or +1) for each example (in a list)
        '''
        ########################################################
        # TODO: implement "predict"
        ########################################################
        preds = []
        for b_t, h_t in zip(self.betas, self.clfs_picked):
            preds.append(np.multiply(b_t, h_t.predict(features)))

        preds = np.sign(np.sum(preds, axis=0))
        preds[preds == 0] = -1  # accounting for `np.sign(0) == 0`, update as per @720

        return preds.tolist()


class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return

    def train(self, features: List[List[float]], labels: List[int]):
        '''
        Inputs:
        - features: the features of all examples
        - labels: the label of all examples

        Require:
        - store what you learn in self.clfs_picked and self.betas
        '''

    ############################################################
    # TODO: implement "train"
    ############################################################
        # pre-convert labels to a numpy array for better performance
        labels = np.array(labels)

        # initialize w_t - step (1)
        w_t = np.full(len(features), 1 / len(features))
        for t in range(self.T):
            # find h_t, e_t - step (3, 4)
            h_e = []
            for h in self.clfs:
                h_e.append(
                    (h, (w_t * (labels != np.array(h.predict(features)))).sum())
                )
            h_t, e_t = min(h_e, key=itemgetter(-1))
            # compute b_t - step (5)
            b_t = .5 * np.log((1 - e_t) / e_t) if e_t != 0 else 0
            # update w_t+1 - step (6)
            w_t = np.where(
                labels == np.array(h_t.predict(features)),
                w_t * np.exp(-b_t),
                w_t * np.exp(b_t)
            )
            # normalize w_t+1 - step (7)
            w_t /= np.sum(w_t)

            # store h_t and b_t
            self.clfs_picked.append(h_t)
            self.betas.append(b_t)

def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
