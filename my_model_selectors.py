import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_num_components = self.min_n_components
        minBICscore = np.inf
        for n_component in range(self.min_n_components,self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=1000,
                random_state=self.random_state, verbose=False)
                model_fitted = model.fit(self.X, self.lengths)
                logL = model_fitted.score(self.X, self.lengths)
                param = n_component**2 + 2 * n_component* len(self.X[0]) -1
                BICScore = -2 * logL + math.log(len(self.X)) * param
                if BICScore < minBICscore:
                    minBICscore = BICScore
                    best_num_components = n_component
            except:
                pass
        model = self.base_model(best_num_components)
        return model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_num_components = self.min_n_components
        bestDIC = -np.inf
        for n_component in range(self.min_n_components,self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=1000,
                random_state=self.random_state, verbose=False)
                model_fitted = model.fit(self.X, self.lengths)
                selfScore = model_fitted.score(self.X, self.lengths)
                scores = [model_fitted.score(self.hwords[word][0], self.hwords[word][1]) for word in self.words]
                DIC = selfScore - np.mean(scores)
                if DIC > bestDIC:
                    best_num_components = n_component
                    bestDIC = DIC
            except:
                pass
        model = self.base_model(best_num_components)
        return model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        maxLogLikelihood = -np.inf
        best_num_components = self.min_n_components
        for n_component in range(self.min_n_components,self.max_n_components+1):
            n_splits = min(len(self.lengths), 3)
            if len(self.sequences) < n_splits:
                break
            split_method = KFold(n_splits=n_splits)
            model = GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=1000,
            random_state=self.random_state, verbose=False)
            totalLogL = 0
            folds = 0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                train_data = combine_sequences(cv_train_idx, self.sequences)
                test_data = combine_sequences(cv_test_idx,self.sequences)
                try:
                    model_fitted = model.fit(train_data[0], train_data[1])
                    totalLogL += model_fitted.score(test_data[0],test_data[1])
                    folds +=1
                except:
                    pass
            if totalLogL == 0:
                logL = -np.inf
            else:
                logL = totalLogL / folds
            if logL > maxLogLikelihood:
                maxLogLikelihood = logL
                best_num_components = n_component
        model = self.base_model(best_num_components)
        return model