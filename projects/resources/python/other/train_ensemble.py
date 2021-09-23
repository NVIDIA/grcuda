# Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier


def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=1).reshape(X.shape[0], 1)


def logsumexp(X):
    return np.log(np.sum(np.exp(X)))


def naive_bayes_predict(X, feature_log_prob, log_class_prior):
    num_features = X.shape[1]
    num_classes = len(log_class_prior)
    jll = np.zeros((X.shape[0], num_classes))
    jll += log_class_prior

    # K1
    for i in range(num_features):
        for j in range(num_classes):  # col Y.T, i.e. feature_log_prob
            for q in range(X.shape[0]):  # row X
                jll[q, j] += X[q, i] * feature_log_prob[j, i]
    # Same as 
    # jll = X.dot(feature_log_prob.T) + log_class_prior
    
    # K2
    amax = np.amax(jll, axis=1)
    
    # K3
    # l = np.zeros(X.shape[0])
    # for i in range(X.shape[0]):
    #     logsum = 0
    #     for j in range(num_classes):
    #         logsum += np.exp(jll[i, j] - amax[i])
    #     l[i] = np.log(logsum) + amax[i]
    l = logsumexp(jll - np.atleast_2d(amax).T) + amax
        
    # K4
    for q in range(X.shape[0]):
        for j in range(num_classes):
            jll[q, j] = np.exp(jll[q, j] - l[q])
    
    return jll 

def normalize(X):
   return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def ridge_pred(X, coef, intercept):
    return np.dot(X, coef.T) + intercept


#%%

if __name__ == "__main__":
    

    rng = np.random.RandomState(1)
    
    num_features = 1000
    max_occurrence_of_ngram = 10
    num_classes = 5
    
    X = rng.randint(max_occurrence_of_ngram, size=(6000, num_features))
    
    y = rng.randint(num_classes, size=6000)
    
    
    #%% Training;
    
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X, y)
    
    ridge = RidgeClassifier(random_state=rng)
    ridge.fit(X, y)
    
    
    #%% Testing;
    
    # Create some random inputs;
    
    num_test_docs = 100
    X_test = rng.randint(max_occurrence_of_ngram, size=(num_test_docs, num_features))
    
    nb_scores = naive_bayes.predict_proba(X_test)
    print(naive_bayes.predict(X_test))
    ridge_scores = ridge.decision_function(X_test)
    print(ridge.predict(X_test))
        
    print(np.argmax(softmax(nb_scores) + softmax(ridge_scores), axis=1))
    
    
    #%% Testing, using hand-made functions;
    
    nb_res_2 = naive_bayes_predict(X_test, naive_bayes.feature_log_prob_, naive_bayes.class_log_prior_)
    print(np.argmax(nb_res_2, axis=1))

    ridge_pred_2 = ridge_pred(X_test, ridge.coef_, ridge.intercept_)
    print(np.argmax(ridge_pred_2, axis=1))

    print(np.argmax(softmax(nb_res_2) + softmax(ridge_pred_2), axis=1)) 
    
    
    #%% Store matrices used in predicting results;
    np.savetxt("data/nb_feat_log_prob.csv", naive_bayes.feature_log_prob_, delimiter=",")
    np.savetxt("data/nb_class_log_prior.csv", naive_bayes.class_log_prior_, delimiter=",")
    np.savetxt("data/ridge_coeff.csv", ridge.coef_, delimiter=",")
    np.savetxt("data/ridge_intercept.csv", ridge.intercept_, delimiter=",")
    
    