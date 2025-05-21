# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:41:29 2025

@author: theob
"""

import numpy as np


class DecisionTreeRegressor:
    
    def __init__(self):
        
        self.tree = None

    def mse(self, y):
        
        return np.var(y) * len(y)

    def best_split(self, X, y):
        
        m, n = X.shape
        best_mse = float('inf')
        split_idx, split_val = None, None

        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            
            for threshold in thresholds:
                left_idx = X[:, feature_index] <= threshold
                right_idx = X[:, feature_index] > threshold

                if len(y[left_idx]) < 1 or len(y[right_idx]) < 1:
                    continue

                mse_left = self.mse(y[left_idx])
                mse_right = self.mse(y[right_idx])
                mse_total = mse_left + mse_right

                if mse_total < best_mse:
                    best_mse = mse_total
                    split_idx = feature_index
                    split_val = threshold
        
        return split_idx, split_val


#recursive binary splitting

    def build_tree(self, X, y, observations=5):
        
        k = 2
        split_idx, split_val = self.best_split( X, y)
        arbre = dict()
        arbre["feature_index"] = split_idx
        arbre["threshold"]  = split_val
        #liste_threshold_and_feature_index = [[split_idx, split_val]]
        left_idx = X[:, split_idx] <= split_val
        right_idx = X[:, split_idx] > split_val
        X1 = X[left_idx]
        X2 = X[right_idx]
        y1 = y[left_idx]
        y2 = y[right_idx]
        arbre["left"] = y1
        arbre["right"] = y2
        regions_X = [ X1 , X2 ]
        regions_y = [ y1 , y2 ]
        profondeur_liste_regions = [ 1 , 1 ]
        j1 = 0
        
        while j1 <k:
            
            if len(regions_X[j1]) <= observations:
                
                j1 += 1
                
            else:
                
                best_mse = float('inf')
                j3 = 0
                feature_index,threshold = None, None
                
                
                
                for j2 in range(k): 
                    if len(regions_X[j2])<=observations:
                        
                        continue
                    split_idx, split_val = self.best_split(regions_X[j2], regions_y[j2])
                    left_idx = regions_X[j2][:, split_idx] <= split_val
                    right_idx = regions_X[j2][:, split_idx] > split_val
                    y1 = regions_y[j2][left_idx]
                    y2 = regions_y[j2][right_idx]
                    mse_left = self.mse(y1)
                    mse_right = self.mse(y2)
                    mse_total = mse_left + mse_right
                    
                    if mse_total < best_mse:
                        
                        best_mse = mse_total
                        j3 = j2
                        feature_index,threshold = split_idx, split_val
                
                left_idx = regions_X[j3][:, feature_index] <= threshold
                right_idx = regions_X[j3][:, feature_index] > threshold
                X1 = regions_X[j3][left_idx]
                X2 = regions_X[j3][right_idx]
                yy1 = regions_y[j3][left_idx]
                yy2 = regions_y[j3][right_idx]
                qq = profondeur_liste_regions.pop(j3)
                
                def inserer_dans_arbre(arbre,value_to_split, profondeur,feature_index,threshold, valeur_left,valeur_right):
                    
                    
                    if profondeur == 2 :
                        if np.array_equal(arbre["left"],value_to_split):

                            arbre["left"] = {"feature_index" : feature_index,
                                             "threshold" : threshold,
                                             "left" : valeur_left,
                                             "right" : valeur_right}
                            return arbre
                        
                        elif np.array_equal(arbre["right"],value_to_split):

                            arbre["right"] = {"feature_index" : feature_index,
                                             "threshold" : threshold,
                                             "left" : valeur_left,
                                             "right" : valeur_right}
                            return arbre
                        
                        else:

                            return arbre


                    
                    elif profondeur != 2 :
                        
                        if isinstance(arbre["left"],dict):
                            arbre1 = inserer_dans_arbre(arbre["left"],value_to_split, profondeur-1,feature_index,threshold, valeur_left,valeur_right)
                            
                            if arbre1 != arbre["left"]:
                                arbre["left"] = arbre1
                                return arbre
                            
                            else :
                                if isinstance(arbre["right"],dict):
                                    arbre["right"] = inserer_dans_arbre(arbre["right"],value_to_split, profondeur-1,feature_index,threshold, valeur_left,valeur_right)
                                    return arbre
                                else :
                                    return arbre
                                
                        else :
                            if isinstance(arbre["right"],dict):
                                arbre["right"] = inserer_dans_arbre(arbre["right"],value_to_split, profondeur-1,feature_index,threshold, valeur_left,valeur_right)
                                return arbre
                            else:
                                return arbre
                        
                
                arbre = inserer_dans_arbre(arbre,regions_y[j3], qq+1,feature_index,threshold, yy1,yy2)
                
                regions_X.pop(j3)
                regions_y.pop(j3)
                regions_X.insert(j3,X2)
                profondeur_liste_regions.insert(j3,qq+1)
                regions_X.insert(j3,X1)
                profondeur_liste_regions.insert(j3,qq+1)
                regions_y.insert(j3,yy2)
                regions_y.insert(j3,yy1)

                
                     

                k += 1
                j1 = 0


        return arbre,profondeur_liste_regions
             
        #code à compléter qui renvoie l'arbre de décision
                

    def fit(self, X_train, y_train,observations =5):
        
        self.tree,_ = self.build_tree(np.array(X_train), np.array(y_train),observations)

#fonction de prédiction des prix sur le set de test à écrire

    def predict_sample(self, x, tree):
        
        if not isinstance(tree, dict):
            
            return np.mean(tree)
        
        else:

            if x[tree['feature_index']] <= tree['threshold']:
            
                return self.predict_sample(x, tree['left'])
            
            else:
            
                return self.predict_sample(x, tree['right'])

    def predict(self, X_train):
        
        return np.array([self.predict_sample(x, self.tree) for x in X_train])
    
    def pruning(self,tree,alpha):
        pass

    def mean_square_error(self,y_test, y_predict_test):  
        somme = 0
        for k in range(len(y_test)):
            somme += (y_test[k]-y_predict_test[k])*(y_test[k]-y_predict_test[k])
        return somme/len(y_test)
    
    def metric_error_for_pruning(self,y_test, y_predict_test,alpha):
        pass
"""# Données jouet
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([1, 1.5, 2, 2.5, 3, 3.5])

# Modèle
tree = DecisionTreeRegressor()
tree.fit(X, y)

# Prédictions
predictions = tree.predict(X)
print(predictions)"""


#essai1
"""import numpy as np
thresholds = np.unique([4,2,8,5,9,14,3,7])
X = np.array([[4,3,7,52,14], [2,8,15,1,3], [8,4,2,9,10], [6,2,18,24,5], [9,7,13,11,20], [14,12,4,8,12],[3,13,23,53,28],[7,77,52,63,33]])
y = np.array([1, 1.5, 2, 2.5, 3, 3.5,5,7.8])
print(thresholds)
for threshold in thresholds:
    left_idx = X[:,0]<= threshold
    print(left_idx)
    print(y[left_idx])
    right_idx = X[:,0]> threshold
    print(right_idx)
    print(y[right_idx])"""
    


#essai2

"""
X = np.array([[4,3,7,52,14], [2,8,15,1,3], [8,4,2,9,10], [6,2,18,24,5], [9,7,13,11,20], [14,12,4,8,12],[3,13,23,53,28],[7,77,52,63,33]])
y = np.array([1, 1.5, 2, 2.5, 3, 3.5,5,7.8])

dd = DecisionTreeRegressor()
print(dd.build_tree_step_two(X, y,5))"""

#essai3

import pandas as pd 
df = pd.read_csv("C:/Users/theob/Downloads/fake_reg.csv")
df = df_shuffled = df.iloc[np.random.permutation(len(df))]
X = df[['feature1','feature2']].values
#,'feature2'
y = df['price'].values

X_train = X[0:500][:]
y_train = y[0:500]

X_test = X[500:][:]
y_test = y[500:]
from time import time
t1 = time()
dd = DecisionTreeRegressor()
dd.fit(X_train,y_train,5)
#print(dd.tree)
y_train_predict = dd.predict(X_train)

y_predict = dd.predict(X_test)
t2 = time()
mse_train = np.sqrt(dd.mean_square_error(y_train, y_train_predict))
mse_test = np.sqrt(dd.mean_square_error(y_test, y_predict))


print(mse_train)
print(mse_test)
print(t2-t1)


