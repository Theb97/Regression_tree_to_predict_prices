# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:41:29 2025

@author: theob
"""

import numpy as np
import copy
import time

class DecisionTreeRegressor:
    
    def __init__(self):
        
        self.tree = None
        self.terminal_nodes = 0
        self.profondeur_liste = []
        
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


        return arbre,k,profondeur_liste_regions
             
        #code qui renvoie l'arbre de décision
                

    def fit(self, X_train, y_train,observations =5):
        
        self.tree, self.terminal_nodes, self.profondeur_liste = self.build_tree(np.array(X_train), np.array(y_train),observations)

#fonction de prédiction des prix sur le set de test

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
    
    def pruning(self,tree,X_train,y_train,alpha,profondeur=0): 
            
        def fusionner_branches(arbre,y):
            
            if not isinstance(arbre["left"],dict) and not isinstance(arbre["right"],dict):
                y -=1
                return np.concatenate([arbre["left"], arbre["right"]]),y
            
            elif isinstance(arbre["left"],dict) and not isinstance(arbre["right"],dict):
                arleft,y = fusionner_branches(arbre["left"],y)
                y -=1
                return np.concatenate([arleft, arbre["right"]]),y
            
            elif not isinstance(arbre["left"],dict) and isinstance(arbre["right"],dict):
                y -=1
                arright,y = fusionner_branches(arbre["right"],y)
                return np.concatenate([arbre["left"], arright]),y
            
            else :
                arleft,y = fusionner_branches(arbre["left"],y)
                arright,y = fusionner_branches(arbre["right"],y)
                y-=1
                return np.concatenate([arleft, arright]),y
        
        def descente_et_fusion(arbre,x,k,y):
 
            if k == x:

                return arbre, x, y
            
            else:
                
                if not isinstance(arbre["left"],dict) and not isinstance(arbre["right"],dict):
                    if k == x:
                        return arbre,x,y
                    k+=1
                    if k == x:
                       
                        arbre,y = fusionner_branches(arbre,y)
                        
                        return arbre,x,y
                    k+=1
                    if k == x:
                       
                        arbre,y = fusionner_branches(arbre,y)
                        
                        return arbre,x,y
                    else :
                        
                        return arbre,k,y
                elif not isinstance(arbre["left"],dict) and isinstance(arbre["right"],dict):
                    if k == x:
                        
                        return arbre,x, y 
                    k += 1 
                    if k == x:
                        
                        arbre,y = fusionner_branches(arbre,y)
                        
                        return arbre, x,y
                    
                    arbre2,k,z = descente_et_fusion(arbre["right"],x,k,y)
                   
                    if k == x:
                        arbre["right"]=arbre2
                        
                        return arbre,x,z
                    else:
                        return arbre,k,y
                elif isinstance(arbre["left"],dict) and not isinstance(arbre["right"],dict):
                    
                    if k == x:
                       
                        return arbre,x
                    
                    arbre2,k,z = descente_et_fusion(arbre["left"],x,k,y)
                    
                    if k == x:
                        arbre["left"]=arbre2
                        
                        return arbre,x,z
                    else:
                        k += 1 
                        if k == x:
                            arbre,y = fusionner_branches(arbre,y)
                            return arbre,x,y
                        else:
                            return arbre,k,y
                            
                        
                elif isinstance(arbre["left"],dict) and isinstance(arbre["right"],dict):
                    
                    if k == x:
                        return arbre,x,y
                    arbre2,k,z = descente_et_fusion(arbre["left"],x,k,y)
                    if k == x:
                        arbre["left"] = arbre2
                        return arbre,x,z
                    else:
                        arbre3,k,z = descente_et_fusion(arbre["right"],x,k,y)
                        if k == x:
                            arbre["right"] = arbre3
                            return arbre,x,z
                        else:
                            return arbre,k,y
                        
                        
        
            

        
        self.tree = tree
        erreur = self.metric_error_for_pruning(y_train, self.predict(X_train), alpha)
        k = -1
        if self.terminal_nodes <= 2:
            print(0.1)
            return self.tree,self.terminal_nodes
        
        else:
            ddd = DecisionTreeRegressor()
            tree = copy.deepcopy(self.tree)
            terminal_nodes = copy.deepcopy(self.terminal_nodes)
            qqq = 0
            for x in np.random.permutation(terminal_nodes):
                
                ddd.tree = copy.deepcopy(tree)
                _,_,ddd.terminal_nodes = descente_et_fusion(ddd.tree,x,k,terminal_nodes)
                y_predict_train = ddd.predict(X_train)
                if erreur > ddd.metric_error_for_pruning(y_train, y_predict_train, alpha):
                    self.terminal_nodes = ddd.terminal_nodes
                    self.tree, self.terminal_nodes = self.pruning(ddd.tree,X_train,y_train,alpha,profondeur)
                    
                    erreur = self.metric_error_for_pruning(y_train, self.predict(X_train), alpha)
                
                elif erreur > ddd.mean_square_error(y_train, y_predict_train) and profondeur==0:
                    yyyy = np.random.randint(0,ddd.terminal_nodes)
                    ddd.tree,_, ddd.terminal_nodes = descente_et_fusion(ddd.tree,yyyy,k,ddd.terminal_nodes)
                    erreur2 = ddd.metric_error_for_pruning(y_train, ddd.predict(X_train), alpha)
                    if erreur > erreur2:
                        print("a")
                        self.tree, self.terminal_nodes = ddd.pruning(ddd.tree,X_train,y_train,alpha,profondeur+1)
                        
                        erreur = self.metric_error_for_pruning(y_train, self.predict(X_train), alpha)
                    else:
                        continue 
               
                else:
                    continue

            return self.tree,self.terminal_nodes

                        
                        
            
                        
                        
                

                        
                    
                    
                    

            
            

    def mean_square_error(self,y_test, y_predict_test):  
        somme = 0
        for k in range(len(y_test)):
            somme += (y_test[k]-y_predict_test[k])*(y_test[k]-y_predict_test[k])
        return somme/len(y_test)
    
    def metric_error_for_pruning(self,y_test, y_predict_test,alpha):
        
        erreur1 = self.mean_square_error(y_test, y_predict_test)
        erreur2 = self.terminal_nodes
        return erreur1 + alpha*erreur2
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


#print(mse_train)
#print(mse_test)
#print(t2-t1)

#essai4
mon_dictionnaire = {'feature_index': 1, 'threshold': 1000.0651197080786, 'left': {'feature_index': 0, 'threshold': 999.67125607388, 'left': {'feature_index': 1, 'threshold': 999.04709011227, 'left': np.array([338.0902386 , 381.07367881, 343.73541799, 375.62302356,
       320.22783441]), 'right': np.array([458.52789767, 454.32901321, 437.90401436, 417.54122635,
       444.67970846])}, 'right': {'feature_index': 1, 'threshold': 999.7117705391285, 'left': {'feature_index': 1, 'threshold': 998.8885109601988, 'left': np.array([453.9716661 , 451.79140134, 446.21230389]), 'right': {'feature_index': 0, 'threshold': 1000.4595490022132, 'left': np.array([475.37241721, 481.20614899, 484.229812  , 466.49020065]), 'right': np.array([498.6284209 , 508.19978205, 512.04904895])}}, 'right': np.array([531.48237221, 510.88504979, 531.66193891, 559.96108847])}}, 'right': {'feature_index': 1, 'threshold': 1000.7544434659072, 'left': {'feature_index': 0, 'threshold': 999.2524657510324, 'left': np.array([491.03773416, 501.09520221, 521.61588724, 470.9635849 ]), 'right': {'feature_index': 0, 'threshold': 1001.029236892782, 'left': {'feature_index': 1, 'threshold': 1000.4104056820228, 'left': np.array([529.71971585, 547.71292837, 559.04378534, 522.35358676]), 'right': np.array([547.12901446, 558.87741587, 551.56006043])}, 'right': {'feature_index': 1, 'threshold': 1000.177815166058, 'left': np.array([576.8926402 , 563.58363749, 562.62597526]), 'right': np.array([601.76780838, 587.10134867, 608.64466352, 594.68664574])}}}, 'right': {'feature_index': 0, 'threshold': 999.1020792472432, 'left': np.array([576.85255167, 558.16409431]), 'right': {'feature_index': 1, 'threshold': 1000.9267078456826, 'left': np.array([604.7965235]), 'right': np.array([665.04739236, 666.44170815, 637.30962074, 643.13593178,
       657.71868008])}}}}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
dd = DecisionTreeRegressor()
dd.fit(X_train,y_train,10) 

print(dd.terminal_nodes)
print(dd.mean_square_error(y_train, dd.predict(X_train)))
print(dd.metric_error_for_pruning(y_train, dd.predict(X_train), 0.5)) 
X = dd.pruning(dd.tree,X_train,y_train,0.5)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
arbre, terminal_nodes = X  
dd.tree =arbre
print(dd.metric_error_for_pruning(y_train, dd.predict(X_train), 0.5))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
print(terminal_nodes)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
