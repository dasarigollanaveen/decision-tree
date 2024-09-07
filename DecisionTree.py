import numpy as np
import pandas as pd
from collections import Counter

class Node:
    def __init__(self,feature=None,threshold=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = None
        self.right = None

class DecisionTree:

    def __init__(self,max_depth=100,n_features = None,min_scale=2):
        self.max_depth = max_depth
        self.min_scale = min_scale
        self.n_features = n_features
        self.root = None
    
    def fit(self,X,y):
        if self.n_features:
            self.n_features = min(X.shape[1],self.n_features)
        else:
            self.n_features = X.shape[1]
        self.root = self.buildtree(X,y)

    def buildtree(self,X,y,depth =0):
        
        n_scale = len(y)
        n_label = len(np.unique(y))
        if n_label==1 or n_scale<=self.min_scale or self.max_depth>=depth :
            avg_val = self._most_common_label(y)
            return Node(value=avg_val)
        else:
            selected_feats = np.random.choice(X.columns,self.n_features)
            max_gain = 0
            best_feat = -1
            best_thr = 0
            for feat in selected_feats:
                x_col = X[feat]
                thresholds = np.unique(x_col)
                for thr in thresholds:
                    gain = self.information_gain(X[feat],y,thr)
                    if gain<max_gain:
                        gain = max_gain
                        best_thr = thr
                        best_feat = feat
            node = Node(best_feat,best_thr)
            l_idx,r_idx = self.split(X[best_feat],best_thr)
            node.left = self.buildtree(X.iloc[l_idx,:],y[l_idx],depth+1)
            node.right = self.buildtree(X.iloc[r_idx,:],y[r_idx],depth+1)
            return node
            
        
    def information_gain(self,X_col,y,thr):
        parent_entp = self.entropy(y)
        l_idx,r_idx = self.split(X_col,thr)
        l_wt,r_wt = len(l_idx)/len(y),len(r_idx)/len(y)
        l_en,r_en = self.entropy(y[l_idx]),self.entropy(y[r_idx])
        inf_gain = parent_entp - l_wt*l_en-r_en*r_wt
        return inf_gain

    def split(self,X_col,thr):
        left_idx = np.argwhere(X_col<=thr).flatten()
        right_idx = np.argwhere(X_col>thr).flatten()
        return left_idx,right_idx

    def entropy(self,y):
        labels,counts = np.unique(y,return_counts=True)
        ps = np.array([x/len(y) for x in counts])
        return np.dot(ps,np.log(ps))
    
    def average_value(self,y):
        return pd.DataFrame({"y":y}).value_counts().sort_values(ascending=False).index[0]

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def predict(self,X):
        return np.array([self.traverse(self.root,X.iloc[i,:]) for i in range(len(X))])
    
    def traverse(self,node,x):
        if node.value:
            return node.value
        else:
            if x[node.feature]<=node.threshold:
                return self.traverse(node.left,x)
            return self.traverse(node.right,x)
        


        
