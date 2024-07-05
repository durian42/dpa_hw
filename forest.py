from sklearn.model_selection import train_test_split
import numpy as np
from tree import Tree_base

class random_forest:
    def __init__(self, X, y, number=44, alpha=0.85):
        self.number = number
        self.alpha = alpha
        self.X = X
        self.y = y
        self.forest = self.bust_forest(self.X, self.y, number=self.number, alpha=self.alpha)
    
    def bust_forest(self, X, y, number=44, alpha=0.85):
        forest = []
        for i in (range(number)):
            X_train, _, y_train, _ = train_test_split(X, y, train_size=alpha, random_state=i)
            forest.append(Tree_base(np.array(X_train), y_train))
        return forest

    def predict(self, x):
        return np.mean(list(map(lambda t: t.predict(x), np.array(self.forest))), axis=0)
