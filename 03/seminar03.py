import numpy as np

class MyGaussianNBClassifier():
    def __init__(self, priors=None):
        self.priors = priors
        self.count_class = None
        self.exp_val = []
        self.dispersion = []

    def fit(self, X, y):
        self.count_class = np.unique(y)
        self.priors = []
        for mark in self.count_class:
            solo_prior = np.mean(y == mark)
            solo_exp_val = np.mean(X[y == mark], axis=0)
            solo_dispersion = np.var(X[y == mark], axis=0)
            self.priors.append(solo_prior)
            self.exp_val.append(solo_exp_val)
            self.dispersion.append(solo_dispersion)

    def predict(self, X):
        if self.priors is None:
            raise ValueError("The model is not trained. First do the fit")
        predicted_marks = []
        for x in X:
            probabilities = self.predict_proba(x)
            predicted_mark = self.count_class[np.argmax(probabilities)]
            predicted_marks.append(predicted_mark)
        return predicted_marks

    def predict_proba(self, X):
        if self.priors is None:
            raise ValueError("The model is not trained. First do the fit")
        probabilities = []
        for index_class in range(len(self.count_class)):
            prior = self.priors[index_class]
            exp_val = self.exp_val[index_class]
            dispersion = self.dispersion[index_class]
            probability_solo = (np.exp(-0.5 * ((X - exp_val) ** 2) / dispersion))/(np.sqrt(2 * np.pi * dispersion))
            probability = np.prod(probability_solo)
            probabilities.append(prior * probability)
        return probabilities

    def score(self, x, y):
        if self.priors is None:
            raise ValueError("The model is not trained. First do the fit")
        predictions = self.predict(x)
        return np.mean(predictions == y)