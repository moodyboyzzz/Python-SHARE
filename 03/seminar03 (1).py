#!/usr/bin/env python
# coding: utf-8

# ### Наивный байесовский классификатор

# $$P(y | x_1,x_2,...,x_n) = \frac{P(y) P(x_1,x_2,...,x_n|y)}{P(x_1,x_2,...,x_n)} $$

# В силу 'наивного' предположения о независимости признаков $x_1,x_2,..,x_n$ получаем:

# $$ P(y | x_1,x_2,...,x_n) = \frac{P(y) \prod\limits_{i=1}^{n}P(x_i| y)}{P(x_1,x_2,...,x_n)} $$

# Откуда следует, что

# $$ P(y | x_1,x_2,...,x_n) \propto P(y) \prod\limits_{i=1}^{n}P(x_i| y) $$

# $$\hat{y} = arg \max_{y} P(y) \prod\limits_{i=1}^{n}P(x_i| y) $$

# В данном задании будем предполагать, что 

# $$ P(x_i | y) = \frac{1}{\sqrt{2\pi \sigma_y^2}} e^{- \frac{(x_i - \mu_y)^2}{2\sigma_y^2}}, $$

# где $\mu_y$ и $\sigma_y$ считаются по оценке максимального правдобия

# ### Задание

# Необходимо реализовать наивный байесовский классификтор для нормального распределения.
# Сам код необходимо оформить и отправить боту в виде класса MyGaussianNBClassifier в файле seminar03.py
# 

# In[1]:


import numpy as np
class MyGaussianNBClassifier():
    def __init__(self, priors=None):
        self.priors = priors
        self.classes = None
        self.class_priors = None
        self.class_means = None
        self.class_variances = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = self._calculate_class_priors(y)
        self.class_means = self._calculate_class_means(X, y)
        self.class_variances = self._calculate_class_variances(X, y)

    def predict(self, X):
        predicted_labels = []
        for x in X:
            probabilities = self.predict_proba(x)
            predicted_label = self.classes[np.argmax(probabilities)]
            predicted_labels.append(predicted_label)
        return np.array(predicted_labels)

    def predict_proba(self, X):
        probabilities = []
        for class_idx in range(len(self.classes)):
            class_prior = self.class_priors[class_idx]
            class_mean = self.class_means[class_idx]
            class_variance = self.class_variances[class_idx]
            class_probability = self._calculate_class_probability(X, class_mean, class_variance)
            probabilities.append(class_prior * class_probability)
        
        probabilities /= np.sum(probabilities)
        return probabilities

    def score(self, X, y):
        predicted_labels = self.predict(X)
        return np.mean(predicted_labels == y)

    def _calculate_class_priors(self, y):
        class_priors = []
        for class_label in self.classes:
            class_prior = np.mean(y == class_label)
            class_priors.append(class_prior)
        return class_priors

    def _calculate_class_means(self, X, y):
        class_means = []
        for class_label in self.classes:
            class_mean = np.mean(X[y == class_label], axis=0)
            class_means.append(class_mean)
        return class_means

    def _calculate_class_variances(self, X, y):
        class_variances = []
        for class_label in self.classes:
            class_variance = np.var(X[y == class_label], axis=0)
            class_variances.append(class_variance)
        return class_variances

    def _calculate_class_probability(self, x, class_mean, class_variance):
        probabilities = (np.exp(-0.5 * ((x - class_mean) ** 2) / class_variance))/(np.sqrt(2 * np.pi * class_variance))
        return np.prod(probabilities)
    
    


# Ваша реализация дожна поддерживать методы predict, predict_proba, score аналоично методам класса sklearn.naive_bayes.GaussianNB

# In[2]:


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = MyGaussianNBClassifier()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1], [5, 6], [-1, 0], [0, 0]]))


# In[ ]:




