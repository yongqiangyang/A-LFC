import numpy as np
import time
from numpy import nan
class agg():

    def __init__(self, model, data_train, answers, num_classes, batch_size=64, pi_prior=1.0):
        self.model = model
        self.data_train = data_train
        self.answers = answers
        self.batch_size = batch_size
        self.n_train = answers.shape[0]
        self.num_classes = num_classes
        self.num_annotators = answers.shape[1]

        # initialize estimated ground truth with majority voting
        self.ground_truth_est = np.zeros((self.n_train, self.num_classes))
        for i in range(self.n_train):
            votes = np.zeros(self.num_annotators)
            for r in range(self.num_annotators):
                if answers[i, r] != -1:
                    votes[answers[i, r]] += 1
            self.ground_truth_est[i, np.argmax(votes)] = 1.0

    def e_step(self):
        self.ground_truth_est = self.model.predict(self.data_train)
        adjustment_factor = np.ones((self.n_train, self.num_classes))
        for i in range(self.n_train):
            for r in range(self.num_annotators):
                if self.answers[i, r] != -1:
                    adjustment_factor[i] *= self.pi[r, :, self.answers[i][r]]
        self.ground_truth_est = adjustment_factor * self.ground_truth_est
        self.ground_truth_est = self.ground_truth_est / np.sum(self.ground_truth_est, 1).reshape(
            (self.n_train, 1))

    def m_step(self, epochs):
        hist = self.model.fit(self.data_train, self.ground_truth_est, epochs=1, shuffle=True,
                              batch_size=self.batch_size, verbose=0)

        self.pi = np.zeros((self.num_annotators, self.num_classes, self.num_classes))
        for r in range(self.num_annotators):
            normalizer = np.zeros(self.num_classes)
            for i in range(self.n_train):
                if self.answers[i, r] != -1:
                    normalizer += self.ground_truth_est[i, :]
                    self.pi[r, :, self.answers[i, r]] += self.ground_truth_est[i, :]
            normalizer[normalizer == 0] = 0.00001
            self.pi[r] = self.pi[r] / normalizer.reshape(self.num_classes, 1)

        return self.model, self.pi

    def run(self, epochs = 20):
        for epoch in range(epochs):
            model, pi = self.m_step(epoch)
            self.e_step()
        return model, pi

class adv_agg():

    def __init__(self, model, training_data, adv_training_data, answers, num_classes, batch_size=64, alpha=0.5):
        self.model = model
        self.training_data = training_data
        self.adv_training_data = adv_training_data
        self.answers = answers
        self.batch_size = batch_size
        self.n_train = answers.shape[0]
        self.num_classes = num_classes
        self.num_annotators = answers.shape[1]
        self.alpha = alpha

        # initialize estimated ground truth with majority voting
        self.ground_truth_est = np.zeros((self.n_train, self.num_classes))
        for i in range(self.n_train):
            votes = np.zeros(self.num_annotators)
            for r in range(self.num_annotators):
                if answers[i, r] != -1:
                    votes[answers[i, r]] += 1
            self.ground_truth_est[i, np.argmax(votes)] = 1.0
        
        self.ground_truth_est_adv_training = np.vstack((self.ground_truth_est, self.ground_truth_est))
        
    def e_step(self):
        adjustment_factor = np.ones((self.n_train, self.num_classes))
        for i in range(self.n_train):
            for r in range(self.num_annotators):
                if self.answers[i, r] != -1:
                    adjustment_factor[i] *= self.pi[r, :, self.answers[i][r]]

        ground_truth_est_1 = self.model.predict(self.training_data)
        self.ground_truth_est_1 = adjustment_factor * ground_truth_est_1
        self.ground_truth_est_1 = self.ground_truth_est_1 / np.sum(self.ground_truth_est_1, 1).reshape(
            (self.n_train, 1))

        ground_truth_est_2 = self.model.predict(self.adv_training_data)
        self.ground_truth_est_2 = adjustment_factor * ground_truth_est_2
        self.ground_truth_est_2 = self.ground_truth_est_2 / np.sum(self.ground_truth_est_2, 1).reshape(
            (self.n_train, 1))
        self.ground_truth_est_2[np.isnan(self.ground_truth_est_2)] = 1 / self.num_classes

        # mu = np.reshape(np.sum((ground_truth_est_1 * self.ground_truth_est_1), axis=1), (-1,1))
        
        self.ground_truth_est = (1 - self.alpha) * self.ground_truth_est_1 + self.alpha * self.ground_truth_est_2 
        self.ground_truth_est_adv_training = np.vstack((self.ground_truth_est_1, self.ground_truth_est_2))
        return self.ground_truth_est

    def m_step(self, epochs):
        self.training_data_concat = np.vstack((self.training_data, self.adv_training_data))
        hist = self.model.fit(self.training_data_concat, self.ground_truth_est_adv_training, epochs=1, shuffle=True, batch_size=self.batch_size, verbose=0)

        self.pi = np.zeros((self.num_annotators, self.num_classes, self.num_classes))
        for r in range(self.num_annotators):
            normalizer = np.zeros(self.num_classes)
            for i in range(self.n_train):
                if self.answers[i, r] != -1:
                    normalizer += self.ground_truth_est[i, :]
                    self.pi[r, :, self.answers[i, r]] += self.ground_truth_est[i, :]
            normalizer[normalizer == 0] = 0.00001
            self.pi[r] = self.pi[r] / normalizer.reshape(self.num_classes, 1)

        return self.model, self.pi

    def run(self, epochs = 20):
        for epoch in range(epochs):
            model, pi = self.m_step(epoch)
            ground_truth_est = self.e_step()
        return model, ground_truth_est, pi