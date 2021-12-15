import numpy as np

import sklearn.linear_model as lm

import math

class Crowdsdsalgorithm_mulclass():



    def __init__(self,  answers, batch_size=16, pi_prior=1.0):

        #self.model = model

        #self.data_train = data_train

        self.answers = answers

        self.batch_size = batch_size

        self.pi_prior = pi_prior

        self.n_train = answers.shape[0]

        self.num_classes = np.max(answers) + 1

        self.num_annotators = answers.shape[1]



        # initialize annotators as reliable (almost perfect)

        self.pi = self.pi_prior * np.ones((self.num_classes, self.num_classes, self.num_annotators))



        # initialize estimated ground truth with majority voting

        self.ground_truth_est = np.zeros((self.n_train, self.num_classes))

        # for i in range(self.n_train):

        # 	votes = np.zeros(self.num_annotators)

        # 	for r in range(self.num_annotators):

        # 		if answers[i,r] != -1:

        # 			votes[answers[i,r]] += 1

        # 	self.ground_truth_est[i,np.argmax(votes)] = 1.0

        for i in range(self.n_train):

            votes = np.zeros(self.num_classes)

            for r in range(self.num_annotators):

                if answers[i, r] != -1:

                    votes[answers[i, r]] += 1

            self.ground_truth_est[i, np.argmax(votes)] = 1.0



    def e_step(self):

        # print "E-step"

        for i in range(self.n_train):

            adjustment_factor = np.ones(self.num_classes)

            for r in range(self.num_annotators):

                if self.answers[i, r] != -1:

                    adjustment_factor *= self.pi[:, self.answers[i, r],

                                         r]  # 妙极！pi 连乘的那一列分别是真值是�?开始的。然后对应位置点乘。就是一起算所有类别上的�?

            self.ground_truth_est[i, :] = np.transpose(adjustment_factor) * self.ground_truth_est[i, :]

            self.ground_truth_est[i, :] = self.ground_truth_est[i, :] / np.sum(self.ground_truth_est[i, :])

        # 最后点乘分类器概率

        # 这里一个问题就是预测概率之和不等于1 没有归一�?



        return self.ground_truth_est



    def m_step(self, ):

        # print "M-step"

        #hist = self.model.fit(self.data_train, self.ground_truth_est, epochs=1, shuffle=True,

                              #batch_size=self.batch_size, verbose=0)

        # print "loss:", hist.history["loss"][-1]

        #self.ground_truth_est = self.model.predict(self.data_train)

        # 混淆矩阵先验知识全都�?？？？预测概率之和不等于1

        self.pi = self.pi_prior * np.ones((self.num_classes, self.num_classes, self.num_annotators))

        for r in range(self.num_annotators):

            for i in range(self.num_classes):

                self.pi[i, i, r] = 1.0

        for r in range(self.num_annotators):

            normalizer = np.zeros(self.num_classes)

            for i in range(self.n_train):

                if self.answers[i, r] != -1:

                    # self.pi[:,self.answers[i,r],r] += np.transpose(self.ground_truth_est[i,:])



                    self.pi[:, self.answers[i, r], r] += np.transpose(self.ground_truth_est[i, :])

                    normalizer += self.ground_truth_est[i, :]

            normalizer = np.expand_dims(normalizer, axis=1)

            self.pi[:, :, r] = self.pi[:, :, r] / np.tile(normalizer, [1, self.num_classes])



        return self.pi

class Crowdsdsalgorithm_binaryclass():

    def __init__(self,answers, batch_size=16, alpha_prior=1.0, beta_prior=1.0):



        self.answers = answers

        self.batch_size = batch_size

        self.alpha_prior = alpha_prior

        self.beta_prior = beta_prior

        self.n_train = answers.shape[0]

        self.num_classes = np.max(answers) + 1

        self.num_annotators = answers.shape[1]



        # initialize annotators as reliable (almost perfect)

        self.alpha = 0.5 * np.ones(self.num_annotators)

        self.beta = 0.5 * np.ones(self.num_annotators)



        # initialize estimated ground truth with majority voting

        self.ground_truth_est = np.zeros((self.n_train, 2))

        for i in range(self.n_train):

            votes = np.zeros(self.num_classes)

            for r in range(self.num_annotators):

                if answers[i, r] != -1:

                    votes[answers[i, r]] += 1

            self.ground_truth_est[i, np.argmax(votes)] = 1



    def e_step(self):

        # print "E-step"

        for i in range(self.n_train):

            a = 1.0

            b = 1.0

            for r in range(self.num_annotators):

                if self.answers[i, r] != -1:

                    if self.answers[i, r] == 1:

                        a *= self.alpha[r]

                        b *= (1 - self.beta[r])

                    elif self.answers[i, r] == 0:

                        a *= (1 - self.alpha[r])

                        b *= self.beta[r]

                    else:

                        raise Exception()

            mu = (a * self.ground_truth_est[i, 1]) / (a * self.ground_truth_est[i, 1] + b * self.ground_truth_est[i, 0])

            self.ground_truth_est[i, 1] = mu

            self.ground_truth_est[i, 0] = 1.0 - mu



        return self.ground_truth_est



    def m_step(self, epochs=1):

        # print "M-step"

        

        # print "loss:", hist.history["loss"][-1]



        self.alpha = self.alpha_prior * np.zeros(self.num_annotators)

        self.beta = self.beta_prior * np.zeros(self.num_annotators)

        for r in range(self.num_annotators):

            if r == 75:

                s=1

            alpha_norm = 0.00000001

            beta_norm = 0.00000001

            for i in range(self.n_train):

                if self.answers[i, r] != -1:

                    alpha_norm += self.ground_truth_est[i, 1]

                    beta_norm += self.ground_truth_est[i, 0]

                    if self.answers[i, r] == 1:

                        self.alpha[r] += self.ground_truth_est[i, 1]

                    elif self.answers[i, r] == 0:

                        self.beta[r] += self.ground_truth_est[i, 0]

                    else:

                        raise Exception()

            if r == 75:

                s = 1

            self.alpha[r]/= alpha_norm

            self.beta[r]/= beta_norm



        return self.alpha, self.beta





class CrowdsCategoricalLFC():



    def __init__(self, data_train, answers, batch_size=16, pi_prior=1.0):



        self.data_train = data_train

        self.answers = answers

        self.batch_size = batch_size

        self.pi_prior = pi_prior

        self.n_train = answers.shape[0]

        self.num_classes = np.max(answers) + 1

        self.num_annotators = answers.shape[1]



        # initialize annotators as reliable (almost perfect)

        self.pi = self.pi_prior * np.ones((self.num_classes, self.num_classes, self.num_annotators))



        # initialize estimated ground truth with majority voting

        self.ground_truth_est = np.zeros((self.n_train, self.num_classes))

        # for i in range(self.n_train):

        # 	votes = np.zeros(self.num_annotators)

        # 	for r in range(self.num_annotators):

        # 		if answers[i,r] != -1:

        # 			votes[answers[i,r]] += 1

        # 	self.ground_truth_est[i,np.argmax(votes)] = 1.0

        for i in range(self.n_train):

            votes = np.zeros(self.num_classes)

            for r in range(self.num_annotators):

                if answers[i, r] != -1:

                    votes[answers[i, r]] += 1

            self.ground_truth_est[i, np.argmax(votes)] = 1.0



    def e_step(self):

        # print "E-step"

        for i in range(self.n_train):

            adjustment_factor = np.ones(self.num_classes)

            for r in range(self.num_annotators):

                if self.answers[i, r] != -1:

                    adjustment_factor *= self.pi[:, self.answers[i, r],

                                         r]  # 妙极！pi 连乘的那一列分别是真值是从0开始的。然后对应位置点乘。就是一起算所有类别上的。

            self.ground_truth_est[i, :] = np.transpose(adjustment_factor) * self.ground_truth_est[i, :]

            self.ground_truth_est[i, :] = self.ground_truth_est[i, :] / np.sum(self.ground_truth_est[i, :])

        # 最后点乘分类器概率

        # 这里一个问题就是预测概率之和不等于1 没有归一化



        return self.ground_truth_est



    def m_step(self, ):

        # print "M-step"

        # hist = self.model.fit(self.data_train, self.ground_truth_est, epochs=1, shuffle=True,

        #                       batch_size=self.batch_size, verbose=0)

        model = lm.LogisticRegression(solver='liblinear', C=50)  # C

        model.fit(self.data_train, self.ground_truth_est)

        # print "loss:", hist.history["loss"][-1]

        self.ground_truth_est = self.model.predict(self.data_train)

        # 混淆矩阵先验知识全都是1？？？预测概率之和不等于1

        self.pi = self.pi_prior * np.ones((self.num_classes, self.num_classes, self.num_annotators))

        for r in range(self.num_annotators):

            for i in range(self.num_classes):

                self.pi[i, i, r] = 1.0

        for r in range(self.num_annotators):

            normalizer = np.zeros(self.num_classes)

            for i in range(self.n_train):

                if self.answers[i, r] != -1:

                    # self.pi[:,self.answers[i,r],r] += np.transpose(self.ground_truth_est[i,:])



                    self.pi[:, self.answers[i, r], r] += np.transpose(self.ground_truth_est[i, :])

                    normalizer += self.ground_truth_est[i, :]

            normalizer = np.expand_dims(normalizer, axis=1)

            self.pi[:, :, r] = self.pi[:, :, r] / np.tile(normalizer, [1, self.num_classes])



        return self.model, self.pi







class EM:

    def __init__(self, e2wl, w2el, label_set, initquality):

        self.e2wl = e2wl

        self.w2el = w2el

        self.workers = list(self.w2el.keys())

        self.label_set = label_set

        self.initalquality = initquality



    # E-step

    def Update_e2lpd(self):

        self.e2lpd = {}



        for example, worker_label_set in list(self.e2wl.items()):  # e2wl is the instance wrt worker and label

            lpd = {}

            total_weight = 0



            for tlabel, prob in list(self.l2pd.items()):

                weight = prob

                for (w, label) in worker_label_set:

                    weight *= self.w2cm[w][tlabel][label]



                lpd[tlabel] = weight

                total_weight += weight



            for tlabel in lpd:  # 归一化

                if total_weight == 0:

                    # uniform distribution

                    lpd[tlabel] = 1.0 / len(self.label_set)

                else:

                    lpd[tlabel] = lpd[tlabel] * 1.0 / total_weight



            self.e2lpd[example] = lpd



            # M-step



    def Update_l2pd(self):  # l2pd    update prior prob of classes   l2pd

        for label in self.l2pd:

            self.l2pd[label] = 0



        for _, lpd in list(self.e2lpd.items()):  # e2lpd post prob of instances

            for label in lpd:

                self.l2pd[label] += lpd[label]



        for label in self.l2pd:

            self.l2pd[label] *= 1.0 / len(self.e2lpd)



    def Update_w2cm(self):



        for w in self.workers:  # the confuse matrix is reset to 0

            for tlabel in self.label_set:

                for label in self.label_set:

                    self.w2cm[w][tlabel][label] = 0



        w2lweights = {}

        for w in self.w2el:

            w2lweights[w] = {}

            for label in self.label_set:

                w2lweights[w][label] = 0

            for example, _ in self.w2el[w]:

                for label in self.label_set:

                    w2lweights[w][label] += self.e2lpd[example][label]  # 计算出工人给的某一类别的    标签  概率   之和



            for tlabel in self.label_set:



                if w2lweights[w][tlabel] == 0:

                    for label in self.label_set:

                        if tlabel == label:

                            self.w2cm[w][tlabel][label] = self.initalquality

                        else:

                            self.w2cm[w][tlabel][label] = (1 - self.initalquality) * 1.0 / (len(self.label_set) - 1)



                    continue



                for example, label in self.w2el[w]:

                    self.w2cm[w][tlabel][label] += self.e2lpd[example][tlabel] * 1.0 / w2lweights[w][tlabel]



        return self.w2cm



    # initialization

    def Init_l2pd(self):

        # uniform probability distribution

        l2pd = {}

        for label in self.label_set:

            l2pd[label] = 1.0 / len(self.label_set)

        return l2pd



    def Init_w2cm(self):

        w2cm = {}  ##the confuse matrix of  work: diagonal element is init quality,

        for worker in self.workers:

            w2cm[worker] = {}

            for tlabel in self.label_set:

                w2cm[worker][tlabel] = {}

                for label in self.label_set:

                    if tlabel == label:

                        w2cm[worker][tlabel][label] = self.initalquality

                    else:

                        w2cm[worker][tlabel][label] = (1 - self.initalquality) / (len(self.label_set) - 1)



        return w2cm



    def Run(self, iterr=20):



        self.l2pd = self.Init_l2pd()

        self.w2cm = self.Init_w2cm()



        while iterr > 0:

            # E-step

            self.Update_e2lpd()



            # M-step

            self.Update_l2pd()

            self.Update_w2cm()



            # compute the likelihood

            # print self.computelikelihood()



            iterr -= 1



        return self.e2lpd, self.w2cm



    def computelikelihood(self):



        lh = 0



        for _, worker_label_set in list(self.e2wl.items()):

            temp = 0

            for tlabel, prior in list(self.l2pd.items()):

                inner = prior

                for worker, label in worker_label_set:

                    inner *= self.w2cm[worker][tlabel][label]

                temp += inner



            lh += math.log(temp)



        return lh

