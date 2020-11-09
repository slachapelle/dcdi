"""Defining a set of classes that represent causal functions/ mechanisms.

Author: Diviyan Kalainathan
Modified by Philippe Brouillard, July 24th 2019

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""

import random
import numpy as np
from scipy.stats import bernoulli
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.gaussian_process import GaussianProcessRegressor
import torch as th
import copy


class LinearMechanism(object):
    """Linear mechanism, where Effect = alpha*Cause + Noise."""

    def __init__(self, ncauses, points, noise_function, d=4, noise_coeff=.4):
        """Init the mechanism."""
        super(LinearMechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.coefflist = []
        self.other_coefflist = []

        self.noise_coeff = noise_coeff
        self.noise_function = noise_function

        for i in range(ncauses):
            coeff = np.random.uniform(0.25, 1)
            if np.random.randint(2) == 0:
                coeff *= -1
            self.coefflist.append(coeff)

        self.old_coefflist = self.coefflist[:]

    def parametric_intervention(self):
        for i,c in enumerate(self.old_coefflist):
            change = np.random.uniform(0.5, 1)
            if c > 0:
                coeff = c + change
            else:
                coeff = c - change
            self.coefflist[i] = coeff

    def unique_parametric_intervention(self):
        if len(self.other_coefflist) == 0:
            for i,c in enumerate(self.old_coefflist):
                change = np.random.uniform(2, 5)
                if np.random.randint(2) == 0:
                    change *= -1
                if c > 0:
                    coeff = c + change
                else:
                    coeff = c - change

                self.other_coefflist.append(coeff)
        self.coefflist = self.other_coefflist[:]

    def reinit(self):
        self.coefflist = self.old_coefflist[:]

    def __call__(self, causes):
        """Run the mechanism."""
        # Additive only, for now
        effect = np.zeros((self.points, 1))
        self.noise = self.noise_coeff * self.noise_function(self.points)

        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            effect[:, 0] = effect[:, 0] + self.coefflist[par]*causes[:, par]
        effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect


class SigmoidMix_Mechanism(object):
    def __init__(self, ncauses, points, noise_function, d=4, noise_coeff=.4):
        """Init the mechanism."""
        super(SigmoidMix_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points

        self.a = np.random.exponential(1/4) + 1
        ber = bernoulli.rvs(0.5)
        self.b = ber * np.random.uniform(-2, -0.5) + (1-ber)*np.random.uniform(0.5, 2)
        self.c = np.random.uniform(-2, 2)
        self.noise_coeff = noise_coeff
        self.noise_function = noise_function

        self.old_b = self.b
        self.old_c = self.c
        self.other_b = None
        self.other_c = None

    def parametric_intervention(self):
        change = np.random.uniform(0.5, 1)
        if self.b <= -0.5:
            self.b -= change
        else:
            self.b += change

        change = np.random.uniform(-1, 1)
        self.c += change

    def unique_parametric_intervention(self):
        if self.other_b is None and self.other_c is None:
            self.parametric_intervention()
            self.other_b = self.b
            self.other_c = self.c
        self.b = self.other_b
        self.c = self.other_c

    def reinit(self):
        self.b = self.old_b
        self.c = self.old_c

    def mechanism(self, causes):
        """Mechanism function."""
        self.noise = self.noise_coeff * self.noise_function(self.points)

        result = np.zeros((self.points, 1))
        for i in range(self.points):
            pre_add_effect = 0
            for c in range(causes.shape[1]):
                pre_add_effect += causes[i, c]
            pre_add_effect += self.noise[i]

            result[i, 0] = self.a * self.b * \
                (pre_add_effect + self.c)/(1 + abs(self.b*(pre_add_effect + self.c)))

        return result

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution

        effect[:, 0] = self.mechanism(causes)[:, 0]
        return effect


class SigmoidAM_Mechanism(object):
    def __init__(self, ncauses, points, noise_function, d=4, noise_coeff=.4):
        """Init the mechanism."""
        super(SigmoidAM_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points

        self.a = np.random.exponential(1/4) + 1
        ber = bernoulli.rvs(0.5)
        self.b = ber * np.random.uniform(-2, -0.5) + (1-ber)*np.random.uniform(0.5, 2)
        self.c = np.random.uniform(-2, 2)
        self.noise_coeff = noise_coeff
        self.noise_function = noise_function

        self.old_b = self.b
        self.old_c = self.c
        self.other_b = None
        self.other_c = None

    def mechanism(self, x):
        """Mechanism function."""

        result = np.zeros((self.points, 1))
        for i in range(self.points):
            result[i, 0] = self.a * self.b * (x[i] + self.c) / (1 + abs(self.b * (x[i] + self.c)))

        return result

    def __call__(self, causes):
        """Run the mechanism."""
        # Additive only
        self.noise = self.noise_coeff * self.noise_function(self.points)
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            effect[:, 0] = effect[:, 0] + self.mechanism(causes[:, par])[:, 0]

        effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect


class ANM_Mechanism(object):
    def __init__(self, ncauses, points, noise_function, noise_coeff=.4):
        """Init the mechanism."""
        super(ANM_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.noise_function = noise_function
        self.noise_coeff = noise_coeff
        self.nb_step = 0

    def mechanism(self, x):
        """Mechanism function."""
        self.nb_step += 1
        x = np.reshape(x, (x.shape[0], x.shape[1]))

        if(self.nb_step == 1):
            cov = computeGaussKernel(x)
            mean = np.zeros((1, self.points))[0, :]
            y = np.random.multivariate_normal(mean, cov)
            self.gpr = GaussianProcessRegressor()
            self.gpr.fit(x, y)
        else:
            y = self.gpr.predict(x)

        return y

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        self.noise = self.noise_coeff * self.noise_function(self.points)

        # Compute each cause's contribution
        if(causes.shape[1] > 0):
            effect[:, 0] = self.mechanism(causes)
        else:
            effect[:, 0] = self.mechanism(self.noise)

        effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect


class NN_Mechanism_Add(object):
    def __init__(self, ncauses, points, noise_function, nh=10, noise_coeff=.4):
        """Init the mechanism."""
        super(NN_Mechanism_Add, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.noise_coeff = noise_coeff
        self.noise_function = noise_function
        self.nb_step = 0
        self.nh = nh
        self.layers = self.initialize()
        self.old_layers = copy.deepcopy(self.layers)
        self.other_layers = None

    def weight_init(self, model):
        if isinstance(model, th.nn.modules.Linear):
            th.nn.init.normal_(model.weight.data, mean=0., std=1)

    def initialize(self):
        """Mechanism function."""
        layers = []

        layers.append(th.nn.modules.Linear(self.n_causes, self.nh))
        layers.append(th.nn.PReLU())
        layers.append(th.nn.modules.Linear(self.nh, 1))

        layers = th.nn.Sequential(*layers)
        layers.apply(self.weight_init)

        return layers

    def parametric_intervention(self):
        for i,layer in enumerate(self.layers):
            if isinstance(layer, th.nn.modules.Linear):
                with th.no_grad():
                    layer.weight += th.empty_like(layer.weight).normal_(mean=0, std=.1)

    def unique_parametric_intervention(self):
        if self.other_layers is None:
            self.other_layers = copy.deepcopy(self.layers)
            for i,layer in enumerate(self.other_layers):
                if isinstance(layer, th.nn.modules.Linear) and i > 0:
                    with th.no_grad():
                        layer.weight += th.empty_like(layer.weight).normal_(mean=0, std=1)
        self.layers = copy.deepcopy(self.other_layers)

    def reinit(self):
        self.layers = copy.deepcopy(self.old_layers)

    def apply_nn(self, x):
        data = x.astype('float32')
        data = th.from_numpy(data)

        return np.reshape(self.layers(data).data, (x.shape[0],))

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        self.noise = self.noise_coeff * self.noise_function(self.points)

        # Compute each cause's contribution
        if (causes.shape[1] > 0):
            effect[:, 0] = self.apply_nn(causes)
        else:
            print("abnormal")
        effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect


class NN_Mechanism(object):
    def __init__(self, ncauses, points, noise_function, nh=20, noise_coeff=.4):
        """Init the mechanism."""
        super(NN_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.noise_coeff = noise_coeff
        self.noise_function = noise_function
        self.nb_step = 0
        self.nh = nh
        self.layers = self.initialize()
        self.old_layers = copy.deepcopy(self.layers)
        self.other_layers = None

    def weight_init(self, model):
        if isinstance(model, th.nn.modules.Linear):
            th.nn.init.normal_(model.weight.data, mean=0., std=1)

    def initialize(self):
        """Mechanism function."""
        layers = []

        layers.append(th.nn.modules.Linear(self.n_causes+1, self.nh))
        layers.append(th.nn.Tanh())
        layers.append(th.nn.modules.Linear(self.nh, 1))

        layers = th.nn.Sequential(*layers)
        layers.apply(self.weight_init)

        return layers

    def parametric_intervention(self):
        for i,layer in enumerate(self.layers):
            if isinstance(layer, th.nn.modules.Linear):
                with th.no_grad():
                    layer.weight += th.empty_like(layer.weight).normal_(mean=0, std=.1)

    def unique_parametric_intervention(self):
        if self.other_layers is None:
            self.other_layers = copy.deepcopy(self.layers)
            for i,layer in enumerate(self.other_layers):
                if isinstance(layer, th.nn.modules.Linear) and i > 0:
                    with th.no_grad():
                        layer.weight += th.empty_like(layer.weight).normal_(mean=0, std=1)
        self.layers = copy.deepcopy(self.other_layers)

    def reinit(self):
        self.layers = copy.deepcopy(self.old_layers)

    def apply_nn(self, x):
        data = x.astype('float32')
        data = th.from_numpy(data)

        return np.reshape(self.layers(data).data, (x.shape[0],))

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        self.noise = self.noise_coeff * self.noise_function(self.points)
        # Compute each cause's contribution
        if (causes.shape[1] > 0):
            mix = np.hstack((causes, self.noise))
            effect[:, 0] = self.apply_nn(mix)
        else:
            effect[:, 0] = self.apply_nn(self.noise)

        return effect


# === Multimodal Mechanisms ===
class Multimodal_X_Mechanism(object):
    """Mecanism with multimodal distribution: usually a combination of multiple
    functions"""

    def __init__(self, ncauses, points, noise_function, d=4, noise_coeff=.4):
        """Init the mechanism."""
        super(Multimodal_X_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.coefflist = []
        self.other_coefflist = []

        self.noise_coeff = noise_coeff
        self.noise_function = noise_function

        for i in range(ncauses):
            coeff = np.random.uniform(0.5, 1)
            if np.random.randint(2) == 0:
                coeff *= -1
            self.coefflist.append(coeff)

        self.old_coefflist = self.coefflist[:]

    def parametric_intervention(self):
        for i,c in enumerate(self.old_coefflist):
            change = np.random.uniform(0.5, 1)
            if c > 0:
                coeff = c + change
            else:
                coeff = c - change
            self.coefflist[i] = coeff

    def unique_parametric_intervention(self):
        if len(self.other_coefflist) == 0:
            for i,c in enumerate(self.old_coefflist):
                change = np.random.uniform(0.5, 1)
                if c > 0:
                    coeff = c + change
                else:
                    coeff = c - change
                self.other_coefflist.append(coeff)
        self.coefflist = self.other_coefflist[:]

    def reinit(self):
        self.coefflist = self.old_coefflist[:]

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        self.noise = self.noise_coeff * self.noise_function(self.points)

        selector = np.random.choice([-1,1], size=self.points)

        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            for i, sel in enumerate(selector):
                effect[i, 0] = effect[i, 0] + sel*self.coefflist[par]*causes[i, par]
        effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect


class Multimodal_Circle_Mechanism(object):
    def __init__(self, ncauses, points, noise_function, d=4, noise_coeff=.4):
        """Init the mechanism."""
        super(Multimodal_Circle_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.noise_coeff = noise_coeff
        self.noise_function = noise_function

        self.sin_scale = np.random.uniform(0.5, 1.5) #1
        self.period = np.random.uniform(0.5, 1.5) #1
        self.phase_shift = np.pi/2

        # make copy of initial parameters
        self.old_sin_scale = self.sin_scale
        self.old_period = self.period
        self.old_phase_shift = self.phase_shift
        self.other_sin_scale = None
        self.other_period = None
        self.other_phase_shift = None

    def parametric_intervention(self):
        change = np.random.uniform(0.5, 1.5)
        self.sin_scale = self.old_phase_shift
        self.period = np.random.uniform(0.5, 1.5) #1
        self.phase_shift = np.pi/2

    def unique_parametric_intervention(self):
        if self.other_sin_scale is None:
            self.parametric_intervention()
            self.other_sin_scale = self.sin_scale
            self.other_period = self.period
            self.other_phase_shift = self.phase_shift
        self.sin_scale = self.other_sin_scale
        self.period = self.other_period
        self.phase_shift = self.other_phase_shift

    def reinit(self):
        self.sin_scale = self.old_sin_scale
        self.period = self.old_period
        self.phase_shift = self.old_phase_shift

    def mechanism(self, sel, x):
        if sel:
            sin_scale = -self.sin_scale
        else:
            sin_scale = self.sin_scale
        return sin_scale * np.sin(self.period * (x + self.phase_shift))

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        self.noise = self.noise_coeff * self.noise_function(self.points)

        selector = np.random.choice([0,1], size=self.points)

        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            for i, sel in enumerate(selector):
                effect[i, 0] = effect[i, 0] + self.mechanism(sel, causes[i, par])
        effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect

class Multimodal_ADN_Mechanism(object):
    def __init__(self, ncauses, points, noise_function, d=4, noise_coeff=.4):
        """Init the mechanism."""
        super(Multimodal_ADN_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.noise_coeff = noise_coeff
        self.noise_function = noise_function

        self.sin_scale = np.random.uniform(0.5, 1.5) #1
        self.period = np.random.uniform(1, 2) #1
        self.phase_shift = np.pi/2

        # make copy of initial parameters
        self.old_sin_scale = self.sin_scale
        self.old_period = self.period
        self.old_phase_shift = self.phase_shift
        self.other_sin_scale = None
        self.other_period = None
        self.other_phase_shift = None

    def parametric_intervention(self):
        # change = np.random.uniform(1, 2)
        self.sin_scale = self.old_phase_shift
        change = np.random.uniform(1, 2)
        self.period = self.old_period + change
        self.phase_shift = np.pi/2

    def unique_parametric_intervention(self):
        if self.other_sin_scale is None:
            self.parametric_intervention()
            self.other_sin_scale = self.sin_scale
            self.other_period = self.period
            self.other_phase_shift = self.phase_shift
        self.sin_scale = self.other_sin_scale
        self.period = self.other_period
        self.phase_shift = self.other_phase_shift

    def reinit(self):
        self.sin_scale = self.old_sin_scale
        self.period = self.old_period
        self.phase_shift = self.old_phase_shift

    def mechanism(self, sel, x):
        if sel:
            sin_scale = -self.sin_scale
        else:
            sin_scale = self.sin_scale
        return sin_scale * np.sin(self.period * (x + self.phase_shift))

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        self.noise = self.noise_coeff * self.noise_function(self.points)

        selector = np.random.choice([0,1], size=self.points)

        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            for i, sel in enumerate(selector):
                effect[i, 0] = effect[i, 0] + self.mechanism(sel, causes[i, par])
        effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect

class Function_Template:
    def __init__(self, sign, slope, intercept, sin_scale, period, phase_shift):
        self.sign = sign
        self.slope = slope
        self.intercept = intercept
        self.sin_scale = sin_scale
        self.period = period
        self.phase_shift = phase_shift

    def __call__(self, x):
        return self.sign*self.slope*x + self.intercept \
            + self.sin_scale*np.sin(self.period*(x + self.phase_shift))


# ====================================

class Polynomial_Mechanism(object):
    def __init__(self, ncauses, points, noise_function, d=2, noise_coeff=.4):
        """Init the mechanism."""
        super(Polynomial_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.d = d
        self.polycause = []

        for c in range(ncauses):
            self.coefflist = []
            for j in range(self.d + 1):
                self.coefflist.append(random.random())
            self.polycause.append(self.coefflist)

        self.ber = bernoulli.rvs(0.5)
        self.noise = noise_coeff * noise_function(points)

    def mechanism(self, x, par):
        """Mechanism function."""
        list_coeff = self.polycause[par]
        result = np.zeros((self.points, 1))
        for i in range(self.points):
            for j in range(self.d+1):
                result[i, 0] += list_coeff[j]*np.power(x[i], j)
            result[i, 0] = min(result[i, 0], 1)
            result[i, 0] = max(result[i, 0], -1)

        return result

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            effect[:, 0] = effect[:, 0] + self.mechanism(causes[:, par], par)[:, 0]

        if(self.ber > 0 and causes.shape[1] > 0):
            effect[:, 0] = effect[:, 0] * self.noise[:, 0]
        else:
            effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect


def computeGaussKernel(x):
    """Compute the gaussian kernel on a 1D vector."""
    xnorm = np.power(euclidean_distances(x, x), 2)
    return np.exp(-xnorm / (2.0))


class GaussianProcessAdd_Mechanism(object):

    def __init__(self, ncauses, points, noise_function, noise_coeff=.4):
        """Init the mechanism."""
        super(GaussianProcessAdd_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points

        self.noise = noise_coeff * noise_function(points)
        self.nb_step = 0

    def mechanism(self, x):
        """Mechanism function."""
        self.nb_step += 1
        x = np.reshape(x, (x.shape[0], 1))

        cov = computeGaussKernel(x)
        mean = np.zeros((1, self.points))[0, :]
        y = np.random.multivariate_normal(mean, cov)

        # if(self.nb_step < 5):
        #     cov = computeGaussKernel(x)
        #     mean = np.zeros((1, self.points))[0, :]
        #     y = np.random.multivariate_normal(mean, cov)
        # elif(self.nb_step == 5):
        #     cov = computeGaussKernel(x)
        #     mean = np.zeros((1, self.points))[0, :]
        #     y = np.random.multivariate_normal(mean, cov)
        #     self.gpr = GaussianProcessRegressor()
        #     self.gpr.fit(x, y)
        #     y = self.gpr.predict(x)
        # else:
        #     y = self.gpr.predict(x)

        return y

    def __call__(self, causes):
        """Run the mechanism."""
        # Additive only
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            effect[:, 0] = effect[:, 0] + self.mechanism(causes[:, par])

        effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect


class GaussianProcessMix_Mechanism(object):

    def __init__(self, ncauses, points, noise_function, noise_coeff=.4):
        """Init the mechanism."""
        super(GaussianProcessMix_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.noise = noise_coeff * noise_function(points)
        self.nb_step = 0

    def mechanism(self, x):
        """Mechanism function."""
        self.nb_step += 1
        x = np.reshape(x, (x.shape[0], x.shape[1]))

        if(self.nb_step < 2):
            cov = computeGaussKernel(x)
            mean = np.zeros((1, self.points))[0, :]
            y = np.random.multivariate_normal(mean, cov)
        elif(self.nb_step == 2):
            cov = computeGaussKernel(x)
            mean = np.zeros((1, self.points))[0, :]
            y = np.random.multivariate_normal(mean, cov)
            self.gpr = GaussianProcessRegressor()
            self.gpr.fit(x, y)
            y = self.gpr.predict(x)
        else:
            y = self.gpr.predict(x)

        return y

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution
        if(causes.shape[1] > 0):
            mix = np.hstack((causes, self.noise))
            effect[:, 0] = self.mechanism(mix)
        else:
            effect[:, 0] = self.mechanism(self.noise)

        return effect




class pnl_gp_mechanism(object):
    """ Post-Nonlinear model using a GP with additive noise. The
    second non-linearity is a sigmoid """

    def __init__(self, ncauses, points, noise_function, noise_coeff=.4):
        """Init the mechanism."""
        super(pnl_gp_mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.noise = noise_coeff * noise_function(points)
        self.nb_step = 0
        self.f2 = lambda x: 1 / (1 + np.exp(-x))

    def mechanism(self, x):
        """Mechanism function."""
        self.nb_step += 1
        x = np.reshape(x, (x.shape[0], x.shape[1]))

        if(self.nb_step == 1):
            cov = computeGaussKernel(x)
            mean = np.zeros((1, self.points))[0, :]
            y = np.random.multivariate_normal(mean, cov)
            self.gpr = GaussianProcessRegressor()
            self.gpr.fit(x, y)
            y = self.gpr.predict(x)
        else:
            y = self.gpr.predict(x)

        return y

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution
        if(causes.shape[1] > 0):
            effect[:, 0] = self.mechanism(causes)
            effect[:, 0] = effect[:, 0] + self.noise[:, 0]
        else:
            effect[:, 0] = self.mechanism(self.noise)
        effect[:, 0] = self.f2(effect[:, 0])

        return effect


class pnl_mult_mechanism(object):
    """ Post-Nonlinear model using a exp and log as the non-linearities.
    This results in a multiplicative model. """

    def __init__(self, ncauses, points, noise_function, noise_coeff=.4):
        """Init the mechanism."""
        super(pnl_mult_mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.noise_function = noise_function
        self.noise_coeff = noise_coeff

        self.f1 = lambda x: np.log(np.sum(x, axis=1))
        self.f2 = lambda x: np.exp(x)

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        self.noise = self.noise_coeff * self.noise_function(self.points)

        # Compute each cause's contribution
        if(causes.shape[1] > 0):
            effect[:, 0] = self.f1(causes) #[:, 0]
        else:
            effect[:, 0] = self.f1(self.noise)

        effect[:, 0] = effect[:, 0] + self.noise[:, 0]
        effect[:, 0] = self.f2(effect[:, 0])

        return effect



class PostNonLinear_Mechanism:
    def __init__(self, ncauses, points, noise_function, f1=None, f2=None, noise_coeff=.4):
        self.gp = GaussianProcessAdd_Mechanism(ncauses, points, noise_function,
                                               noise_coeff=0)
        self.points = points
        self.noise = noise_coeff * noise_function(points)
        self.f1 = f1
        self.f2 = f2

        if f1 is None and f2 is None:
            raise ValueError("f1 and f2 have to de defined!")
        elif f1 is None and f2 is not None:
            self.f1 = self.gp

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution
        if(causes.shape[1] > 0):
            effect[:, 0] = self.f1(causes)[:,0] # mult [:, 0]
        else:
            effect[:, 0] = self.f1(self.noise)

        effect[:, 0] = effect[:, 0] + self.noise[:, 0]
        effect[:, 0] = self.f2(effect[:, 0])

        return effect


def gmm_cause(points, k=4, p1=2, p2=2):
    """Init a root cause with a Gaussian Mixture Model w/ a spherical covariance type."""
    g = GMM(k, covariance_type="spherical")
    g.fit(np.random.randn(300, 1))

    g.means_ = p1 * np.random.randn(k, 1)
    g.covars_ = np.power(abs(p2 * np.random.randn(k, 1) + 1), 2)
    g.weights_ = abs(np.random.rand(k))
    g.weights_ = g.weights_ / sum(g.weights_)
    return g.sample(points)[0].reshape(-1)

def gaussian_cause(points):
    """Init a root cause with a Gaussian."""
    return np.random.randn(points, 1)[:, 0]

def variable_gaussian_cause(points):
    """Init a root cause with a Gaussian. Similar to gaussian_cause
    but have variable variance. Identical to J.Peters with default value (set noise_coeff=0.2)"""
    # + np.random.rand(points, 1)[:, 0] - 1
    return np.sqrt(np.random.rand(1) + 1) * np.random.randn(points, 1)[:, 0]

def uniform_cause(points):
    """Init a root cause with a uniform."""
    return np.random.rand(points, 1)[:, 0] * 2 - 1

def uniform_cause_positive(points):
    """Init a root cause with a uniform."""
    return np.random.rand(points, 1)[:, 0] * 2

def normal_noise(points):
    """Init a normal noise variable."""
    return np.random.rand(1) * np.random.randn(points, 1) \
        + random.sample([2, -2], 1)

def variable_normal_noise(points):
    """Init a normal noise variable. Similar to normal_noise
    but make sure to have at least a std of 1. Identical to
    J.Peters with default value (set noise_coeff=0.2)"""
    return np.sqrt(np.random.rand(1) + 1) * np.random.randn(points, 1)

def absolute_gaussian_noise(points):
    """Init an absolute normal noise variable."""
    return np.abs(np.random.rand(points, 1) * np.random.rand(1))

def laplace_noise(points):
    """Init a Laplace noise variable."""
    lambda_ = np.random.rand(1)
    return np.random.laplace(0, lambda_, (points, 1))

def uniform_noise(points):
    """Init a uniform noise variable."""
    return np.random.rand(1) * np.random.uniform(size=(points, 1)) \
        + random.sample([2, -2], 1)

class NormalCause(object):
    def __init__(self, mean=0, std=1, std_min=None, std_max=None):
        self.mean = mean
        if std_min is None and std_max is None:
            self.std = std
        else:
            self.std = np.random.uniform(std_min, std_max)

    def __call__(self, points):
        return np.random.normal(self.mean, self.std, \
                                size=(points))

class UniformCause(object):
    def __init__(self, _min=-1, _max=1):
        self._min = _min
        self._max = _max

    def __call__(self, points):
        return np.random.uniform(self._min, self._max, size=(points))


class nn_noise(object):
    def __init__(self, noise=variable_normal_noise, n_hidden=20):
        """Init the mechanism."""
        super(nn_noise, self).__init__()
        self.noise = noise
        self.n_hidden = n_hidden
        self.initialize_nn()

    def initialize_nn(self):
        layers = []

        layers.append(th.nn.modules.Linear(1, self.n_hidden))
        layers.append(th.nn.Tanh())
        layers.append(th.nn.modules.Linear(self.n_hidden, 1))

        self.layers = th.nn.Sequential(*layers)
        # use a normal initialization
        # self.layers.apply(self.weight_init)

    def weight_init(self, model):
        if isinstance(model, th.nn.modules.Linear):
            th.nn.init.normal_(model.weight.data, mean=0., std=0.5)

    def __call__(self, points):
        x = self.noise(points)
        data = x.astype('float32')
        data = th.from_numpy(data)
        data = self.layers(data).data.numpy()
        return data
