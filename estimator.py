import numpy as np
from scipy.optimize import LinearConstraint, Bounds
from scipy.stats import poisson, bernoulli
from statsmodels.base.model import GenericLikelihoodModel


class Neighbour(GenericLikelihoodModel):
    def __init__(self, endog, **kwds):
        self.df_model = np.nan
        self.df_resid = np.nan
        super(Neighbour, self).__init__(endog, **kwds)

    def nloglikeobs(self, params):
        n_i = self.endog
        n = len(n_i)
        g = len(params) // 3
        alphas = np.clip(params[:g], 0, 1)
        ps = np.clip(params[g:(g * 2)], 0, 1)
        lambdas = np.clip(params[(g * 2):], 0, n - 1)
        ll_matrix = np.zeros((n, g))
        ll_matrix[n_i < 1] = poisson.logpmf(0, lambdas) + bernoulli.logpmf(0, ps)
        n_i_pos = np.column_stack([n_i[n_i > 0]] * g)
        ll_matrix[n_i > 0] = poisson.logpmf(n_i_pos - 1, lambdas) + np.logaddexp(
            bernoulli.logpmf(0, ps) + np.log(lambdas) - np.log(n_i_pos),
            bernoulli.logpmf(1, ps))
        nll_obs = -np.log(np.exp(ll_matrix) @ alphas)
        return nll_obs

    def fit(self, start_params=None, **kwds):
        if start_params is None:
            start_params = np.zeros(3)
        g = len(start_params) // 3
        start_params = start_params[:(g * 3)]
        self.data.xnames = [param + str(level)
                            for param in ('alpha', 'p', 'lambda')
                            for level in range(g)]
        self.df_model = len(start_params)
        self.df_resid = len(self.endog) - self.df_model
        return super(Neighbour, self).fit(start_params=start_params, **kwds)


if __name__ == '__main__':
    np.seterr(all='warn')
    n_is = np.ones(10000, dtype=np.int_)
    constraint = LinearConstraint(np.concatenate((np.ones((1, 1), dtype=np.int_),
                                                  np.zeros((1, 2), dtype=np.int_)),
                                                 axis=1),
                                  1, 1)
    bounds = Bounds(np.zeros(3, dtype=np.int_),
                    np.concatenate((np.ones(2, dtype=np.int_), np.array([9999], dtype=np.int_))))
    neighbour_model_fit = Neighbour(n_is).fit(np.array([1, 0.5, 0.5]),
                                              method='minimize',
                                              min_method='trust-constr',
                                              constraints=constraint,
                                              bounds=bounds,
                                              verbose=2)
    print(neighbour_model_fit.summary())
