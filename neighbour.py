import numpy as np
from scipy.optimize import LinearConstraint, Bounds
from scipy.special import logsumexp
from scipy.stats import poisson, bernoulli
from statsmodels.base.model import GenericLikelihoodModel


class Neighbour(GenericLikelihoodModel):
    def __init__(self, n_i, **kwds):
        self.df_model = np.nan
        self.df_resid = np.nan
        self.g = np.nan
        n_i_unique, n_i_counts = np.unique(n_i,
                                           return_counts=True)
        n_i_hist = np.zeros(n_i_unique[-1] + 1, dtype=np.int_)
        n_i_hist[n_i_unique] = n_i_counts
        super(Neighbour, self).__init__(n_i_hist, **kwds)

    def nloglikeobs(self, params):
        n_i_hist = self.endog
        g = self.g
        n_i_n = len(n_i_hist)
        n = sum(n_i_hist)
        alphas = np.clip(params[:g], 0, 1)
        ps = np.clip(params[g:(g * 2)], 0, 1)
        lambdas = np.clip(params[(g * 2):], 0, n - 1)
        n_i_pos = np.column_stack([np.arange(1, n_i_n)] * g)
        ll_matrix = np.zeros((n_i_n, g))
        ll_matrix[0] = poisson.logpmf(0, lambdas) + np.nan_to_num(bernoulli.logpmf(0, ps), nan=np.nan)
        ll_matrix[1:] = poisson.logpmf(n_i_pos - 1, lambdas) + np.logaddexp(
            bernoulli.logpmf(0, ps) + np.log(lambdas) - np.log(n_i_pos),
            bernoulli.logpmf(1, ps))
        nll_obs = -(logsumexp(ll_matrix, axis=1, b=alphas) * n_i_hist)
        return nll_obs

    def fit(self, start_params=None, g=1, **kwds):
        param_names = ('alpha', 'p', 'lambda')
        if start_params is None:
            start_params = np.concatenate(
                (np.full(g, 1 / g),
                 np.full((len(param_names) - 1) * g, 0.5)))
        g = len(start_params) // len(param_names)
        start_params = start_params[:(len(param_names) * g)]
        n_i_hist = self.endog
        n_i_n = len(n_i_hist)
        n = sum(n_i_hist)
        self.data.xnames = [param_name + str(param_level)
                            for param_name in param_names
                            for param_level in range(g)]
        self.df_model = len(start_params)
        self.df_resid = n_i_n - self.df_model
        self.g = g
        alpha_constraint = LinearConstraint(
            np.concatenate(
                (np.ones((1, g), dtype=np.int_),
                 np.zeros((1, (len(param_names) - 1) * g), dtype=np.int_)),
                axis=1),
            1, 1)
        param_bounds = Bounds(
            np.zeros(len(param_names) * g, dtype=np.int_),
            np.concatenate(
                (np.ones((len(param_names) - 1) * g, dtype=np.int_),
                 np.full(g, n - 1, dtype=np.int_))))
        return super(Neighbour, self).fit(start_params=start_params,
                                          method='minimize',
                                          min_method='trust-constr',
                                          constraints=alpha_constraint,
                                          bounds=param_bounds,
                                          verbose=2,
                                          **kwds)


if __name__ == '__main__':
    np.seterr(all='warn')
    records_n_i = np.ones(10000, dtype=np.int_)
    neighbour_model = Neighbour(records_n_i)
    neighbour_model_fit = neighbour_model.fit()
    print(neighbour_model_fit.summary())
