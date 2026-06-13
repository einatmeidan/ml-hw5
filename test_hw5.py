"""
Test suite for hw5.py.

Run with:  pytest test_hw5.py -v

Six tests are provided for every function/class required by the assignment.
Tests rely only on numpy, scipy (for ground-truth reference values) and the
Python standard library, and do not read or write any files.
"""

import statistics

import numpy as np
import pytest
import scipy.stats

from hw5 import (
    poisson_log_pmf,
    poission_analytic_mle,
    poission_confidence_interval,
    get_poisson_log_likelihoods,
    conditional_independence,
    normal_pdf,
    NaiveNormalClassDistribution,
    MAPClassifier,
    multi_normal_pdf,
    MultiNormalClassDistribution,
    compute_accuracy,
    DiscreteNBClassDistribution,
)


# ---------------------------------------------------------------------------
# Shared synthetic datasets (2 well separated clusters + a 3-feature variant)
# ---------------------------------------------------------------------------

DATASET_2F = np.array([
    [1.0, 10.0, 0],
    [2.0, 20.0, 0],
    [3.0, 12.0, 0],
    [10.0, 100.0, 1],
    [12.0, 110.0, 1],
    [14.0, 90.0, 1],
])

DATASET_3F = np.array([
    [1.0, 10.0, 5.0, 0],
    [2.0, 20.0, 7.0, 0],
    [3.0, 12.0, 6.0, 0],
    [1.5, 15.0, 5.5, 0],
    [2.5, 18.0, 6.5, 0],
    [10.0, 100.0, 50.0, 1],
    [12.0, 110.0, 55.0, 1],
    [14.0, 90.0, 45.0, 1],
    [11.0, 95.0, 48.0, 1],
    [13.0, 105.0, 52.0, 1],
])

DISCRETE_DATA = np.array([
    [0, 5, 0],
    [1, 5, 0],
    [1, 5, 0],
    [2, 7, 1],
    [2, 7, 1],
    [0, 7, 1],
])

DISCRETE_DATA_3F = np.array([
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
])

D_GOALS = [3, 3, 3, 6, 4, 7, 5, 1, 3, 1, 2, 2, 6, 0, 2, 1, 5, 3, 6, 4, 4, 4,
           5, 5, 2, 8, 3, 6, 5, 4, 4, 5, 5, 2, 3, 4, 0, 3, 2, 6, 3, 4, 1, 4,
           4, 2, 5, 1, 3, 2]


# ---------------------------------------------------------------------------
# 1. poisson_log_pmf
# ---------------------------------------------------------------------------

class TestPoissonLogPmf:
    def test_scalar_zero(self):
        assert poisson_log_pmf(0, 1.0) == pytest.approx(scipy.stats.poisson.logpmf(0, 1.0))

    def test_scalar_various(self):
        for k, rate in [(3, 2.0), (5, 4.5), (1, 0.5)]:
            assert poisson_log_pmf(k, rate) == pytest.approx(scipy.stats.poisson.logpmf(k, rate))

    def test_array_input_matches_scipy(self):
        k = np.array([0, 1, 2, 3, 4, 5])
        rate = 2.5
        result = poisson_log_pmf(k, rate)
        expected = scipy.stats.poisson.logpmf(k, rate)
        np.testing.assert_allclose(result, expected)

    def test_array_output_shape(self):
        k = np.array([0, 1, 2])
        result = poisson_log_pmf(k, 3.0)
        assert len(result) == 3

    def test_large_k(self):
        k, rate = 50, 10.0
        assert poisson_log_pmf(k, rate) == pytest.approx(scipy.stats.poisson.logpmf(k, rate), rel=1e-6)

    def test_multiple_rates_consistency(self):
        k = np.array([0, 1, 2, 3, 4])
        for rate in [0.5, 1.0, 3.0, 7.5]:
            result = poisson_log_pmf(k, rate)
            expected = scipy.stats.poisson.logpmf(k, rate)
            np.testing.assert_allclose(result, expected, rtol=1e-8)


# ---------------------------------------------------------------------------
# 2. poission_analytic_mle
# ---------------------------------------------------------------------------

class TestPoissonAnalyticMle:
    def test_simple_mean(self):
        assert poission_analytic_mle([1, 2, 3, 4, 5]) == pytest.approx(3.0)

    def test_constant_samples(self):
        assert poission_analytic_mle([4] * 10) == pytest.approx(4.0)

    def test_single_sample(self):
        assert poission_analytic_mle([7]) == pytest.approx(7.0)

    def test_matches_numpy_mean(self):
        rng = np.random.default_rng(0)
        samples = rng.poisson(lam=4, size=200)
        assert poission_analytic_mle(samples) == pytest.approx(np.mean(samples))

    def test_includes_zero(self):
        assert poission_analytic_mle([0, 0, 1, 2, 3]) == pytest.approx(1.2)

    def test_known_dataset(self):
        assert poission_analytic_mle(D_GOALS) == pytest.approx(sum(D_GOALS) / len(D_GOALS))


# ---------------------------------------------------------------------------
# 3. poission_confidence_interval
# ---------------------------------------------------------------------------

class TestPoissonConfidenceInterval:
    def test_default_alpha_matches_manual_formula(self):
        n = len(D_GOALS)
        lam = sum(D_GOALS) / n
        lower, upper = poission_confidence_interval(lam, n)
        z = statistics.NormalDist().inv_cdf(0.975)
        margin = np.sqrt(lam / n) * z
        assert lower == pytest.approx(lam - margin)
        assert upper == pytest.approx(lam + margin)

    def test_symmetric_around_mle(self):
        lower, upper = poission_confidence_interval(5.0, 100, alpha=0.05)
        assert (lower + upper) / 2 == pytest.approx(5.0)

    def test_alpha_001_wider_than_alpha_005(self):
        l05, u05 = poission_confidence_interval(5.0, 100, alpha=0.05)
        l01, u01 = poission_confidence_interval(5.0, 100, alpha=0.01)
        assert (u01 - l01) > (u05 - l05)

    def test_larger_n_gives_narrower_interval(self):
        l1, u1 = poission_confidence_interval(5.0, 50)
        l2, u2 = poission_confidence_interval(5.0, 500)
        assert (u2 - l2) < (u1 - l1)

    def test_lower_less_than_upper(self):
        lower, upper = poission_confidence_interval(2.0, 30)
        assert lower < upper

    def test_lambda_zero_gives_degenerate_interval(self):
        lower, upper = poission_confidence_interval(0.0, 50)
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. get_poisson_log_likelihoods
# ---------------------------------------------------------------------------

class TestGetPoissonLogLikelihoods:
    def test_matches_sum_of_log_pmf_single_rate(self):
        samples = [1, 2, 3, 4, 5]
        ll = get_poisson_log_likelihoods(samples, [2.0])
        expected = np.sum(poisson_log_pmf(np.array(samples), 2.0))
        assert ll[0] == pytest.approx(expected)

    def test_output_length_matches_rates(self):
        rates = np.linspace(0.1, 5, 20)
        ll = get_poisson_log_likelihoods([1, 2, 3], rates)
        assert len(ll) == 20

    def test_maximum_near_mle(self):
        rates = np.linspace(0.01, 10, 1000)
        ll = get_poisson_log_likelihoods(D_GOALS, rates)
        mle = poission_analytic_mle(D_GOALS)
        best_rate = rates[np.argmax(ll)]
        assert best_rate == pytest.approx(mle, abs=0.05)

    def test_matches_scipy_sum_for_multiple_rates(self):
        samples = [0, 1, 2, 3, 4]
        rates = [1.0, 2.0, 3.0]
        ll = get_poisson_log_likelihoods(samples, rates)
        for i, r in enumerate(rates):
            expected = np.sum(scipy.stats.poisson.logpmf(samples, r))
            assert ll[i] == pytest.approx(expected)

    def test_single_rate_array(self):
        samples = [2, 2, 2]
        ll = get_poisson_log_likelihoods(samples, [2.0])
        expected = np.sum(scipy.stats.poisson.logpmf(samples, 2.0))
        assert ll[0] == pytest.approx(expected)

    def test_peak_at_sample_mean(self):
        samples = [5] * 10
        rates = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        ll = get_poisson_log_likelihoods(samples, rates)
        assert np.argmax(ll) == 2


# ---------------------------------------------------------------------------
# 5. conditional_independence
# ---------------------------------------------------------------------------

class TestConditionalIndependence:
    @pytest.fixture
    def ci(self):
        return conditional_independence()

    def test_all_distributions_valid(self, ci):
        for d in [ci.X, ci.Y, ci.C, ci.X_Y, ci.X_C, ci.Y_C, ci.X_Y_C]:
            vals = np.array(list(d.values()), dtype=float)
            assert np.all(vals >= 0)
            assert sum(vals) == pytest.approx(1.0)

    def test_X_marginal_consistent_with_joint(self, ci):
        for x in [0, 1]:
            total = sum(p for (xx, yy, cc), p in ci.X_Y_C.items() if xx == x)
            assert total == pytest.approx(ci.X[x])

    def test_X_Y_consistent_with_joint(self, ci):
        for (x, y), p in ci.X_Y.items():
            total = sum(pp for (xx, yy, cc), pp in ci.X_Y_C.items() if xx == x and yy == y)
            assert total == pytest.approx(p)

    def test_X_C_and_Y_C_consistent_with_joint(self, ci):
        for (x, c), p in ci.X_C.items():
            total = sum(pp for (xx, yy, cc), pp in ci.X_Y_C.items() if xx == x and cc == c)
            assert total == pytest.approx(p)
        for (y, c), p in ci.Y_C.items():
            total = sum(pp for (xx, yy, cc), pp in ci.X_Y_C.items() if yy == y and cc == c)
            assert total == pytest.approx(p)

    def test_X_and_Y_are_dependent(self, ci):
        assert ci.is_X_Y_dependent() is True

    def test_X_and_Y_independent_given_C(self, ci):
        assert ci.is_X_Y_given_C_independent() is True


# ---------------------------------------------------------------------------
# 6. normal_pdf
# ---------------------------------------------------------------------------

class TestNormalPdf:
    def test_standard_normal_at_zero(self):
        assert normal_pdf(0, 0, 1) == pytest.approx(1 / np.sqrt(2 * np.pi))

    def test_matches_scipy_various_parameters(self):
        for x, mu, sigma in [(1, 0, 1), (2, 1, 0.5), (-3, 2, 2), (0.5, 0.5, 3)]:
            assert normal_pdf(x, mu, sigma) == pytest.approx(scipy.stats.norm.pdf(x, mu, sigma))

    def test_symmetric_around_mean(self):
        mu, sigma = 5, 2
        assert normal_pdf(mu + 1, mu, sigma) == pytest.approx(normal_pdf(mu - 1, mu, sigma))

    def test_peak_is_at_mean(self):
        mu, sigma = 3, 1
        peak = normal_pdf(mu, mu, sigma)
        for d in [0.5, 1, 2, 5]:
            assert normal_pdf(mu + d, mu, sigma) < peak

    def test_larger_std_gives_lower_peak(self):
        assert normal_pdf(0, 0, 2) < normal_pdf(0, 0, 1)

    def test_array_input_matches_scipy(self):
        x = np.array([-1, 0, 1])
        result = np.asarray(normal_pdf(x, 0, 1), dtype=float)
        expected = scipy.stats.norm.pdf(x, 0, 1)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# 7. NaiveNormalClassDistribution
# ---------------------------------------------------------------------------

class TestNaiveNormalClassDistribution:
    def test_prior_class0(self):
        cd = NaiveNormalClassDistribution(0)
        cd.fit(DATASET_2F)
        assert cd.get_prior() == pytest.approx(0.5)

    def test_prior_class1(self):
        cd = NaiveNormalClassDistribution(1)
        cd.fit(DATASET_2F)
        assert cd.get_prior() == pytest.approx(0.5)

    def test_likelihood_peaks_near_class_mean(self):
        cd = NaiveNormalClassDistribution(0)
        cd.fit(DATASET_2F)
        rows = DATASET_2F[DATASET_2F[:, -1] == 0, :-1]
        x_at_mean = rows.mean(axis=0)
        x_far_away = x_at_mean + 100
        assert cd.get_instance_likelihood(x_at_mean) > cd.get_instance_likelihood(x_far_away)

    def test_likelihood_matches_normal_pdf_product(self):
        cd = NaiveNormalClassDistribution(0)
        cd.fit(DATASET_2F)
        rows = DATASET_2F[DATASET_2F[:, -1] == 0, :-1]
        means = rows.mean(axis=0)
        stds = rows.std(axis=0)
        x = np.array([2.0, 15.0])
        expected = normal_pdf(x[0], means[0], stds[0]) * normal_pdf(x[1], means[1], stds[1])
        assert cd.get_instance_likelihood(x) == pytest.approx(expected)

    def test_joint_prob_equals_prior_times_likelihood(self):
        cd = NaiveNormalClassDistribution(1)
        cd.fit(DATASET_2F)
        x = np.array([11.0, 105.0])
        expected = cd.get_prior() * cd.get_instance_likelihood(x)
        assert cd.get_instance_joint_prob(x) == pytest.approx(expected)

    def test_works_with_more_than_two_features(self):
        cd = NaiveNormalClassDistribution(1)
        cd.fit(DATASET_3F)
        x = np.array([12.0, 100.0, 50.0])
        assert cd.get_instance_likelihood(x) > 0


# ---------------------------------------------------------------------------
# 8. MAPClassifier
# ---------------------------------------------------------------------------

class _MockCD:
    """Minimal stand-in for a ClassDistribution with a fixed joint probability."""

    def __init__(self, joint_prob):
        self._jp = joint_prob

    def get_instance_joint_prob(self, x):
        return self._jp


class TestMAPClassifier:
    def test_predicts_class0_when_higher(self):
        clf = MAPClassifier(_MockCD(0.7), _MockCD(0.3))
        assert clf.predict(np.array([0, 0])) == 0

    def test_predicts_class1_when_higher(self):
        clf = MAPClassifier(_MockCD(0.2), _MockCD(0.8))
        assert clf.predict(np.array([0, 0])) == 1

    def test_tie_predicts_class1(self):
        clf = MAPClassifier(_MockCD(0.5), _MockCD(0.5))
        assert clf.predict(np.array([0, 0])) == 1

    def test_with_real_distributions_class0_point(self):
        cd0 = NaiveNormalClassDistribution(0)
        cd0.fit(DATASET_2F)
        cd1 = NaiveNormalClassDistribution(1)
        cd1.fit(DATASET_2F)
        clf = MAPClassifier(cd0, cd1)
        assert clf.predict(np.array([1.5, 11.0])) == 0

    def test_with_real_distributions_class1_point(self):
        cd0 = NaiveNormalClassDistribution(0)
        cd0.fit(DATASET_2F)
        cd1 = NaiveNormalClassDistribution(1)
        cd1.fit(DATASET_2F)
        clf = MAPClassifier(cd0, cd1)
        assert clf.predict(np.array([12.0, 100.0])) == 1

    def test_predict_consistent_with_manual_argmax(self):
        cd0 = NaiveNormalClassDistribution(0)
        cd0.fit(DATASET_2F)
        cd1 = NaiveNormalClassDistribution(1)
        cd1.fit(DATASET_2F)
        clf = MAPClassifier(cd0, cd1)
        for x in DATASET_2F[:, :-1]:
            jp0 = cd0.get_instance_joint_prob(x)
            jp1 = cd1.get_instance_joint_prob(x)
            expected = 0 if jp0 > jp1 else 1
            assert clf.predict(x) == expected


# ---------------------------------------------------------------------------
# 9. multi_normal_pdf
# ---------------------------------------------------------------------------

class TestMultiNormalPdf:
    def test_matches_scipy_2d_identity_cov(self):
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        x = np.array([1.0, 1.0])
        expected = scipy.stats.multivariate_normal.pdf(x, mean, cov)
        assert multi_normal_pdf(x, mean, cov) == pytest.approx(expected)

    def test_matches_scipy_with_correlation(self):
        mean = np.array([1.0, 2.0])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        x = np.array([0.0, 1.0])
        expected = scipy.stats.multivariate_normal.pdf(x, mean, cov)
        assert multi_normal_pdf(x, mean, cov) == pytest.approx(expected)

    def test_matches_scipy_3d(self):
        mean = np.array([0.0, 0.0, 0.0])
        cov = np.eye(3) * 2
        x = np.array([1.0, -1.0, 0.5])
        expected = scipy.stats.multivariate_normal.pdf(x, mean, cov)
        assert multi_normal_pdf(x, mean, cov) == pytest.approx(expected)

    def test_value_at_mean_with_identity_cov(self):
        mean = np.zeros(2)
        cov = np.eye(2)
        assert multi_normal_pdf(mean, mean, cov) == pytest.approx((2 * np.pi) ** -1)

    def test_diagonal_cov_matches_univariate_product(self):
        mean = np.array([1.0, 2.0])
        cov = np.array([[4.0, 0.0], [0.0, 9.0]])
        x = np.array([2.0, 5.0])
        expected = scipy.stats.norm.pdf(x[0], 1.0, 2.0) * scipy.stats.norm.pdf(x[1], 2.0, 3.0)
        assert multi_normal_pdf(x, mean, cov) == pytest.approx(expected)

    def test_single_dimension_matches_normal_pdf(self):
        mean = np.array([3.0])
        cov = np.array([[4.0]])
        x = np.array([5.0])
        expected = scipy.stats.norm.pdf(5.0, 3.0, 2.0)
        assert multi_normal_pdf(x, mean, cov) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 10. MultiNormalClassDistribution
# ---------------------------------------------------------------------------

class TestMultiNormalClassDistribution:
    def test_prior(self):
        cd = MultiNormalClassDistribution(0)
        cd.fit(DATASET_2F)
        assert cd.get_prior() == pytest.approx(0.5)

    def test_mean_matches_sample_mean(self):
        cd = MultiNormalClassDistribution(1)
        cd.fit(DATASET_2F)
        rows = DATASET_2F[DATASET_2F[:, -1] == 1, :-1]
        np.testing.assert_allclose(np.asarray(cd.mean, dtype=float), rows.mean(axis=0))

    def test_cov_matches_sample_covariance(self):
        cd = MultiNormalClassDistribution(0)
        cd.fit(DATASET_2F)
        rows = DATASET_2F[DATASET_2F[:, -1] == 0, :-1]
        expected_cov = np.cov(rows, rowvar=False, bias=True)
        np.testing.assert_allclose(np.asarray(cd.cov, dtype=float), expected_cov)

    def test_likelihood_matches_multi_normal_pdf(self):
        cd = MultiNormalClassDistribution(0)
        cd.fit(DATASET_2F)
        x = np.array([2.0, 15.0])
        expected = multi_normal_pdf(x, cd.mean, cd.cov)
        assert cd.get_instance_likelihood(x) == pytest.approx(expected)

    def test_joint_prob_equals_prior_times_likelihood(self):
        cd = MultiNormalClassDistribution(1)
        cd.fit(DATASET_2F)
        x = np.array([11.0, 105.0])
        expected = cd.get_prior() * cd.get_instance_likelihood(x)
        assert cd.get_instance_joint_prob(x) == pytest.approx(expected)

    def test_works_with_more_than_two_features(self):
        cd = MultiNormalClassDistribution(0)
        cd.fit(DATASET_3F)
        x = np.array([2.0, 15.0, 6.0])
        assert cd.get_instance_likelihood(x) > 0


# ---------------------------------------------------------------------------
# 11. compute_accuracy
# ---------------------------------------------------------------------------

class _AlwaysPredict:
    """Classifier stub that ignores its input and always predicts a fixed label."""

    def __init__(self, label):
        self.label = label

    def predict(self, x):
        return self.label


class TestComputeAccuracy:
    def test_constant_predictor_half_correct_label0(self):
        test_set = np.array([[1, 2, 0], [3, 4, 1], [5, 6, 0], [7, 8, 1]])
        assert compute_accuracy(test_set, _AlwaysPredict(0)) == pytest.approx(0.5)

    def test_constant_predictor_half_correct_label1(self):
        test_set = np.array([[1, 2, 0], [3, 4, 1], [5, 6, 0], [7, 8, 1]])
        assert compute_accuracy(test_set, _AlwaysPredict(1)) == pytest.approx(0.5)

    def test_constant_predictor_three_quarters_correct(self):
        test_set = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]])
        assert compute_accuracy(test_set, _AlwaysPredict(0)) == pytest.approx(0.75)

    def test_perfect_accuracy(self):
        test_set = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        assert compute_accuracy(test_set, _AlwaysPredict(1)) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        test_set = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        assert compute_accuracy(test_set, _AlwaysPredict(0)) == pytest.approx(0.0)

    def test_with_real_classifier_on_separated_clusters(self):
        cd0 = NaiveNormalClassDistribution(0)
        cd0.fit(DATASET_2F)
        cd1 = NaiveNormalClassDistribution(1)
        cd1.fit(DATASET_2F)
        clf = MAPClassifier(cd0, cd1)
        # The two classes are well separated, so the model should
        # classify (most of) its own training data correctly.
        assert compute_accuracy(DATASET_2F, clf) >= 0.8


# ---------------------------------------------------------------------------
# 12. DiscreteNBClassDistribution
# ---------------------------------------------------------------------------

class TestDiscreteNBClassDistribution:
    def test_prior_both_classes(self):
        cd0 = DiscreteNBClassDistribution(0)
        cd0.fit(DISCRETE_DATA)
        cd1 = DiscreteNBClassDistribution(1)
        cd1.fit(DISCRETE_DATA)
        assert cd0.get_prior() == pytest.approx(0.5)
        assert cd1.get_prior() == pytest.approx(0.5)

    def test_likelihood_class0_observed_values(self):
        cd = DiscreteNBClassDistribution(0)
        cd.fit(DISCRETE_DATA)
        # P(X0=0|Y=0)=(1+1)/(3+3)=1/3 ; P(X1=5|Y=0)=(3+1)/(3+2)=4/5
        assert cd.get_instance_likelihood(np.array([0, 5])) == pytest.approx((1 / 3) * (4 / 5))

    def test_likelihood_class0_laplace_smoothing_for_unseen_value(self):
        cd = DiscreteNBClassDistribution(0)
        cd.fit(DISCRETE_DATA)
        # value 2 never occurs for feature 0 within class 0
        # P(X0=2|Y=0)=(0+1)/(3+3)=1/6 ; P(X1=5|Y=0)=4/5
        assert cd.get_instance_likelihood(np.array([2, 5])) == pytest.approx((1 / 6) * (4 / 5))

    def test_likelihood_class1_observed_values(self):
        cd = DiscreteNBClassDistribution(1)
        cd.fit(DISCRETE_DATA)
        # P(X0=2|Y=1)=(2+1)/(3+3)=1/2 ; P(X1=7|Y=1)=(3+1)/(3+2)=4/5
        assert cd.get_instance_likelihood(np.array([2, 7])) == pytest.approx((1 / 2) * (4 / 5))

    def test_joint_prob_equals_prior_times_likelihood(self):
        cd = DiscreteNBClassDistribution(0)
        cd.fit(DISCRETE_DATA)
        x = np.array([0, 5])
        expected = cd.get_prior() * cd.get_instance_likelihood(x)
        assert cd.get_instance_joint_prob(x) == pytest.approx(expected)

    def test_works_with_more_than_two_features(self):
        cd = DiscreteNBClassDistribution(1)
        cd.fit(DISCRETE_DATA_3F)
        # class 1 rows: [1,0,0] and [1,1,1] -> n_1=2, |A_j|=2 for all features
        # P(X0=1|Y=1)=(2+1)/(2+2)=3/4 ; P(X1=0|Y=1)=(1+1)/4=1/2 ; P(X2=1|Y=1)=(1+1)/4=1/2
        expected = (3 / 4) * (1 / 2) * (1 / 2)
        assert cd.get_instance_likelihood(np.array([1, 0, 1])) == pytest.approx(expected)
