from .base import PredictorBase
from scipy.optimize import minimize
from scipy.stats import poisson
import numpy as np
import penaltyblog as pb

class AwesomePredictor(PredictorBase):
    MAX_GOALS = 15

    def calculate_probabilities(self, rate_home, rate_deuce, rate_road):
        return pb.implied.power([rate_home, rate_deuce, rate_road])["implied_probabilities"]
    
    def _mse(self, params, home, draw, away):
        exp_params = np.exp(params)

        mu1 = poisson.pmf(np.arange(0, self.MAX_GOALS+1), exp_params[0])
        mu2 = poisson.pmf(np.arange(0, self.MAX_GOALS+1), exp_params[1])

        mat = np.outer(mu1, mu2)

        pred = np.array([
            np.sum(np.tril(mat, -1)),  # home
            np.sum(np.diag(mat)),  # draw
            np.sum(np.triu(mat, 1))  # away
        ])

        obs = np.array([home, draw, away])

        mse = np.mean((pred - obs) ** 2)

        return mse

    def goal_expectation(self, home, draw, away):
        options = {
            "maxiter": 1000,
            "disp": False,
        }

        res = minimize(
            fun=self._mse,
            x0=[0.5, -0.5],  # Initial guess for log(lambda) values
            args=(home, draw, away),
            options=options,
        )

        output = {
            "home_exp": np.exp(res["x"][0]),
            "away_exp": np.exp(res["x"][1]),
            "error": res["fun"],
            "success": res["success"],
        }

        return output["home_exp"], output["away_exp"]

    def poisson_prob(self, lamb, k):
        """Calculate Poisson probability for k goals given lambda (expected goals)."""
        return poisson.pmf(k, lamb)

    def calculate_reward(self, predicted, actual):
        """Calculate the reward based on predicted and actual results."""
        if predicted == actual:
            return 4
        pred_home, pred_away = predicted
        act_home, act_away = actual

        if (pred_home - pred_away) == (act_home - act_away):
            return 2 if act_home == act_away else 3
        if (pred_home > pred_away and act_home > act_away) or (pred_home < pred_away and act_home < act_away):
            return 2
        return 0

    def predict(self, match):
        win_prob, draw_prob, lose_prob = self.calculate_probabilities(match.rate_home, match.rate_deuce, match.rate_road)
        expected_goals_home, expected_goals_away = self.goal_expectation(win_prob, draw_prob, lose_prob)

        # Calculate probabilities for each goal count from 0 to MAX_GOALS
        home_goal_probs = [self.poisson_prob(expected_goals_home, i) for i in range(self.MAX_GOALS + 1)]
        away_goal_probs = [self.poisson_prob(expected_goals_away, i) for i in range(self.MAX_GOALS + 1)]

        # Calculate expected rewards for each possible outcome
        best_prediction = (0, 0)
        best_expected_reward = 0

        for home_goals in range(self.MAX_GOALS + 1):
            for away_goals in range(self.MAX_GOALS + 1):
                predicted_result = (home_goals, away_goals)
                expected_reward = 0

                for actual_home_goals in range(self.MAX_GOALS + 1):
                    for actual_away_goals in range(self.MAX_GOALS + 1):
                        actual_result = (actual_home_goals, actual_away_goals)
                        reward = self.calculate_reward(predicted_result, actual_result)
                        probability = home_goal_probs[actual_home_goals] * away_goal_probs[actual_away_goals]
                        expected_reward += reward * probability

                if expected_reward >= best_expected_reward:
                    best_expected_reward = expected_reward
                    best_prediction = predicted_result

        return best_prediction
