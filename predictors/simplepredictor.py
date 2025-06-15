"""
Simple preditor for kicktipp bet bot.
"""
from helper.match import Match
from .base import PredictorBase
import math


class SimplePredictor(PredictorBase):
    DOMINATION_THRESHOLD = 6
    DRAW_THRESHOLD = 1.2

    def predict(self, match: Match, win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points, quote_points):

        diff = math.fabs(match.rate_home - match.rate_road)
        home_wins = match.rate_home < match.rate_road

        if diff < self.DRAW_THRESHOLD:
            return (1, 1)

        if diff >= self.DOMINATION_THRESHOLD:
            result = (3, 1)
        elif diff >= self.DOMINATION_THRESHOLD / 2:
            result = (2, 1)
        else:
            result = (1, 0)

        return result if home_wins else tuple(reversed(result))
