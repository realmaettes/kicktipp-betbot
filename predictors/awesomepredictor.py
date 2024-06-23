import math
import numpy as np
import pandas as pd
from scipy.stats import poisson
from .base import PredictorBase
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import glob
import os

class AwesomePredictor(PredictorBase):
    MAX_GOALS = 0

    def __init__(self, historical_data_path='data'):
        # Load historical data and fit linear regression models
        self.historical_data_path = historical_data_path
        self.model = self.fit_linear_regression()

    def read_csv_files(self):
        # List all CSV files in the specified folder and subfolders
        csv_files = glob.glob(os.path.join(self.historical_data_path, '**', '*.csv'), recursive=True)

        # List of relevant columns
        required_columns = [
            'FTHG', 'FTAG',
            'BWH', 'BWD', 'BWA'
        ]
    
        # Read and concatenate all CSV files into a single DataFrame
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                # Filter the relevant columns
                df = df[required_columns]
                # Check if all necessary columns are present and have no missing values
                if all(col in df.columns for col in required_columns):
                    df = df.dropna(subset=required_columns)
                    if not df.empty:
                        dataframes.append(df)
            except Exception as e:
                os.remove(file)

        combined_df = pd.concat(dataframes, ignore_index=True)
    
        # Drop rows with NaN values
        combined_df.dropna(inplace=True)
    
        return combined_df

    def fit_linear_regression(self):
        """Fit multi-output linear regression model for home and away goals based on historical data."""
        df = self.read_csv_files()

        # Transform data
        def transform_data(row):
            matches = []
            # Calculate probabilities
            win_prob, draw_prob, lose_prob = self.calculate_probabilities(row['BWH'],row['BWD'],row['BWA'])
            matches.append({
                'home_goals': row['FTHG'],
                'away_goals': row['FTAG'],
                'win_prob': win_prob,
                'draw_prob': draw_prob,
                'lose_prob': lose_prob,
            })
            return matches

        # Split data into individual training samples
        expanded_rows = []
        for _, row in df.iterrows():
            expanded_rows.extend(transform_data(row))

        # Create a new DataFrame
        expanded_df = pd.DataFrame(expanded_rows)

        # Features and targets
        X = expanded_df[['win_prob', 'draw_prob', 'lose_prob']]
        y = expanded_df[['home_goals', 'away_goals']]

        self.MAX_GOALS = math.ceil(max(y['home_goals'].max(), y['away_goals'].max()))
        
        # Fit multi-output linear regression model
        model = MultiOutputRegressor(LinearRegression()).fit(X.values, y)
        
        return model

    def poisson_prob(self, lamb, k):
        """Calculate Poisson probability for k goals given lambda (expected goals)."""
        return poisson.pmf(k, lamb)
    
    def calculate_probabilities(self, rate_home, rate_deuce, rate_road):
        win_prob = (1 / rate_home) / ((1 / rate_home) + (1 / rate_deuce) + (1 / rate_road))
        draw_prob = (1 / rate_deuce) / ((1 / rate_home) + (1 / rate_deuce) + (1 / rate_road))
        lose_prob = (1 / rate_road) / ((1 / rate_home) + (1 / rate_deuce) + (1 / rate_road))
        return win_prob, draw_prob, lose_prob

    def calculate_expected_goals(self, win_prob, draw_prob, lose_prob):
        """Calculate the expected goals for home and away teams based on probabilities using linear regression."""
        X = np.array([[win_prob, draw_prob, lose_prob]])
        expected_goals = self.model.predict(X)[0]
        return expected_goals[0], expected_goals[1]

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
        expected_goals_home, expected_goals_road = self.calculate_expected_goals(win_prob, draw_prob, lose_prob)

        # Calculate probabilities for each goal count from 0 to MAX_GOALS
        home_goal_probs = [self.poisson_prob(expected_goals_home, i) for i in range(self.MAX_GOALS + 1)]
        road_goal_probs = [self.poisson_prob(expected_goals_road, i) for i in range(self.MAX_GOALS + 1)]

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
                        probability = home_goal_probs[actual_home_goals] * road_goal_probs[actual_away_goals]
                        expected_reward += reward * probability

                if expected_reward >= best_expected_reward:
                    best_expected_reward = expected_reward
                    best_prediction = predicted_result

        return best_prediction
