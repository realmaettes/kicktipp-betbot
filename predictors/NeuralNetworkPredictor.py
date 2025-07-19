from .base import PredictorBase
import numpy as np
import pandas as pd
import glob
import penaltyblog as pb
import os
from sklearn.neural_network import MLPClassifier
import joblib

class NeuralNetworkPredictor(PredictorBase):
    MAX_GOALS = 10  # This will be set dynamically

    def __init__(self):
        # Load the data
        # Load and concatenate all CSV files from the "data" folder
        csv_files = glob.glob(os.path.join('data', '*.csv'))

        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file, usecols=['PSCH', 'PSCD', 'PSCA', 'FTHG', 'FTAG'])
        
                # Drop rows with missing or non-integer or invalid scores
                df = df.dropna(subset=['PSCH', 'PSCD', 'PSCA', 'FTHG', 'FTAG'])

                df = df[
                    df['FTHG'].apply(lambda x: isinstance(x, (int, float)) and x >= 0 and x == int(x)) &
                    df['FTAG'].apply(lambda x: isinstance(x, (int, float)) and x >= 0 and x == int(x)) &
                    df['PSCH'].apply(lambda x: isinstance(x, (int, float)) and x >= 1) &
                    df['PSCD'].apply(lambda x: isinstance(x, (int, float)) and x >= 1) &
                    df['PSCA'].apply(lambda x: isinstance(x, (int, float)) and x >= 1)
                ]

                df['FTHG'] = df['FTHG'].astype(int)
                df['FTAG'] = df['FTAG'].astype(int)

                if not df.empty:
                    dataframes.append(df)

            except Exception as e:
                continue

        # Combine all the cleaned DataFrames
        data = pd.concat(dataframes, ignore_index=True)

        # Dynamically set MAX_GOALS based on data
        max_home_goals = data['FTHG'].max()
        max_away_goals = data['FTAG'].max()
        self.MAX_GOALS = max(max_home_goals, max_away_goals)
        # ------------------------------------------------

        # Compute implied probabilities
        probabilities = data.apply(
            lambda row: self.calculate_probabilities(row['PSCH'], row['PSCD'], row['PSCA']),
            axis=1
        )
        probabilities_df = pd.DataFrame(probabilities.tolist(), columns=['home_prob', 'draw_prob', 'away_prob'])

        # Prepare features and targets
        X = probabilities_df

        # Create a categorical target variable for scoreline
        # Map each scoreline (home_goals, away_goals) to a unique integer
        # Encode the scoreline as integer labels directly
        data['scoreline'] = data.apply(
            lambda row: row['FTHG'] * (self.MAX_GOALS + 1) + row['FTAG'], axis=1
        )

        X = probabilities_df
        y = data['scoreline'].values

        # Save number of classes for use in prediction
        self.n_classes = (self.MAX_GOALS + 1) ** 2

        if os.path.exists('neural_predictor_model.pkl'):
            self.model = joblib.load('neural_predictor_model.pkl')
        else:
            self.model = MLPClassifier(random_state=42)
            self.model.fit(X, y)
            joblib.dump(self.model, 'neural_predictor_model.pkl')

    def calculate_probabilities(self, rate_home, rate_deuce, rate_road):
        return pb.implied.power([rate_home, rate_deuce, rate_road])["implied_probabilities"]
    
    def calculate_reward(self, predicted, actual, win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points, quote_points):
        pred_home, pred_away = predicted
        act_home, act_away = actual

        base_reward = 0

        if predicted == actual:
            if act_home == act_away:
                base_reward = draw_exact_score_points
            else:
                base_reward = win_exact_score_points
        elif (pred_home - pred_away) == (act_home - act_away):
            if act_home == act_away:
                base_reward = draw_tendency_points
            else:
                base_reward = win_goal_difference_points
        elif (pred_home > pred_away and act_home > act_away) or (pred_home < pred_away and act_home < act_away):
            base_reward = win_tendency_points

        community_odds_points = 0
        if act_home > act_away:
            community_odds_points = quote_points[0]
        elif act_home < act_away:
            community_odds_points = quote_points[2]
        else:
            community_odds_points = quote_points[1]

        return base_reward + community_odds_points

    def predict(self, match, win_exact_score_points, win_goal_difference_points, win_tendency_points,
                draw_exact_score_points, draw_tendency_points, quote_points):
        match_probs = self.calculate_probabilities(match.rate_home, match.rate_deuce, match.rate_road)
        X_match = pd.DataFrame([match_probs], columns=['home_prob', 'draw_prob', 'away_prob'])

        # Get predicted probabilities for *only known classes*
        known_class_probs = self.model.predict_proba(X_match)[0]
        known_classes = self.model.classes_

        # Reconstruct full vector (fill missing classes with 0 probability)
        full_probs = np.zeros(self.n_classes)
        for class_index, prob in zip(known_classes, known_class_probs):
            full_probs[int(class_index)] = prob

        best_prediction = (0, 0)
        best_expected_reward = 0

        for home_goals in range(self.MAX_GOALS + 1):
            for away_goals in range(self.MAX_GOALS + 1):
                predicted = (home_goals, away_goals)
                expected_reward = 0.0

                for actual_home in range(self.MAX_GOALS + 1):
                    for actual_away in range(self.MAX_GOALS + 1):
                        actual = (actual_home, actual_away)
                        idx = actual_home * (self.MAX_GOALS + 1) + actual_away
                        prob = full_probs[idx]
                        reward = self.calculate_reward(predicted, actual,
                                                       win_exact_score_points, win_goal_difference_points,
                                                       win_tendency_points, draw_exact_score_points,
                                                       draw_tendency_points, quote_points)
                        expected_reward += prob * reward

                if expected_reward >= best_expected_reward:
                    best_expected_reward = expected_reward
                    best_prediction = predicted

        return best_prediction
