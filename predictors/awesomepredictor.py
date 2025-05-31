from .base import PredictorBase
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical # type: ignore
import glob
import penaltyblog as pb
from sklearn.model_selection import train_test_split
import random
import os

class AwesomePredictor(PredictorBase):
    MAX_GOALS = 5  # This will be set dynamically

    def __init__(self):
        # Set seeds for determinism
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['PYTHONHASHSEED'] = '42'
        tf.config.experimental.enable_op_determinism()

        # Load the data
        # Load and concatenate all CSV files from the "data" folder
        csv_files = glob.glob(os.path.join('data', '*.csv'))

        dataframes = []
        for file in csv_files:
            df = pd.read_csv(file)
            df = df[['PSCH', 'PSCD', 'PSCA', 'FTHG', 'FTAG']]

            dataframes.append(df)

        # Combine all the cleaned DataFrames
        data = pd.concat(dataframes, ignore_index=True)

        # --- Dynamically set MAX_GOALS based on data ---
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
        data['scoreline'] = data.apply(
            lambda row: row['FTHG'] * (self.MAX_GOALS + 1) + row['FTAG'], axis=1
        )
        # Filter out scorelines beyond MAX_GOALS for training (shouldn't be necessary now, but good practice)
        y = to_categorical(data['scoreline'], num_classes=(self.MAX_GOALS + 1)**2)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Build and train TensorFlow model
        self.model = self.build_model()
        self.model.fit(X_train, y_train, epochs=200, verbose=0, shuffle=False)

    def calculate_probabilities(self, rate_home, rate_deuce, rate_road):
        return pb.implied.power([rate_home, rate_deuce, rate_road])["implied_probabilities"]
    
    def calculate_reward(self, predicted, actual, win_prob, draw_prob, lose_prob, win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points, min_points, max_points):
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

        # Calculate bonus points if MIN and MAX are greater than 0
        bonus_points = 0
        if min_points > 0 or max_points > 0:
            r = 0
            if act_home > act_away: # Actual result is a home win
                r = win_prob
            elif act_home < act_away: # Actual result is an away win
                r = lose_prob
            else: # Actual result is a draw
                r = draw_prob

            # Ensure r is not zero to avoid division by zero
            if r != 0:
                points = max_points / (10 * r) - max_points / 10 + min_points
                bonus_points = round(points)

                # Apply MIN and MAX constraints to bonus points
                if bonus_points < min_points:
                    bonus_points = min_points
                if bonus_points > max_points:
                    bonus_points = max_points

        return base_reward + bonus_points
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense((self.MAX_GOALS + 1)**2, activation='softmax')  # Output: probabilities for all scorelines
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict(self, match, win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points, min_points, max_points):
        match_probabilities = self.calculate_probabilities(match.rate_home, match.rate_deuce, match.rate_road)
        match_data = pd.DataFrame([match_probabilities], columns=['home_prob', 'draw_prob', 'away_prob'])
        win_prob, draw_prob, lose_prob = match_probabilities
        # Predict scoreline probabilities with TensorFlow model
        scoreline_probabilities_full = self.model.predict(match_data, verbose=0)[0]

        best_prediction = (0, 0)
        best_expected_reward = 0 # Initialize with a low value

        # Iterate through all possible scorelines and calculate expected reward
        for home_goals in range(self.MAX_GOALS + 1):
            for away_goals in range(self.MAX_GOALS + 1):
                predicted_scoreline = (home_goals, away_goals)
                expected_reward = 0

                # Iterate through all possible actual scorelines (up to MAX_GOALS)
                for actual_home_goals in range(self.MAX_GOALS + 1):
                    for actual_away_goals in range(self.MAX_GOALS + 1):
                        actual_scoreline = (actual_home_goals, actual_away_goals)

                        # Ensure the actual scoreline is within the trained range
                        if actual_home_goals <= self.MAX_GOALS and actual_away_goals <= self.MAX_GOALS:
                            # Get the probability of this actual scoreline
                            scoreline_index = actual_home_goals * (self.MAX_GOALS + 1) + actual_away_goals

                            # Probability is directly available from the TensorFlow model output
                            probability = scoreline_probabilities_full[scoreline_index]

                            reward = self.calculate_reward(predicted_scoreline, actual_scoreline, win_prob, draw_prob, lose_prob, win_exact_score_points, win_goal_difference_points, win_tendency_points, draw_exact_score_points, draw_tendency_points, min_points, max_points)
                            expected_reward += reward * probability

                # Update best prediction if current one has higher expected reward
                if expected_reward >= best_expected_reward:
                    best_expected_reward = expected_reward
                    best_prediction = predicted_scoreline

        return best_prediction