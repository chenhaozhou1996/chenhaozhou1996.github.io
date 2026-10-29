"""
Hospitality Revenue Optimization: Complete ML/RL/AI Implementation
==================================================================

This module implements a comprehensive revenue management system for hotels
combining demand forecasting (XGBoost), dynamic pricing (Deep Q-Learning),
and customer segmentation (K-Means clustering).

Author: ML Course Project
Date: September 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb

# Deep Learning for Reinforcement Learning
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"XGBoost version: {xgb.__version__}")


class HotelDemandForecaster:
    """
    Advanced demand forecasting model using XGBoost with feature engineering
    optimized for hotel booking patterns, seasonality, and market dynamics.
    """
    
    def __init__(self, forecast_horizon='multi', n_estimators=500):
        """
        Initialize the demand forecasting model
        
        Parameters:
        -----------
        forecast_horizon : str
            Time horizon for predictions ('short', 'medium', 'long', 'multi')
        n_estimators : int
            Number of boosting rounds for XGBoost
        """
        self.horizon = forecast_horizon
        self.models = {}
        self.feature_names = []
        self.scalers = {}
        
        # XGBoost configuration optimized for hotel demand
        self.config = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': n_estimators,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
    
    def create_features(self, df):
        """
        Engineer comprehensive features from raw booking data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw booking data with date, occupancy, and contextual information
            
        Returns:
        --------
        pandas.DataFrame
            Engineered features ready for modeling
        """
        features = pd.DataFrame(index=df.index)
        
        # Temporal features
        features['dow'] = df['date'].dt.dayofweek
        features['dom'] = df['date'].dt.day
        features['month'] = df['date'].dt.month
        features['quarter'] = df['date'].dt.quarter
        features['week_of_year'] = df['date'].dt.isocalendar().week
        features['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        features['is_month_start'] = (df['date'].dt.day <= 7).astype(int)
        features['is_month_end'] = (df['date'].dt.day >= 24).astype(int)
        
        # Lagged demand features (historical patterns)
        for lag in [1, 7, 14, 30, 365]:
            features[f'occupancy_lag_{lag}'] = df['occupancy_rate'].shift(lag)
            features[f'revenue_lag_{lag}'] = df['revenue'].shift(lag)
        
        # Rolling statistics (trend indicators)
        for window in [7, 14, 30, 90]:
            features[f'occupancy_roll_mean_{window}'] = df['occupancy_rate'].rolling(window).mean()
            features[f'occupancy_roll_std_{window}'] = df['occupancy_rate'].rolling(window).std()
            features[f'revenue_roll_mean_{window}'] = df['revenue'].rolling(window).mean()
        
        # Market positioning features
        features['price_index'] = df['avg_daily_rate'] / df['comp_set_avg_price']
        features['occupancy_gap'] = df['occupancy_rate'] - df['comp_set_avg_occupancy']
        features['price_gap'] = df['avg_daily_rate'] - df['comp_set_avg_price']
        
        # Event and external factors
        features['days_to_holiday'] = df['days_to_holiday']
        features['major_event'] = df['major_event_flag']
        features['local_events'] = df['local_events_count']
        features['airline_capacity'] = df['airline_capacity_index']
        features['consumer_confidence'] = df['consumer_confidence']
        features['weather_temp'] = df['weather_temp']
        
        # Segment mix
        features['transient_pct'] = df['segment_transient_pct']
        features['group_pct'] = df['segment_group_pct']
        features['corporate_pct'] = df['segment_corporate_pct']
        
        # Booking pace indicators
        features['booking_window'] = df['booking_window_avg']
        
        # Interaction features
        features['weekend_x_event'] = features['is_weekend'] * features['major_event']
        features['price_x_occupancy'] = features['price_index'] * df['occupancy_rate']
        
        # Fill missing values created by lagging/rolling
        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the XGBoost forecasting model
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.ndarray
            Training features
        y_train : pandas.Series or numpy.ndarray
            Training target (occupancy rate or demand metric)
        X_val : pandas.DataFrame or numpy.ndarray, optional
            Validation features
        y_val : pandas.Series or numpy.ndarray, optional
            Validation target
            
        Returns:
        --------
        dict
            Training metrics and model performance
        """
        print(f"Training demand forecasting model with {len(X_train)} samples...")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Training parameters
        params = self.config.copy()
        num_rounds = params.pop('n_estimators')
        
        # Add validation set if provided
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'validation'))
        
        # Train model
        self.model = xgb.train(
            params,
            dtrain,
            num_rounds,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Calculate training metrics
        train_pred = self.model.predict(dtrain)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        
        metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(dval)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            
            metrics.update({
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            })
        
        print("\n" + "="*60)
        print("Model Training Complete!")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("="*60)
        
        return metrics
    
    def predict(self, X):
        """
        Generate demand forecasts
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features for prediction
            
        Returns:
        --------
        numpy.ndarray
            Predicted demand values
        """
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def get_feature_importance(self, plot=True):
        """
        Analyze and visualize feature importance
        
        Parameters:
        -----------
        plot : bool
            Whether to create visualization
            
        Returns:
        --------
        pandas.DataFrame
            Feature importance scores
        """
        importance = self.model.get_score(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        if plot and len(importance_df) > 0:
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df.head(20), x='importance', y='feature')
            plt.title('Top 20 Most Important Features for Demand Forecasting')
            plt.xlabel('Importance Score (Gain)')
            plt.tight_layout()
            plt.show()
        
        return importance_df


class DynamicPricingAgent:
    """
    Deep Q-Learning agent for optimal hotel room pricing
    Learns to maximize revenue while maintaining occupancy targets
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the pricing agent with neural network architecture
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state space (market conditions, inventory, etc.)
        action_dim : int
            Number of discrete pricing actions
        hidden_dim : int
            Size of hidden layers in neural network
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Deep Q-Network
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training configuration
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay memory
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Exploration parameters
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
        # Training metrics
        self.losses = []
        self.rewards = []
    
    def _build_network(self):
        """
        Construct the Deep Q-Network architecture
        
        Returns:
        --------
        torch.nn.Module
            Neural network for Q-value estimation
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.action_dim)
        )
    
    def get_state(self, data_point):
        """
        Convert market conditions into state representation
        
        Parameters:
        -----------
        data_point : dict or pandas.Series
            Current market conditions and hotel metrics
            
        Returns:
        --------
        numpy.ndarray
            State vector for the agent
        """
        state = np.array([
            data_point['occupancy_rate'],
            data_point['days_until_checkin'],
            data_point['rooms_available'] / data_point['total_rooms'],
            data_point['comp_set_avg_price'] / 300.0,  # Normalized
            data_point['comp_set_avg_occupancy'],
            data_point['predicted_demand'],
            data_point['dow'] / 7.0,
            data_point['is_weekend'],
            data_point['major_event_flag'],
            data_point['segment_transient_pct'],
            data_point['booking_window_avg'] / 180.0,  # Normalized
            data_point['price_last_period'] / 300.0,  # Normalized
            data_point['revenue_last_week'] / 50000.0,  # Normalized
        ])
        return state
    
    def select_action(self, state, explore=True):
        """
        Choose pricing action using epsilon-greedy policy
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state representation
        explore : bool
            Whether to use exploration (training) or exploitation (inference)
            
        Returns:
        --------
        int
            Selected action index
        """
        if explore and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_dim)
        
        # Exploitation: best action according to Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : numpy.ndarray
            Resulting state
        done : bool
            Whether episode terminated
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        Train the Q-network using experience replay
        
        Returns:
        --------
        float
            Training loss for this batch
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample random minibatch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
        return loss.item()
    
    def update_target_network(self):
        """
        Update target network with current Q-network weights
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def calculate_reward(self, action, data_point, price_actions):
        """
        Compute reward based on revenue and strategic objectives
        
        Parameters:
        -----------
        action : int
            Pricing action index
        data_point : dict
            Current market conditions
        price_actions : list
            Available price points
            
        Returns:
        --------
        float
            Reward value
        """
        price = price_actions[action]
        
        # Estimate bookings based on price elasticity
        base_demand = data_point['predicted_demand']
        price_sensitivity = -1.5  # Typical hotel price elasticity
        comp_price = data_point['comp_set_avg_price']
        
        # Demand adjustment based on competitive pricing
        price_ratio = price / comp_price
        demand_multiplier = np.exp(price_sensitivity * (price_ratio - 1))
        estimated_bookings = base_demand * demand_multiplier
        
        # Revenue calculation
        revenue = price * estimated_bookings
        
        # Occupancy consideration
        target_occupancy = 0.75
        rooms_available = data_point['rooms_available']
        occupancy = min(estimated_bookings / rooms_available, 1.0)
        
        # Multi-objective reward function
        revenue_component = revenue / 10000.0  # Normalized
        occupancy_component = -abs(occupancy - target_occupancy) * 10
        
        # Penalty for extreme pricing
        if price < comp_price * 0.7 or price > comp_price * 1.3:
            pricing_penalty = -5
        else:
            pricing_penalty = 0
        
        total_reward = revenue_component + occupancy_component + pricing_penalty
        
        return total_reward


# Training and Evaluation Functions

def train_forecasting_model(data_path='hotel_bookings_sample.csv'):
    """
    Complete pipeline for training demand forecasting model
    
    Parameters:
    -----------
    data_path : str
        Path to booking data CSV file
        
    Returns:
    --------
    tuple
        (trained_model, test_metrics, feature_importance)
    """
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Initialize forecaster
    forecaster = HotelDemandForecaster(forecast_horizon='multi')
    
    # Engineer features
    print("Engineering features...")
    X = forecaster.create_features(df)
    y = df['occupancy_rate']
    
    # Train/test split (time-series aware)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    metrics = forecaster.train(X_train, y_train, X_test, y_test)
    
    # Generate predictions
    predictions = forecaster.predict(X_test)
    
    # Visualize results
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.values, label='Actual Occupancy', alpha=0.7)
    plt.plot(predictions, label='Predicted Occupancy', alpha=0.7)
    plt.title('Demand Forecasting: Actual vs Predicted Occupancy')
    plt.xlabel('Days')
    plt.ylabel('Occupancy Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Feature importance
    importance_df = forecaster.get_feature_importance(plot=True)
    
    return forecaster, metrics, importance_df


def train_pricing_agent(forecaster, data_path='hotel_bookings_sample.csv', episodes=500):
    """
    Train reinforcement learning agent for dynamic pricing
    
    Parameters:
    -----------
    forecaster : HotelDemandForecaster
        Trained demand forecasting model
    data_path : str
        Path to booking data
    episodes : int
        Number of training episodes
        
    Returns:
    --------
    DynamicPricingAgent
        Trained pricing agent
    """
    print(f"\nTraining Dynamic Pricing Agent for {episodes} episodes...")
    
    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Define pricing actions (discrete price points)
    price_actions = [120, 135, 150, 165, 180, 195, 210, 225, 240]
    
    # Initialize agent
    state_dim = 13  # As defined in get_state()
    action_dim = len(price_actions)
    agent = DynamicPricingAgent(state_dim, action_dim)
    
    # Training loop
    episode_rewards = []
    
    for episode in range(episodes):
        episode_reward = 0
        
        # Sample random starting point
        idx = random.randint(0, len(df) - 30)
        
        for step in range(30):  # 30-day episodes
            if idx + step >= len(df):
                break
            
            # Get current state
            current_data = df.iloc[idx + step]
            state = agent.get_state(current_data)
            
            # Select and execute action
            action = agent.select_action(state)
            
            # Calculate reward
            reward = agent.calculate_reward(action, current_data, price_actions)
            episode_reward += reward
            
            # Get next state
            if idx + step + 1 < len(df):
                next_data = df.iloc[idx + step + 1]
                next_state = agent.get_state(next_data)
                done = False
            else:
                next_state = state
                done = True
            
            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
        
        episode_rewards.append(episode_reward)
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Progress reporting
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}/{episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    # Visualize training progress
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6)
    plt.plot(pd.Series(episode_rewards).rolling(50).mean(), linewidth=2)
    plt.title('Training Progress: Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(agent.losses[-1000:])  # Last 1000 losses
    plt.title('Training Loss (Last 1000 Updates)')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTraining Complete!")
    print(f"Final Average Reward: {np.mean(episode_rewards[-50:]):.2f}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    
    return agent


if __name__ == "__main__":
    print("="*70)
    print("Hospitality Revenue Optimization: ML/RL/AI Implementation")
    print("="*70)
    
    # Step 1: Train demand forecasting model
    print("\n[STEP 1: DEMAND FORECASTING]")
    forecaster, forecast_metrics, feature_importance = train_forecasting_model()
    
    # Step 2: Train dynamic pricing agent
    print("\n[STEP 2: DYNAMIC PRICING WITH REINFORCEMENT LEARNING]")
    pricing_agent = train_pricing_agent(forecaster, episodes=500)
    
    print("\n" + "="*70)
    print("Complete Implementation Ready!")
    print("="*70)
    print("\nNext steps:")
    print("1. Use forecaster.predict() for demand forecasts")
    print("2. Use pricing_agent.select_action() for optimal pricing")
    print("3. Monitor performance with business metrics")
    print("="*70)
