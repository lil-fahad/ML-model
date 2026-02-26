#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run ML models on gym-anytrading simulator.

This script integrates the trained ML models with the gym-anytrading
trading environment to evaluate their performance in a simulated trading scenario.
"""
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import gymnasium as gym
from gym_anytrading.envs import StocksEnv

# Setup paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from enhanced_features import build_enhanced_features, FEATURES_10

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_name: str) -> Tuple[Any, Dict[str, Any]]:
    """Load a model and its metadata."""
    model_path = ROOT / "models" / f"{model_name}.pkl"
    meta_path = ROOT / "models" / f"{model_name}_meta.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    
    return model, meta


def get_sample_data() -> pd.DataFrame:
    """Get sample stock data for testing."""
    # Check if we have data in the data directory
    data_dir = ROOT / "data"
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            # Use the first CSV file
            df = pd.read_csv(csv_files[0])
            df.columns = [c.lower() for c in df.columns]
            logger.info(f"Loaded data from {csv_files[0].name}: {len(df)} rows")
            return df
    
    # Generate synthetic data if no real data available
    logger.info("Generating synthetic stock data for testing...")
    np.random.seed(42)
    num_days = 500
    
    dates = pd.date_range(start="2023-01-01", periods=num_days, freq="D")
    close = 100 + np.cumsum(np.random.randn(num_days) * 2)
    high = close + np.abs(np.random.randn(num_days)) * 2
    low = close - np.abs(np.random.randn(num_days)) * 2
    open_price = close + np.random.randn(num_days) * 0.5
    volume = np.abs(np.random.randn(num_days) * 1000000) + 500000
    
    df = pd.DataFrame({
        "date": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })
    
    return df


class MLTradingEnv(StocksEnv):
    """
    Custom trading environment that uses ML model predictions.
    
    This extends StocksEnv to integrate our trained ML models
    for making trading decisions.
    """
    
    def __init__(self, df: pd.DataFrame, window_size: int = 30, frame_bound: Tuple[int, int] = None):
        """
        Initialize the ML Trading Environment.
        
        Args:
            df: DataFrame with OHLCV data
            window_size: Lookback window for features
            frame_bound: (start, end) bounds for the trading frame
        """
        # Ensure required columns exist and standardize names
        df = df.copy()
        
        # Standardize column names (gym-anytrading expects 'Close')
        col_map = {c.lower(): c for c in df.columns}
        rename_map = {}
        for std_name in ['close', 'open', 'high', 'low', 'volume']:
            if std_name in col_map:
                rename_map[col_map[std_name]] = std_name.capitalize()
        df = df.rename(columns=rename_map)
        
        if frame_bound is None:
            frame_bound = (window_size, len(df))
        
        super().__init__(df=df, window_size=window_size, frame_bound=frame_bound)
    
    def _process_data(self):
        """Process the dataframe to create observation signals."""
        prices = self.df.loc[:, 'Close'].to_numpy()
        
        # Validate index
        prices[self.frame_bound[0] - self.window_size]
        prices = prices[self.frame_bound[0] - self.window_size:self.frame_bound[1]]
        
        # Create signal features (prices and their differences)
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))
        
        return prices.astype(np.float32), signal_features.astype(np.float32)


def run_ml_trading_simulation(
    model_name: str = "enhanced_model",
    num_episodes: int = 5,
    window_size: int = 30,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run ML model on gym-anytrading simulator.
    
    Args:
        model_name: Name of the model to use ('hybrid_model' or 'enhanced_model')
        num_episodes: Number of trading episodes to run
        window_size: Lookback window for the environment
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with simulation results
    """
    # Load model
    model, meta = load_model(model_name)
    features = meta.get("features", FEATURES_10)
    logger.info(f"Loaded model: {model_name}")
    logger.info(f"Model features: {len(features)}")
    
    # Get data
    df = get_sample_data()
    
    # Build features for the entire dataset
    features_df = build_enhanced_features(df)
    
    # Merge back with original data for the environment
    df_with_features = df.copy()
    for col in features:
        if col in features_df.columns:
            df_with_features[col] = features_df[col]
    
    # Drop rows with NaN features
    start_idx = df_with_features[features].first_valid_index() or 0
    valid_df = df_with_features.iloc[start_idx:].dropna(subset=features).reset_index(drop=True)
    
    if len(valid_df) < window_size + 50:
        raise ValueError(f"Not enough valid data points. Need at least {window_size + 50}, got {len(valid_df)}")
    
    logger.info(f"Valid data points: {len(valid_df)}")
    
    # Create environment
    frame_bound = (window_size, len(valid_df))
    env = MLTradingEnv(valid_df, window_size=window_size, frame_bound=frame_bound)
    
    results = {
        "model": model_name,
        "episodes": [],
        "total_profit": 0,
        "avg_profit": 0,
        "win_rate": 0
    }
    
    wins = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_profit = 0
        steps = 0
        actions_taken = {"buy": 0, "sell": 0}
        
        while not done and not truncated:
            # Get current position in the original dataframe
            current_tick = env._current_tick
            
            # Get features for prediction
            try:
                feature_values = valid_df.iloc[current_tick][features].values.reshape(1, -1)
                
                # Replace any NaN with 0
                feature_values = np.nan_to_num(feature_values, nan=0.0)
                
                # Get ML model prediction
                prediction = model.predict(feature_values)[0]
                
                # Map prediction to action (0=hold/sell, 1=buy)
                action = int(prediction)
            except Exception as e:
                # Default to random action on error
                action = env.action_space.sample()
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            episode_profit += reward
            steps += 1
            
            if action == 1:
                actions_taken["buy"] += 1
            else:
                actions_taken["sell"] += 1
        
        final_profit = info.get("total_profit", episode_profit)
        
        episode_result = {
            "episode": episode + 1,
            "steps": steps,
            "profit": final_profit,
            "actions": actions_taken
        }
        results["episodes"].append(episode_result)
        results["total_profit"] += final_profit
        
        if final_profit > 1.0:  # Profitable episode
            wins += 1
        
        if verbose:
            logger.info(f"Episode {episode + 1}: Steps={steps}, Profit={final_profit:.4f}, Actions={actions_taken}")
    
    results["avg_profit"] = results["total_profit"] / num_episodes
    results["win_rate"] = wins / num_episodes
    
    env.close()
    
    return results


def run_random_baseline(
    num_episodes: int = 5,
    window_size: int = 30,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run random trading baseline for comparison.
    
    Args:
        num_episodes: Number of episodes
        window_size: Lookback window
        verbose: Print details
    
    Returns:
        Baseline results
    """
    df = get_sample_data()
    
    # Need at least window_size + some buffer
    if len(df) < window_size + 50:
        raise ValueError(f"Not enough data points")
    
    frame_bound = (window_size, len(df))
    env = MLTradingEnv(df, window_size=window_size, frame_bound=frame_bound)
    
    results = {
        "model": "random_baseline",
        "episodes": [],
        "total_profit": 0,
        "avg_profit": 0,
        "win_rate": 0
    }
    
    wins = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_profit = 0
        steps = 0
        
        while not done and not truncated:
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            episode_profit += reward
            steps += 1
        
        final_profit = info.get("total_profit", episode_profit)
        
        results["episodes"].append({
            "episode": episode + 1,
            "steps": steps,
            "profit": final_profit
        })
        results["total_profit"] += final_profit
        
        if final_profit > 1.0:
            wins += 1
        
        if verbose:
            logger.info(f"Random Episode {episode + 1}: Steps={steps}, Profit={final_profit:.4f}")
    
    results["avg_profit"] = results["total_profit"] / num_episodes
    results["win_rate"] = wins / num_episodes
    
    env.close()
    
    return results


def main():
    """Run simulator comparison between ML models and random baseline."""
    print("=" * 60)
    print("Stock Trading Simulator - ML Model Evaluation")
    print("=" * 60)
    print()
    
    num_episodes = 5
    
    # Run random baseline
    print("Running Random Baseline...")
    print("-" * 40)
    baseline_results = run_random_baseline(num_episodes=num_episodes)
    print(f"\nBaseline Average Profit: {baseline_results['avg_profit']:.4f}")
    print(f"Baseline Win Rate: {baseline_results['win_rate']:.2%}")
    print()
    
    # Run with hybrid model
    print("Running Hybrid Model...")
    print("-" * 40)
    try:
        hybrid_results = run_ml_trading_simulation(
            model_name="hybrid_model",
            num_episodes=num_episodes
        )
        print(f"\nHybrid Model Average Profit: {hybrid_results['avg_profit']:.4f}")
        print(f"Hybrid Model Win Rate: {hybrid_results['win_rate']:.2%}")
    except Exception as e:
        logger.error(f"Hybrid model error: {e}")
        hybrid_results = None
    print()
    
    # Run with enhanced model
    print("Running Enhanced Model...")
    print("-" * 40)
    try:
        enhanced_results = run_ml_trading_simulation(
            model_name="enhanced_model",
            num_episodes=num_episodes
        )
        print(f"\nEnhanced Model Average Profit: {enhanced_results['avg_profit']:.4f}")
        print(f"Enhanced Model Win Rate: {enhanced_results['win_rate']:.2%}")
    except Exception as e:
        logger.error(f"Enhanced model error: {e}")
        enhanced_results = None
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Avg Profit':<15} {'Win Rate':<15}")
    print("-" * 50)
    print(f"{'Random Baseline':<20} {baseline_results['avg_profit']:<15.4f} {baseline_results['win_rate']:<15.2%}")
    if hybrid_results:
        print(f"{'Hybrid Model':<20} {hybrid_results['avg_profit']:<15.4f} {hybrid_results['win_rate']:<15.2%}")
    if enhanced_results:
        print(f"{'Enhanced Model':<20} {enhanced_results['avg_profit']:<15.4f} {enhanced_results['win_rate']:<15.2%}")
    
    return {
        "baseline": baseline_results,
        "hybrid": hybrid_results,
        "enhanced": enhanced_results
    }


if __name__ == "__main__":
    main()
