#!/usr/bin/env python3
"""
Beer Recommendation System - Model Training Script

This script trains a TensorFlow Recommenders model for beer recommendations
using the prepared BeerAdvocate dataset.

Usage:
    python train_model.py --epochs 10 --batch_size 1024 --learning_rate 0.001
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import joblib
from datetime import datetime
from typing import Dict, Text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BeerRecommenderModel(tfrs.Model):
    """Beer recommendation model using TensorFlow Recommenders."""
    
    def __init__(
        self, 
        user_vocab: np.ndarray, 
        beer_vocab: np.ndarray,
        embedding_dim: int = 64,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # User and beer vocabularies
        self.user_vocab = user_vocab
        self.beer_vocab = beer_vocab
        
        # User embedding
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=user_vocab, mask_token=None
            ),
            tf.keras.layers.Embedding(len(user_vocab) + 1, embedding_dim)
        ])
        
        # Beer embedding
        self.beer_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=beer_vocab, mask_token=None
            ),
            tf.keras.layers.Embedding(len(beer_vocab) + 1, embedding_dim)
        ])
        
        # Rating prediction network
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(hidden_dim // 2, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        
        # Task for rating prediction
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError()
            ]
        )
    
    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        user_embeddings = self.user_embedding(features["user_id"])
        beer_embeddings = self.beer_embedding(features["beer_id"])
        
        # Concatenate embeddings
        concatenated_embeddings = tf.concat([user_embeddings, beer_embeddings], axis=1)
        
        return self.rating_model(concatenated_embeddings)
    
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_embedding(features["user_id"])
        beer_embeddings = self.beer_embedding(features["beer_id"])
        
        concatenated_embeddings = tf.concat([user_embeddings, beer_embeddings], axis=1)
        rating_predictions = self.rating_model(concatenated_embeddings)
        
        return self.rating_task(
            labels=features["rating"],
            predictions=rating_predictions
        )

def load_prepared_data(data_dir: str = "data/prepared") -> tuple:
    """Load prepared training and test data."""
    logger.info("Loading prepared datasets...")
    
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
    
    logger.info(f"Loaded train data: {train_df.shape[0]} rows")
    logger.info(f"Loaded test data: {test_df.shape[0]} rows")
    
    return train_df, test_df

def load_encoders(models_dir: str = "models") -> tuple:
    """Load user and beer encoders."""
    logger.info("Loading encoders...")
    
    user_encoder = joblib.load(os.path.join(models_dir, "user_encoder.pkl"))
    beer_encoder = joblib.load(os.path.join(models_dir, "beer_encoder.pkl"))
    
    logger.info(f"User encoder classes: {len(user_encoder.classes_)}")
    logger.info(f"Beer encoder classes: {len(beer_encoder.classes_)}")
    
    return user_encoder, beer_encoder

def create_tf_dataset(df: pd.DataFrame, batch_size: int = 1024, shuffle: bool = True) -> tf.data.Dataset:
    """Create TensorFlow dataset from pandas DataFrame."""
    
    # Convert to string for TFR compatibility
    dataset = tf.data.Dataset.from_tensor_slices({
        "user_id": df["review_profilename"].astype(str).values,
        "beer_id": df["beer_beerid"].astype(str).values,
        "rating": df["review_overall"].astype(np.float32).values
    })
    
    if shuffle:
        dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=False)
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_model(
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    user_vocab: np.ndarray,
    beer_vocab: np.ndarray,
    epochs: int = 10,
    learning_rate: float = 0.001,
    embedding_dim: int = 64,
    hidden_dim: int = 128
) -> BeerRecommenderModel:
    """Train the beer recommendation model."""
    
    logger.info("Initializing model...")
    model = BeerRecommenderModel(
        user_vocab=user_vocab,
        beer_vocab=beer_vocab,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate))
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_root_mean_squared_error',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_root_mean_squared_error',
            factor=0.5,
            patience=2,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/beer_recommender_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    # Train the model
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model: BeerRecommenderModel, test_ds: tf.data.Dataset) -> Dict[str, float]:
    """Evaluate the trained model."""
    logger.info("Evaluating model...")
    
    metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return metrics

def save_model(model: BeerRecommenderModel, save_path: str = "models/beer_recommender"):
    """Save the trained model."""
    logger.info(f"Saving model to {save_path}...")
    
    # Save as SavedModel format
    tf.saved_model.save(model, save_path)
    
    # Also save model weights
    model.save_weights(f"{save_path}_weights")
    
    logger.info("Model saved successfully!")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Beer Recommendation Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--data_dir", type=str, default="data/prepared", help="Data directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Models directory")
    
    args = parser.parse_args()
    
    logger.info("="*50)
    logger.info("BEER RECOMMENDATION MODEL TRAINING")
    logger.info("="*50)
    logger.info(f"Configuration:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  Embedding Dim: {args.embedding_dim}")
    logger.info(f"  Hidden Dim: {args.hidden_dim}")
    
    try:
        # Load data
        train_df, test_df = load_prepared_data(args.data_dir)
        user_encoder, beer_encoder = load_encoders(args.models_dir)
        
        # Create vocabularies
        user_vocab = user_encoder.classes_.astype(str)
        beer_vocab = beer_encoder.classes_.astype(str)
        
        # Create TensorFlow datasets
        logger.info("Creating TensorFlow datasets...")
        train_ds = create_tf_dataset(train_df, args.batch_size, shuffle=True)
        test_ds = create_tf_dataset(test_df, args.batch_size, shuffle=False)
        
        # Train model
        model, history = train_model(
            train_ds=train_ds,
            test_ds=test_ds,
            user_vocab=user_vocab,
            beer_vocab=beer_vocab,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim
        )
        
        # Evaluate model
        metrics = evaluate_model(model, test_ds)
        
        # Save model
        save_model(model, os.path.join(args.models_dir, "beer_recommender"))
        
        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(args.models_dir, "training_history.csv"), index=False)
        
        # Save final metrics
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['metric', 'value'])
        metrics_df.to_csv(os.path.join(args.models_dir, "final_metrics.csv"), index=False)
        
        logger.info("="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info(f"Final RMSE: {metrics.get('root_mean_squared_error', 'N/A'):.4f}")
        logger.info(f"Final MAE: {metrics.get('mean_absolute_error', 'N/A'):.4f}")
        logger.info("Model saved to models/beer_recommender")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
