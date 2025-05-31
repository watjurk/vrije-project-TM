from sklearn import naive_bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import numpy as np


class BayesModel:
    """Naive Bayes model for topic classification."""

    def __init__(self, max_features: int = 5000):
        """
        Initialize the Naive Bayes model.

        Args:
            max_features: Maximum number of features for TF-IDF vectorization
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = naive_bayes.MultinomialNB()
        self.classes_ = None  # Will be populated after fitting

    def fit(self, X_train: pd.Series, y_train: pd.Series) -> "BayesModel":
        """
        Train the model on the provided data.

        Args:
            X_train: Series containing text data
            y_train: Series containing topic labels

        Returns:
            Self for method chaining
        """
        # Transform text to TF-IDF features
        print("Transforming training text to TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Train the model
        print("Training Naive Bayes model...")
        self.model.fit(X_train_tfidf, y_train)

        # Store classes for later use
        self.classes_ = self.model.classes_

        print(f"Model trained successfully with {self.max_features} features.")
        return self

    def predict(self, X_test: pd.Series) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test: Series containing text data

        Returns:
            Array of predicted topics
        """
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained. Call fit() before predict().")

        # Transform test data using the same vectorizer
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Make predictions
        return self.model.predict(X_test_tfidf)

    def evaluate(
        self,
        X_test: pd.Series,
        y_test: pd.Series,
        display_matrix: bool = True,
        display_misclassified: bool = True,
    ) -> Dict:
        """
        Evaluate the model performance.

        Args:
            X_test: Series containing text data
            y_test: Series containing true topic labels
            display_matrix: Whether to display the confusion matrix
            display_misclassified: Whether to display misclassified examples

        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        predictions = self.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")

        # Generate classification report
        report = classification_report(y_test, predictions, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        # Create confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Plot confusion matrix if requested
        if display_matrix:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.classes_,
                yticklabels=self.classes_,
            )
            plt.xlabel("Predicted Topic")
            plt.ylabel("Actual Topic")
            plt.title("Confusion Matrix for Naive Bayes Topic Classification")
            plt.tight_layout()
            plt.show()

        # Create results DataFrame and show misclassified examples
        results_df = pd.DataFrame(
            {"text": X_test, "true_topic": y_test, "predicted_topic": predictions}
        )

        # Find misclassified examples
        misclassified = results_df[
            results_df["true_topic"] != results_df["predicted_topic"]
        ]
        misclassified_pct = len(misclassified) / len(results_df)
        print(
            f"\nMisclassified examples: {len(misclassified)}/{len(results_df)} ({misclassified_pct:.1%})"
        )

        # Display sample of misclassified examples
        if display_misclassified and len(misclassified) > 0:
            print("\nSample of misclassified examples:")
            sample_size = min(5, len(misclassified))
            print(
                misclassified[["text", "true_topic", "predicted_topic"]].sample(
                    sample_size
                )
            )

        # Return evaluation metrics
        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm,
            "misclassified_count": len(misclassified),
            "misclassified_percent": misclassified_pct,
        }


class XGBoostModel:
    """XGBoost model for topic classification."""

    def __init__(
        self,
        max_features: int = 5000,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
    ):
        """
        Initialize the XGBoost model.

        Args:
            max_features: Maximum number of features for TF-IDF vectorization
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate for XGBoost
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.le = LabelEncoder()
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        self.classes_ = None

    def fit(self, X_train: pd.Series, y_train: pd.Series) -> "XGBoostModel":
        """
        Train the model on the provided data.

        Args:
            X_train: Series containing text data
            y_train: Series containing topic labels

        Returns:
            Self for method chaining
        """
        # Transform text to TF-IDF features
        print("Transforming training text to TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Encode labels
        print("Encoding labels...")
        self.le.fit(y_train)
        y_train_encoded = self.le.transform(y_train)
        self.classes_ = self.le.classes_

        print(f"Classes found: {self.classes_}")

        # Convert to dense for XGBoost
        X_train_dense = X_train_tfidf.toarray()

        # Train the model
        print(f"Training XGBoost model with {self.max_features} features...")
        self.model.fit(X_train_dense, y_train_encoded)

        print("XGBoost model trained successfully.")
        return self

    def predict(self, X_test: pd.Series) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test: Series containing text data

        Returns:
            Array of predicted topics
        """
        if self.vectorizer is None or self.model is None or self.le is None:
            raise ValueError("Model not trained. Call fit() before predict().")

        # Transform test data using the same vectorizer
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Convert to dense for XGBoost
        X_test_dense = X_test_tfidf.toarray()

        # Make predictions (returns numeric predictions)
        numeric_preds = self.model.predict(X_test_dense)

        # Convert back to string labels
        return self.le.inverse_transform(numeric_preds)

    def evaluate(
        self,
        X_test: pd.Series,
        y_test: pd.Series,
        display_matrix: bool = True,
        display_misclassified: bool = True,
    ) -> Dict:
        """
        Evaluate the model performance.

        Args:
            X_test: Series containing text data
            y_test: Series containing true topic labels
            display_matrix: Whether to display the confusion matrix
            display_misclassified: Whether to display misclassified examples

        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        predictions = self.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")

        # Generate classification report
        report = classification_report(y_test, predictions, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        # Create confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Plot confusion matrix if requested
        if display_matrix:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.classes_,
                yticklabels=self.classes_,
            )
            plt.xlabel("Predicted Topic")
            plt.ylabel("Actual Topic")
            plt.title("Confusion Matrix for XGBoost Topic Classification")
            plt.tight_layout()
            plt.show()

        # Create results DataFrame and show misclassified examples
        results_df = pd.DataFrame(
            {"text": X_test, "true_topic": y_test, "predicted_topic": predictions}
        )

        # Find misclassified examples
        misclassified = results_df[
            results_df["true_topic"] != results_df["predicted_topic"]
        ]
        misclassified_pct = len(misclassified) / len(results_df)
        print(
            f"\nMisclassified examples: {len(misclassified)}/{len(results_df)} ({misclassified_pct:.1%})"
        )

        # Display sample of misclassified examples
        if display_misclassified and len(misclassified) > 0:
            print("\nSample of misclassified examples:")
            sample_size = min(5, len(misclassified))
            print(
                misclassified[["text", "true_topic", "predicted_topic"]].sample(
                    sample_size
                )
            )

        # Return evaluation metrics
        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm,
            "misclassified_count": len(misclassified),
            "misclassified_percent": misclassified_pct,
        }


class PyTorchNNModel:
    """PyTorch neural network model for topic classification."""

    def __init__(
        self,
        max_features: int = 5000,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        lr: float = 0.001,
        batch_size: int = 64,
        epochs: int = 20,
    ):
        """
        Initialize the PyTorch neural network model.

        Args:
            max_features: Maximum number of features for TF-IDF vectorization
            hidden_dim: Size of hidden layers
            dropout: Dropout rate for regularization
            lr: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        self.max_features = max_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Will be initialized during training
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.le = LabelEncoder()
        self.model = None
        self.classes_ = None
        self.criterion = nn.CrossEntropyLoss()

    class TextClassifier(nn.Module):
        """Neural network architecture for text classification."""

        def __init__(self, input_dim, hidden_dim=128, num_classes=3, dropout=0.3):
            """Initialize the neural network."""
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.66),  # Reduce dropout in deeper layers
            )
            self.output_layer = nn.Linear(hidden_dim // 2, num_classes)

        def forward(self, x):
            """Forward pass through the network."""
            x = self.layer1(x)
            x = self.layer2(x)
            return self.output_layer(x)

    def get_batch(self, X, y, idx):
        """Get a batch of data for training."""
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(X))
        return X[start_idx:end_idx], y[start_idx:end_idx]

    def fit(self, X_train: pd.Series, y_train: pd.Series) -> "PyTorchNNModel":
        """
        Train the model on the provided data.

        Args:
            X_train: Series containing text data
            y_train: Series containing topic labels

        Returns:
            Self for method chaining
        """
        # Transform text to TF-IDF features
        print("Transforming training text to TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Encode labels
        print("Encoding labels...")
        self.le.fit(y_train)
        y_train_encoded = self.le.transform(y_train)
        self.classes_ = self.le.classes_
        num_classes = len(self.classes_)
        print(f"Classes found: {self.classes_}")

        # Convert to dense numpy arrays
        X_train_dense = X_train_tfidf.toarray().astype(np.float32)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_dense).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)

        # Initialize model
        input_dim = X_train_dense.shape[1]
        self.model = self.TextClassifier(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=num_classes,
            dropout=self.dropout,
        ).to(self.device)

        print(
            f"Model created with input dimension: {input_dim}, output classes: {num_classes}"
        )

        # Initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        print(f"Beginning training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            num_batches = int(np.ceil(len(X_train_tensor) / self.batch_size))

            for i in range(num_batches):
                # Get batch
                X_batch, y_batch = self.get_batch(X_train_tensor, y_train_tensor, i)

                # Forward pass
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/num_batches:.4f}"
                )

        print("Training complete!")
        return self

    def predict(self, X_test: pd.Series) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test: Series containing text data

        Returns:
            Array of predicted topics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before predict().")

        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        X_test_dense = X_test_tfidf.toarray().astype(np.float32)
        X_test_tensor = torch.FloatTensor(X_test_dense).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, numeric_predictions = torch.max(outputs, 1)
            numeric_predictions = numeric_predictions.cpu().numpy()

        # Convert back to string labels
        return self.le.inverse_transform(numeric_predictions)

    def evaluate(
        self,
        X_test: pd.Series,
        y_test: pd.Series,
        display_matrix: bool = True,
        display_misclassified: bool = True,
    ) -> Dict:
        """
        Evaluate the model performance.

        Args:
            X_test: Series containing text data
            y_test: Series containing true topic labels
            display_matrix: Whether to display the confusion matrix
            display_misclassified: Whether to display misclassified examples

        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        predictions = self.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")

        # Generate classification report
        report = classification_report(y_test, predictions, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        # Create confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Plot confusion matrix if requested
        if display_matrix:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.classes_,
                yticklabels=self.classes_,
            )
            plt.xlabel("Predicted Topic")
            plt.ylabel("Actual Topic")
            plt.title("Confusion Matrix for PyTorch Neural Network Classification")
            plt.tight_layout()
            plt.show()

        # Create results DataFrame and show misclassified examples
        results_df = pd.DataFrame(
            {"text": X_test, "true_topic": y_test, "predicted_topic": predictions}
        )

        # Find misclassified examples
        misclassified = results_df[
            results_df["true_topic"] != results_df["predicted_topic"]
        ]
        misclassified_pct = len(misclassified) / len(results_df)
        print(
            f"\nMisclassified examples: {len(misclassified)}/{len(results_df)} ({misclassified_pct:.1%})"
        )

        # Display sample of misclassified examples
        if display_misclassified and len(misclassified) > 0:
            print("\nSample of misclassified examples:")
            sample_size = min(5, len(misclassified))
            print(
                misclassified[["text", "true_topic", "predicted_topic"]].sample(
                    sample_size
                )
            )

        # Return evaluation metrics
        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm,
            "misclassified_count": len(misclassified),
            "misclassified_percent": misclassified_pct,
        }


class TopicClassifier:
    """Main class for topic classification that can use different model implementations."""

    def __init__(self, model_name: str = "bayes", **model_params):
        """
        Initialize the topic classifier.

        Args:
            model_name: Name of the model to use ('bayes', 'xgboost', 'pytorch')
            **model_params: Parameters to pass to the model constructor
        """
        self.model_name = model_name.lower()
        self.model = None

        # Initialize the requested model
        if self.model_name == "bayes":
            self.model = BayesModel(**model_params)
        elif self.model_name == "xgboost":
            self.model = XGBoostModel(**model_params)
        elif self.model_name == "pytorch":
            self.model = PyTorchNNModel(**model_params)
        else:
            raise ValueError(
                f"Unknown model name: {model_name}. Supported models: 'bayes', 'xgboost', 'pytorch'"
            )

        print(f"Initialized {self.model_name} classifier")

    def train(
        self,
        train_data: pd.DataFrame,
        text_column: str = "review_text",
        label_column: str = "topic",
    ) -> "TopicClassifier":
        """
        Train the classifier on provided data.

        Args:
            train_data: DataFrame containing training data
            text_column: Name of column containing text
            label_column: Name of column containing topic labels

        Returns:
            Self for method chaining
        """
        if self.model is None:
            raise ValueError("Model not initialized correctly")

        X_train = train_data[text_column]
        y_train = train_data[label_column]

        self.model.fit(X_train, y_train)
        return self

    def predict(self, texts: Union[str, List[str], pd.Series]) -> np.ndarray:
        """
        Make predictions on new texts.

        Args:
            texts: Single text string, list of strings, or pandas Series

        Returns:
            Array of predicted topics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Handle different input types
        if isinstance(texts, str):
            texts = pd.Series([texts])
        elif isinstance(texts, list):
            texts = pd.Series(texts)

        return self.model.predict(texts)

    def evaluate(
        self,
        test_data: pd.DataFrame,
        text_column: str = "sentence",
        label_column: str = "topic",
        **kwargs,
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            test_data: DataFrame containing test data
            text_column: Name of column containing text
            label_column: Name of column containing topic labels
            **kwargs: Additional arguments to pass to model's evaluate method

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_test = test_data[text_column]
        y_test = test_data[label_column]

        return self.model.evaluate(X_test, y_test, **kwargs)
