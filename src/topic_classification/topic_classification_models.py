from sklearn import naive_bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
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
        else:
            raise ValueError(
                f"Unknown model name: {model_name}. Supported models: 'bayes'"
            )

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
