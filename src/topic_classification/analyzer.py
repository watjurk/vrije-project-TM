import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Dict, List, Tuple, Optional, Union
import os


class ErrorAnalyzer:
    """Class for performing basic error analysis on topic classification models."""

    def __init__(self, model_name: str = "unnamed_model"):
        """
        Initialize the error analyzer.

        Args:
            model_name: Name of the model being analyzed
        """
        self.model_name = model_name
        self.results_df = None
        self.class_names = None
        self.error_analysis = {}

    def analyze(
        self,
        texts: pd.Series,
        true_labels: pd.Series,
        predicted_labels: pd.Series,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Analyze prediction errors and store results.

        Args:
            texts: Series containing text data
            true_labels: Series containing true topic labels
            predicted_labels: Series containing predicted topic labels
            probabilities: Optional array of prediction probabilities (if available)

        Returns:
            Dictionary with error analysis results
        """
        # Store class names
        if isinstance(true_labels, pd.Series):
            unique_true = true_labels.unique()
        else:
            unique_true = np.unique(true_labels)

        if isinstance(predicted_labels, pd.Series):
            unique_pred = predicted_labels.unique()
        else:
            unique_pred = np.unique(predicted_labels)

        self.class_names = sorted(list(set(unique_true).union(set(unique_pred))))

        # Create results DataFrame
        self.results_df = pd.DataFrame(
            {
                "text": texts,
                "true_label": true_labels,
                "predicted_label": predicted_labels,
                "is_correct": true_labels == predicted_labels,
            }
        )

        # Add confidence scores if probabilities are provided
        if probabilities is not None:
            if len(probabilities.shape) > 1:  # Multi-class probabilities
                # Get the probability of the predicted class for each sample
                confidence_scores = np.max(probabilities, axis=1)
                self.results_df["confidence"] = confidence_scores

        # Compute basic metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels, labels=self.class_names)
        report = classification_report(true_labels, predicted_labels, output_dict=True)

        # Find misclassified examples
        misclassified_df = self.results_df[~self.results_df["is_correct"]]
        misclassified_pct = len(misclassified_df) / len(self.results_df)

        # Error transition counts (from true label to predicted label)
        error_transitions = (
            misclassified_df.groupby(["true_label", "predicted_label"])
            .size()
            .reset_index(name="count")
        )

        # Store analysis results
        self.error_analysis = {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "classification_report": report,
            "misclassified_count": len(misclassified_df),
            "misclassified_percent": misclassified_pct,
            "error_transitions": error_transitions,
        }

        print(f"Error analysis complete for model: {self.model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(
            f"Misclassified examples: {len(misclassified_df)}/{len(self.results_df)} ({misclassified_pct:.1%})"
        )

        return self.error_analysis

    def plot_confusion_matrix(
        self,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ):
        """
        Plot normalized confusion matrix.

        Args:
            figsize: Size of the figure
            save_path: Path to save the figure (if provided)
        """
        if self.error_analysis is None or "confusion_matrix" not in self.error_analysis:
            raise ValueError("No analysis results available. Run analyze() first.")

        cm = self.error_analysis["confusion_matrix"]

        # Normalize the confusion matrix
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = f"Normalized Confusion Matrix - {self.model_name}"

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_error_distribution(
        self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None
    ):
        """
        Plot distribution of errors by class.

        Args:
            figsize: Size of the figure
            save_path: Path to save the figure (if provided)
        """
        if self.results_df is None:
            raise ValueError("No analysis results available. Run analyze() first.")

        # Calculate error rates by class
        class_errors = (
            self.results_df.groupby("true_label")
            .apply(lambda x: (x["true_label"] != x["predicted_label"]).mean())
            .sort_values(ascending=False)
        )

        # Calculate sample counts by class
        class_counts = self.results_df["true_label"].value_counts()
        class_counts = class_counts.loc[class_errors.index]  # Match the order

        # Create subplot with dual axes
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        # Plot error rates as bars
        bars = ax1.bar(
            class_errors.index, class_errors.values, alpha=0.7, color="crimson"
        )
        ax1.set_ylabel("Error Rate", color="crimson")
        ax1.set_ylim(
            0, min(1.0, class_errors.max() * 1.2)
        )  # Set y-axis limit with some margin
        ax1.tick_params(axis="y", labelcolor="crimson")

        # Plot sample counts as line
        line = ax2.plot(
            class_errors.index,
            class_counts.values,
            marker="o",
            color="navy",
            label="Sample Count",
        )
        ax2.set_ylabel("Sample Count", color="navy")
        ax2.tick_params(axis="y", labelcolor="navy")

        plt.title(f"Error Distribution by Class - {self.model_name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Error distribution plot saved to {save_path}")

        plt.show()

    def get_common_error_pairs(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get the most common error transition pairs (from true label to predicted label).

        Args:
            top_n: Number of top error pairs to return

        Returns:
            DataFrame with common error transitions
        """
        if (
            self.error_analysis is None
            or "error_transitions" not in self.error_analysis
        ):
            raise ValueError("No analysis results available. Run analyze() first.")

        error_transitions = self.error_analysis["error_transitions"]
        return error_transitions.sort_values("count", ascending=False).head(top_n)

    def get_error_examples(
        self,
        true_label: Optional[str] = None,
        predicted_label: Optional[str] = None,
        n_examples: int = 5,
    ) -> pd.DataFrame:
        """
        Get examples of specific error types.

        Args:
            true_label: Filter by true label (optional)
            predicted_label: Filter by predicted label (optional)
            n_examples: Number of examples to return

        Returns:
            DataFrame with error examples
        """
        if self.results_df is None:
            raise ValueError("No analysis results available. Run analyze() first.")

        # Get misclassified examples
        errors = self.results_df[~self.results_df["is_correct"]]

        # Apply filters
        if true_label is not None:
            errors = errors[errors["true_label"] == true_label]
        if predicted_label is not None:
            errors = errors[errors["predicted_label"] == predicted_label]

        return errors[["text", "true_label", "predicted_label"]].head(n_examples)

    def save_error_analysis(self, output_dir: str):
        """
        Save classification results to file.

        Args:
            output_dir: Directory to save the output files
        """
        if self.results_df is None:
            raise ValueError("No analysis results available. Run analyze() first.")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save full results DataFrame
        self.results_df.to_csv(
            os.path.join(output_dir, f"{self.model_name}_results.csv"), index=False
        )

        # Save confusion matrix visualization
        self.plot_confusion_matrix(
            save_path=os.path.join(
                output_dir, f"{self.model_name}_confusion_matrix.png"
            ),
        )

        # Save error distribution visualization
        self.plot_error_distribution(
            save_path=os.path.join(
                output_dir, f"{self.model_name}_error_distribution.png"
            ),
        )

        print(f"Analysis results saved to {output_dir}")
