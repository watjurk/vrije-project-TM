from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import os
from typing import List, Optional, Dict, Union


class TrainingDataConstructor:
    """Class responsible for constructing training datasets for topic classification."""

    def __init__(self, books_topic_path: str, movies_topic_path: str):
        """
        Initialize the training data constructor.

        Args:
            books_topic_path: Path to the books topic data file
            movies_topic_path: Path to the movies topic data file
        """
        self.books_topic_path = books_topic_path
        self.movies_topic_path = movies_topic_path

    def fetch_sport_data(
        self,
        source: str = "newsgroups",
        jsonl_path: Optional[str] = None,
        nrows: int = 10000,
    ) -> pd.DataFrame:
        """
        Fetch sports data from specified source.

        Args:
            source: Data source, either "newsgroups" or "jsonl"
            jsonl_path: Path to JSONL file (required if source is "jsonl")
            nrows: Maximum number of rows to load from JSONL

        Returns:
            DataFrame containing sports data with review_text and topic columns

        Raises:
            ValueError: If invalid source is provided or missing required parameters
        """
        if source == "newsgroups":
            return self._fetch_from_newsgroups()
        elif source == "jsonl":
            if not jsonl_path:
                raise ValueError("jsonl_path must be provided when source='jsonl'")
            return self._fetch_from_jsonl(jsonl_path, nrows)
        else:
            raise ValueError("source must be either 'newsgroups' or 'jsonl'")

    def _fetch_from_newsgroups(self) -> pd.DataFrame:
        """
        Fetch sports data from 20 newsgroups dataset.

        Returns:
            DataFrame with sports articles
        """
        # Choose only sport categories
        sport_categories = ["rec.sport.baseball", "rec.sport.hockey"]
        print(f"\nSelected sport categories: {sport_categories}")

        # Fetch only the sports data
        sports_news = fetch_20newsgroups(
            subset="train",
            categories=sport_categories,
            remove=("headers", "footers", "quotes"),
            random_state=42,
        )

        print(f"\nFetched {len(sports_news.data)} sports articles")

        sports_df = pd.DataFrame({"review_text": sports_news.data, "topic": "sports"})

        # Clean up any empty entries
        sports_df = sports_df.dropna(subset=["review_text"])

        return sports_df

    def _fetch_from_jsonl(self, jsonl_path: str, nrows: int) -> pd.DataFrame:
        """
        Fetch sports data from JSONL file.

        Args:
            jsonl_path: Path to the JSONL file
            nrows: Maximum number of rows to read

        Returns:
            DataFrame with sports reviews

        Raises:
            FileNotFoundError: If JSONL file doesn't exist
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        print(f"Loading sports data from JSONL file: {jsonl_path}")
        # Read the JSONL file directly with pandas
        raw_df = pd.read_json(jsonl_path, lines=True, nrows=nrows)
        print(f"Read {len(raw_df)} reviews from file")

        # Filter for non-null text values
        valid_reviews = raw_df[raw_df["text"].notna()]["text"]

        # Create a DataFrame with the same structure as other topic dataframes
        sports_df = pd.DataFrame({"review_text": valid_reviews, "topic": "sports"})

        print(f"Loaded {len(sports_df)} sport articles from JSONL file")

        # Clean up any empty entries
        sports_df = sports_df.dropna(subset=["review_text"])
        return sports_df

    def fetch_book_data(self, nrows: int = 10000) -> pd.DataFrame:
        """
        Fetch book review data from CSV file.

        Args:
            nrows: Maximum number of rows to read

        Returns:
            DataFrame containing book reviews with review_text and topic columns

        Raises:
            FileNotFoundError: If books file doesn't exist
        """
        if not os.path.exists(self.books_topic_path):
            raise FileNotFoundError(
                f"Books data file not found: {self.books_topic_path}"
            )

        print(f"Loading book data from: {self.books_topic_path}")
        books_df = pd.read_csv(self.books_topic_path, nrows=nrows)
        print(f"Read {len(books_df)} book reviews")

        book_topics_df = pd.DataFrame(
            {
                "review_text": books_df["review/text"],
                "topic": "book",  # Add 'book' as topic label for all entries
            }
        )
        # Clean up any empty entries
        result_df = book_topics_df.dropna(subset=["review_text"])
        print(f"Loaded {len(result_df)} valid book reviews")
        return result_df

    def fetch_movies_data(self, nrows: int = 10000) -> pd.DataFrame:
        """
        Fetch movie review data from CSV file.

        Args:
            nrows: Maximum number of rows to read

        Returns:
            DataFrame containing movie reviews with review_text and topic columns

        Raises:
            FileNotFoundError: If movies file doesn't exist
        """
        if not os.path.exists(self.movies_topic_path):
            raise FileNotFoundError(
                f"Movies data file not found: {self.movies_topic_path}"
            )

        print(f"Loading movie data from: {self.movies_topic_path}")
        movies_df = pd.read_csv(self.movies_topic_path, nrows=nrows)
        print(f"Read {len(movies_df)} movie reviews")

        movie_topics_df = pd.DataFrame(
            {"review_text": movies_df["review"], "topic": "movie"}
        )

        # Clean up any empty entries
        result_df = movie_topics_df.dropna(subset=["review_text"])
        print(f"Loaded {len(result_df)} valid movie reviews")
        return result_df

    def create_combined_dataset(
        self,
        nrows: int = 10000,
        sports_source: str = "newsgroups",
        sports_jsonl_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create a combined dataset of books, movies, and sports data.

        Args:
            nrows: Maximum number of rows to read from each source
            sports_source: Source for sports data ("newsgroups" or "jsonl")
            sports_jsonl_path: Path to sports JSONL file (if using "jsonl" source)
            output_path: Optional path to save the combined dataset as CSV

        Returns:
            Combined DataFrame with all topic data

        Raises:
            ValueError: If invalid parameters are provided
        """
        print("Creating combined topic dataset...")

        # Fetch data from all three sources
        books_df = self.fetch_book_data(nrows=nrows)
        movies_df = self.fetch_movies_data(nrows=nrows)
        sports_df = self.fetch_sport_data(
            source=sports_source, jsonl_path=sports_jsonl_path, nrows=nrows
        )

        # Combine all data into one DataFrame
        combined_df = pd.concat([books_df, movies_df, sports_df], ignore_index=True)
        print(f"Combined dataset created with {len(combined_df)} entries")

        # Save to file if output path is provided
        if output_path:
            print(f"Saving combined dataset to {output_path}")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            combined_df.to_csv(output_path, index=False)
            print("Dataset saved successfully")

        return combined_df

    def get_dataset_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Generate basic statistics about the dataset for presentation.

        Args:
            data: Combined dataset with 'review_text' and 'topic' columns

        Returns:
            Dictionary containing key statistics about the dataset
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("Generating basic dataset statistics...")

        # Initialize statistics dictionary
        stats = {}

        # 1. Overall dataset size
        stats["total_samples"] = len(data)

        # 2. Class distribution
        class_counts = data["topic"].value_counts()
        stats["class_distribution"] = class_counts.to_dict()

        # 3. Calculate text length statistics
        data["text_length"] = data["review_text"].apply(len)
        data["word_count"] = data["review_text"].apply(lambda x: len(str(x).split()))

        # 4. Text statistics by topic
        topic_stats = {}
        for topic in data["topic"].unique():
            topic_data = data[data["topic"] == topic]
            topic_stats[topic] = {
                "count": len(topic_data),
                "avg_text_length": round(topic_data["text_length"].mean(), 1),
                "avg_word_count": round(topic_data["word_count"].mean(), 1),
                "median_text_length": int(topic_data["text_length"].median()),
                "median_word_count": int(topic_data["word_count"].median()),
            }
        stats["topic_stats"] = topic_stats

        # 5. Print summary for quick reference
        print("\nDataset Statistics Summary:")
        print(f"Total samples: {stats['total_samples']}")

        print("\nClass distribution:")
        for topic, count in class_counts.items():
            print(f"  {topic}: {count} samples ({count/len(data):.1%})")

        print("\nText length by topic:")
        for topic, topic_stat in topic_stats.items():
            print(
                f"  {topic}: {topic_stat['avg_word_count']} words (avg), {topic_stat['median_word_count']} words (median)"
            )

        # 6. Create a basic visualization
        plt.figure(figsize=(10, 6))

        # Plot average word counts by topic
        topic_names = list(topic_stats.keys())
        avg_words = [
            stats["topic_stats"][topic]["avg_word_count"] for topic in topic_names
        ]

        sns.barplot(x=topic_names, y=avg_words)
        plt.title("Average Text Length by Topic")
        plt.xlabel("Topic")
        plt.ylabel("Average Word Count")
        plt.tight_layout()
        plt.show()

        return stats

    def tokenize_text(
        self, data: pd.DataFrame, text_column: str = "review_text"
    ) -> pd.DataFrame:
        """
        Tokenize text data in preparation for TF-IDF vectorization.

        Args:
            data: DataFrame containing text data
            text_column: Name of the column containing text to tokenize

        Returns:
            DataFrame with additional 'tokens' column containing tokenized text
        """
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        # Download necessary NLTK resources if not already available
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")

        print("Tokenizing text data...")

        # Initialize lemmatizer and stopwords
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        def process_text(text):
            if not isinstance(text, str):
                return []

            # Convert to lowercase
            text = text.lower()

            # Tokenize
            tokens = word_tokenize(text)

            # Remove stopwords and punctuation, lemmatize
            tokens = [
                lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalpha() and token not in stop_words
            ]

            return tokens

        # Apply tokenization to the text column
        data["tokens"] = data[text_column].apply(process_text)

        # Print sample of tokenized text
        if len(data) > 0:
            print("\nSample tokenization:")
            sample_idx = min(5, len(data) - 1)
            sample_text = data[text_column].iloc[sample_idx]
            if len(sample_text) > 100:
                sample_text = sample_text[:100] + "..."
            sample_tokens = data["tokens"].iloc[sample_idx][:10]
            print(f"Original: {sample_text}")
            print(f"Tokenized: {sample_tokens}")
            if len(data["tokens"].iloc[sample_idx]) > 10:
                print("...")

        print(f"Tokenization complete. Added 'tokens' column to DataFrame.")

        return data
