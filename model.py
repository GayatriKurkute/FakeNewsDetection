import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline

# Define model and tokenizer paths
MODEL_NAME = "distilbert-base-uncased"
# Path to your fine-tuned model. If the model is not found, a base model will be loaded.
# In a real-world scenario, you would fine-tune DistilBERT on a fake news dataset
# (e.g., LIAR, FakeNewsNet) and save it to this path.
FINETUNED_MODEL_PATH = "./fake_news_model" # Placeholder for fine-tuned model

class FakeNewsDetector:
    """
    A class to encapsulate the fake news detection model.
    Uses a pre-trained DistilBERT model from HuggingFace Transformers.
    """
    def __init__(self):
        """
        Initializes the tokenizer and loads the (potentially fine-tuned) DistilBERT model.
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        try:
            # Attempt to load a fine-tuned model
            self.model = DistilBertForSequenceClassification.from_pretrained(FINETUNED_MODEL_PATH)
            print(f"Successfully loaded fine-tuned model from {FINETUNED_MODEL_PATH}")
        except OSError:
            # If fine-tuned model not found, load a base model for demonstration purposes.
            # This base model will likely not perform well on fake news detection
            # until it's properly fine-tuned.
            print(f"Fine-tuned model not found at {FINETUNED_MODEL_PATH}. Loading base model for demonstration.")
            self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
            print("Please note: For optimal performance, fine-tune the model on a relevant dataset.")
        # Initialize the HuggingFace pipeline for sentiment-analysis (binary classification)
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def predict(self, text: str):
        """
        Predicts whether a given text is fake or real news using the loaded model.

        Args:
            text (str): The input text to be classified.

        Returns:
            tuple[float, str]: A tuple containing:
                - fake_probability (float): The probability (0-100%) that the news is fake.
                - prediction_label (str): The predicted label, either "FAKE" or "REAL".
        """
        # The pipeline returns a list of dictionaries, e.g., [{'label': 'LABEL_0', 'score': 0.999}]
        # We extract the label and score from the first (and only) result.
        result = self.pipeline(text)[0]
        label = result['label']
        score = result['score']

        # Map model's internal labels (e.g., 'LABEL_0', 'LABEL_1') to human-readable labels.
        # This mapping assumes 'LABEL_0' corresponds to 'REAL' and 'LABEL_1' to 'FAKE'.
        # This might need adjustment based on how the model was fine-tuned.
        if label == 'LABEL_0': # Typically, LABEL_0 represents the negative class (e.g., REAL)
            fake_probability = (1 - score) * 100
            prediction_label = "REAL"
        else: # Typically, LABEL_1 represents the positive class (e.g., FAKE)
            fake_probability = score * 100
            prediction_label = "FAKE"
            
        return fake_probability, prediction_label

# Example usage for direct testing of the model functionality
if __name__ == "__main__":
    print("\n--- Testing FakeNewsDetector directly ---")
    detector = FakeNewsDetector()

    # Test case 1: Likely fake news
    text_to_check_fake = "COVID-19 vaccine causes autism, widely debunked claim."
    prob_fake, label = detector.predict(text_to_check_fake)
    print(f"Text: '{text_to_check_fake}'")
    print(f"Probability of being FAKE: {prob_fake:.2f}%")
    print(f"Prediction: {label}\n")

    # Test case 2: Likely real news
    text_to_check_real = "Researchers announce breakthrough in cancer treatment."
    prob_fake_real, label_real = detector.predict(text_to_check_real)
    print(f"Text: '{text_to_check_real}'")
    print(f"Probability of being FAKE: {prob_fake_real:.2f}%")
    print(f"Prediction: {label_real}\n")