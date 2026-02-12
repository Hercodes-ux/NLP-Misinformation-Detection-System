import re
import string

class TextPreprocessor:
    """
    Hercodes-ux: Professional NLP Preprocessing Utility.
    Focuses on noise reduction and text standardization.
    """
    def clean_text(self, text):
        # 1. Convert to lowercase
        text = text.lower()
        # 2. Remove URLs (common in fake news)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # 3. Remove punctuation
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        # 4. Remove numbers
        text = re.sub(r'\d+', '', text)
        return text.strip()

if __name__ == "__main__":
    sample = "BREAKING: Check out this link https://fake-news.com !! 1234"
    cleaner = TextPreprocessor()
    print(f"Original: {sample}")
    print(f"Cleaned: {cleaner.clean_text(sample)}")