import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from textblob import TextBlob
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from langdetect import detect
from googletrans import Translator
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# Read the Excel file
df = pd.read_excel('ik reviews.xlsx')

# Function to detect language with error handling
def detect_language(text):
    try:
        if pd.isna(text) or str(text).strip() == '':
            return 'unknown'
        return detect(str(text))
    except:
        return 'unknown'

# Print sample of reviews with their detected languages
print("\n=== Language Detection Sample ===")
sample_reviews = df[df['text'].notna()].sample(min(10, len(df)))
for _, row in sample_reviews.iterrows():
    text = row['text']
    lang = detect_language(text)
    print(f"\nLanguage: {lang}")
    print(f"Rating: {row['rating']} stars")
    print(f"Text: {text[:200]}...")  # Print first 200 characters

# Count languages in the dataset
df['detected_language'] = df['text'].apply(detect_language)
print("\n=== Language Distribution ===")
language_dist = df['detected_language'].value_counts()
print(language_dist)

# Print language statistics
print("\n=== Language Statistics ===")
for lang in language_dist.index:
    lang_reviews = df[df['detected_language'] == lang]
    avg_rating = lang_reviews['rating'].mean()
    review_count = len(lang_reviews)
    print(f"\nLanguage: {lang}")
    print(f"Number of reviews: {review_count}")
    print(f"Average rating: {avg_rating:.2f}")
    print(f"Empty reviews: {lang_reviews['empty'].sum()}")

# Initialize translator
translator = Translator()

# Function to translate text to English with detailed error handling
def translate_to_english(text, source_lang='auto'):
    try:
        if pd.isna(text) or str(text).strip() == '':
            return ''
        if source_lang == 'en' or source_lang == 'unknown':
            return text
        
        translation = translator.translate(text, dest='en', src=source_lang)
        return translation.text
    except Exception as e:
        print(f"Translation error: {str(e)[:100]}...")
        return text  # Return original text if translation fails

print("\nTranslating reviews to English (this may take a while)...")
# Translate non-English reviews with progress bar
df['text_english'] = df['text'].copy()
non_english_mask = (df['detected_language'] != 'en') & (df['detected_language'] != 'unknown') & df['text'].notna()

# Sample translations for verification
print("\n=== Translation Examples ===")
sample_translations = df[non_english_mask].sample(min(5, len(df[non_english_mask])))
for idx, row in sample_translations.iterrows():
    original_text = row['text']
    detected_lang = row['detected_language']
    translated_text = translate_to_english(original_text, detected_lang)
    
    print(f"\nOriginal Language: {detected_lang}")
    print(f"Rating: {row['rating']} stars")
    print(f"Original text: {original_text[:200]}")
    print(f"Translated text: {translated_text[:200]}")
    print("-" * 80)
    
    # Update the dataframe
    df.at[idx, 'text_english'] = translated_text

# Translate remaining non-English reviews
for idx, row in tqdm(df[non_english_mask].iterrows(), desc="Translating remaining reviews"):
    if idx not in sample_translations.index:  # Skip already translated samples
        df.at[idx, 'text_english'] = translate_to_english(row['text'], row['detected_language'])

# Enhanced Sentiment Analysis with translated text
def get_detailed_sentiment(text):
    if pd.isna(text):
        return {'polarity': 0, 'subjectivity': 0}
    blob = TextBlob(str(text))
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

# Add sentiment scores using translated text
print("\nPerforming sentiment analysis...")
sentiments = df['text_english'].apply(get_detailed_sentiment)
df['sentiment_polarity'] = sentiments.apply(lambda x: x['polarity'])
df['sentiment_subjectivity'] = sentiments.apply(lambda x: x['subjectivity'])

# Analyze sentiment patterns by language
print("\n=== Sentiment Analysis by Language ===")
for lang in df['detected_language'].unique():
    lang_data = df[df['detected_language'] == lang]
    if len(lang_data) > 0:
        print(f"\nLanguage: {lang}")
        print(f"Average sentiment polarity: {lang_data['sentiment_polarity'].mean():.3f}")
        print(f"Average sentiment subjectivity: {lang_data['sentiment_subjectivity'].mean():.3f}")
        print(f"Average rating: {lang_data['rating'].mean():.2f}")

# Save detailed analysis
with open('translation_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("IKEA Reviews Translation Analysis\n")
    f.write("==============================\n\n")
    
    f.write("1. Language Distribution\n")
    f.write("---------------------\n")
    f.write(str(language_dist) + "\n\n")
    
    f.write("2. Sample Translations\n")
    f.write("-------------------\n")
    for idx, row in sample_translations.iterrows():
        f.write(f"\nOriginal Language: {row['detected_language']}\n")
        f.write(f"Rating: {row['rating']} stars\n")
        f.write(f"Original text: {row['text']}\n")
        f.write(f"Translated text: {row['text_english']}\n")
        f.write("-" * 80 + "\n")
    
    f.write("\n3. Sentiment Analysis by Language\n")
    f.write("------------------------------\n")
    for lang in df['detected_language'].unique():
        lang_data = df[df['detected_language'] == lang]
        if len(lang_data) > 0:
            f.write(f"\nLanguage: {lang}\n")
            f.write(f"Number of reviews: {len(lang_data)}\n")
            f.write(f"Average sentiment polarity: {lang_data['sentiment_polarity'].mean():.3f}\n")
            f.write(f"Average rating: {lang_data['rating'].mean():.2f}\n")
            f.write(f"Rating distribution:\n{lang_data['rating'].value_counts().sort_index()}\n")

# Create visualizations
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='detected_language', y='sentiment_polarity')
plt.title('Sentiment Distribution by Language')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sentiment_by_language.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='detected_language', y='rating')
plt.title('Rating Distribution by Language')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('rating_by_language.png')
plt.close()

print("\nAnalysis complete! Check 'translation_analysis.txt' for detailed results and examples.") 