# src/text_cleaning.py
import re

def clean_text(text):
    """
    Clean complaint narrative text
    
    Args:
        text (str): Raw complaint text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove boilerplate phrases
    boilerplate_phrases = [
        r'i am writing to file a complaint',
        r'i am writing to complain about',
        r'this is a complaint about',
        r'i would like to file a complaint',
        r'i want to complain about',
        r'i am submitting this complaint',
        r'this complaint concerns',
        r'i hereby file a complaint',
        r'please accept this as my complaint',
        r'this is to inform you',
        r'i am writing to bring to your attention',
        r'dear [a-z\s]+?,',
    ]
    
    for phrase in boilerplate_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep periods and spaces
    text = re.sub(r'[^a-zA-Z\s\.]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def standardize_product(product_name):
    """Standardize product names"""
    if pd.isna(product_name):
        return None
    product_name_lower = str(product_name).lower()
    
    if 'credit card' in product_name_lower:
        return 'Credit card'
    elif 'personal loan' in product_name_lower:
        return 'Personal loan'
    elif 'savings account' in product_name_lower or 'saving account' in product_name_lower:
        return 'Savings account'
    elif 'money transfer' in product_name_lower or 'wire transfer' in product_name_lower:
        return 'Money transfer'
    else:
        return None