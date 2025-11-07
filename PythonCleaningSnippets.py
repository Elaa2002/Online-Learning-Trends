import pandas as pd
import numpy as np
import re
import os

from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# Load the dataset
df = pd.read_csv('Online_Courses.csv')


def translate_text(text, target_language='en'):
    """Translate text to English using Google Translate"""
    if pd.isna(text) or text.strip() == '' or text == 'Info not available':
        return text

    text = str(text).strip()
    
    try:
        detected_lang = detect(text)
    except LangDetectException:
        detected_lang = 'unknown'


    if detected_lang != 'en' or any(ord(c) > 127 for c in text):
        try:
            translated = GoogleTranslator(source='auto', target=target_language).translate(text)
            if translated and translated.strip() != text.strip():
                print(f"Translated: '{text[:40]}...' → '{translated[:40]}...'")
                return translated
        except Exception as e:
            print(f" Translation failed for '{text[:40]}...': {e}")
            return text

    return text


def translate_coursera_columns(df):
    """Translate specific Coursera columns from non-English to English"""
    df = df.copy()
    

    columns_to_translate = ['Title', 'Category', 'Sub-Category']
    
    print("Starting translation for Coursera courses...")
    total_rows = len(df)
    
    for col in columns_to_translate:
        if col in df.columns:
            print(f"\nTranslating column: {col}")
            
            # Track progress
            translated_count = 0
            
            for idx, value in df[col].items():
                if pd.notna(value) and value != 'Info not available':
                    # Translate
                    translated = translate_text(value)
                    
                    # Only update if translation happened
                    if translated != value:
                        df.at[idx, col] = translated
                        translated_count += 1
                
                # Progress indicator every 100 rows
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{total_rows} rows...")
            
            print(f" Translated {translated_count} entries in {col}")
    
    return df

def remove_garbled_characters(text):
    """Remove garbled characters and clean up gibberish words"""
    if pd.isna(text):
        return text
    
    text = str(text)
    
    # Replace common garbled patterns with space
    # Remove special encoded characters like Â®, â„¢, Ã©, etc.
    text = re.sub(r'[ÂÃÄÅÆÇÈÉÊËÌÍÎÏ][\x80-\xff®™©]', ' ', text)
    text = re.sub(r'â[„¢€˜]', ' ', text)
    
    # Split into words and clean each word
    words = text.split()
    cleaned_words = []
    
    for word in words:
        # Count ratio of ASCII letters/numbers vs special characters
        ascii_chars = sum(1 for c in word if c.isalnum())
        special_chars = sum(1 for c in word if not c.isalnum() and not c.isspace())
        total_chars = len(word)
        
        if total_chars == 0:
            continue
        
        # If word is mostly gibberish (more than 30% special chars), skip it
        if special_chars > 0 and (special_chars / total_chars) > 0.3:
            continue
        
        # Otherwise, remove special characters from within the word
        # Keep only alphanumeric, spaces, and common punctuation
        cleaned_word = re.sub(r'[^\w\s\-\.\,\'\"]', ' ', word)
        cleaned_word = cleaned_word.strip()
        
        if cleaned_word:
            cleaned_words.append(cleaned_word)
    
    # Join words and clean up multiple spaces
    result = ' '.join(cleaned_words)
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def clean_text_column(series):
    """Remove _x000D_, garbled characters, and extra spaces from text columns"""
    if series.dtype == 'object':
        # First remove _x000D_
        series = series.str.replace('_x000D_', '', regex=False)
        # Then remove garbled characters
        series = series.apply(remove_garbled_characters)
        # Final cleanup - handle the string conversion more carefully
        series = series.fillna('')  # Temporarily fill NaN with empty string
        series = series.astype(str)  # Convert to string
        series = series.str.strip().str.replace(r'\s+', ' ', regex=True)
        series = series.replace('', np.nan)  # Convert empty strings back to NaN
        return series
    return series

def extract_duration_months(duration_str):
    """Extract number of months from duration string"""
    if pd.isna(duration_str):
        return np.nan
    
    # Convert to string and search for number followed by 'month'
    duration_str = str(duration_str).lower()
    
    # Try to find "X months" or "X month"
    match = re.search(r'(\d+)\s*months?', duration_str)
    if match:
        return int(match.group(1))
    
    # Try to find "X weeks"
    match = re.search(r'(\d+)\s*weeks?', duration_str)
    if match:
        return int(match.group(1))
    
    # Try to find just a number if it's the only thing
    match = re.search(r'(\d+)', duration_str)
    if match:
        return int(match.group(1))
    
    return np.nan

def clean_price(price_str):
    """Convert price to float, handle 'Free' as 0"""
    if pd.isna(price_str):
        return 0.0
    
    price_str = str(price_str).strip().lower()
    
    if 'free' in price_str or price_str == '0':
        return 0.0
    
    # Extract numeric value
    match = re.search(r'[\d,]+\.?\d*', price_str)
    if match:
        return float(match.group().replace(',', ''))
    
    return 0.0

def clean_rating(rating_str):
    """Remove 'stars' text and convert to float"""
    if pd.isna(rating_str):
        return np.nan
    
    rating_str = str(rating_str).lower().replace('stars', '').replace('star', '').strip()
    
    # If empty after cleaning, return NaN
    if not rating_str or rating_str == '':
        return np.nan
    
    try:
        rating = float(rating_str)
        # Validate rating is in reasonable range (0-5)
        if 0 <= rating <= 5:
            return rating
        else:
            return np.nan
    except:
        return np.nan

def clean_number_column(series):
    """Convert text numbers to integers"""
    if series.dtype == 'object':
        def parse_number(val):
            if pd.isna(val):
                return np.nan
            val = str(val).replace(',', '').strip()
            try:
                return int(float(val))
            except:
                return np.nan
        return series.apply(parse_number)
    return series

def extract_premium_price(text):
    """Extract price from Premium course text"""
    if pd.isna(text):
        return np.nan
    
    text = str(text)
    # Look for price pattern like $39 or 39
    match = re.search(r'\$?(\d+)\s*per month', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # Try to find any number
    match = re.search(r'\$?(\d+)', text)
    if match:
        return float(match.group(1))
    
    return np.nan

def is_column_meaningful(series):
    """Check if a column has meaningful data (not all empty or 'Info not available')"""
    try:
        # Check if all values are NaN - use bool() to convert to single boolean
        all_na = bool(series.isna().all())
        if all_na:
            return False
        
        # Get non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        
        # Convert to string and check for meaningful content
        non_null_str = non_null.astype(str)
        
        # Count meaningful values (not empty, not "Info not available")
        meaningful_mask = (
            (non_null_str != '') & 
            (non_null_str != 'Info not available') &
            (non_null_str.str.strip() != '')
        )
        
        meaningful_count = int(meaningful_mask.sum())
        
        # If less than 5% of rows have meaningful data, consider it not meaningful
        if meaningful_count / len(series) < 0.05:
            return False
        
        return True
    except Exception as e:
        # If there's any error, assume the column is meaningful to be safe
        print(f"Warning: Could not check if column is meaningful: {e}")
        return True

def clean_coursera(df):
    """Clean Coursera-specific columns"""
    df = df.copy()
    
    # IMPORTANT: Translate specific columns FIRST before other cleaning
    df = translate_coursera_columns(df)
    
    # Remove "subtitle: " and "Subtitles" from Subtitle Languages
    if 'Subtitle Languages' in df.columns:
        df['Subtitle Languages'] = df['Subtitle Languages'].str.replace('subtitle: ', '', regex=False, case=False)
        df['Subtitle Languages'] = df['Subtitle Languages'].str.replace('Subtitles', '', regex=False, case=True)
        df['Subtitle Languages'] = df['Subtitle Languages'].str.strip()
    
    # Clean Rating - remove "stars" and handle empty values
    if 'Rating' in df.columns:
        df['Rating'] = df['Rating'].apply(clean_rating)
        # Only fill if column has some meaningful data
        if df['Rating'].notna().sum() > 0:
            df['Rating'] = df['Rating'].fillna('No rating available')
    
    # Extract months from Duration and rename column
    if 'Duration' in df.columns:
        df['Duration'] = df['Duration'].apply(extract_duration_months)
        df = df.rename(columns={'Duration': 'Duration (months)'})
    
    # Remove Price column if it only contains 0's or is empty
    if 'Price' in df.columns:
        try:
            price_numeric = pd.to_numeric(df['Price'], errors='coerce')
            non_null_prices = price_numeric.dropna()
            if len(non_null_prices) == 0 or (non_null_prices == 0).all():
                df = df.drop(columns=['Price'])
        except:
            pass
    
    # Remove columns that are completely empty or have no meaningful data
    cols_to_drop = [col for col in df.columns if not is_column_meaningful(df[col])]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df

def clean_udacity(df):
    """Clean Udacity-specific columns"""
    df = df.copy()
    
    # Extract months/weeks from Duration and determine unit
    if 'Duration' in df.columns:
        sample_duration = df['Duration'].dropna().iloc[0] if len(df['Duration'].dropna()) > 0 else ''
        duration_unit = 'months'  # default
        
        if 'week' in str(sample_duration).lower():
            duration_unit = 'weeks'
        elif 'month' in str(sample_duration).lower():
            duration_unit = 'months'
        
        df['Duration'] = df['Duration'].apply(extract_duration_months)
        df = df.rename(columns={'Duration': f'Duration ({duration_unit})'})
    
    # Rename School to Category
    if 'School' in df.columns:
        df = df.rename(columns={'School': 'Category'})
    
    # Remove Price column
    if 'Price' in df.columns:
        df = df.drop(columns=['Price'])
    
    # Remove columns that are completely empty or have no meaningful data first
    cols_to_drop = [col for col in df.columns if not is_column_meaningful(df[col])]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # For remaining columns with some data, fill empty cells with "Info not available"
    for col in df.columns:
        try:
            if df[col].notna().sum() > 0 and df[col].isna().sum() > 0:
                df[col] = df[col].fillna('Info not available')
        except:
            continue
    
    return df

def clean_other_sites(df):
    """Clean Future Learn and other sites"""
    df = df.copy()
    
    site_name = df['Site'].iloc[0] if 'Site' in df.columns and len(df) > 0 else ''
    
    # Future Learn specific cleaning
    if 'future learn' in site_name.lower():
        # Remove duplicate URL columns - keep only one
        if 'URL' in df.columns and 'Course URL' in df.columns:
            # Check if they're identical
            if df['URL'].equals(df['Course URL']):
                df = df.drop(columns=['Course URL'])
            elif df['Course URL'].notna().sum() > df['URL'].notna().sum():
                # If Course URL has more data, keep it and drop URL
                df = df.drop(columns=['URL'])
                df = df.rename(columns={'Course URL': 'URL'})
            else:
                # Otherwise keep URL and drop Course URL
                df = df.drop(columns=['Course URL'])
        
        # Extract duration and determine unit
        if 'Duration' in df.columns:
            sample_duration = df['Duration'].dropna().iloc[0] if len(df['Duration'].dropna()) > 0 else ''
            duration_unit = 'weeks'  # default for Future Learn
            
            if 'week' in str(sample_duration).lower():
                duration_unit = 'weeks'
            elif 'month' in str(sample_duration).lower():
                duration_unit = 'months'
            
            df['Duration'] = df['Duration'].apply(extract_duration_months)
            df = df.rename(columns={'Duration': f'Duration ({duration_unit})'})
        
        # Courses column - remove "courses" word
        if 'Courses' in df.columns:
            df['Courses'] = df['Courses'].str.replace('courses', '', regex=False, case=False)
            df['Courses'] = df['Courses'].str.replace('course', '', regex=False, case=False)
            df['Courses'] = df['Courses'].str.strip()
        
        # ExpertTrack - remove "ExpertTrack" word, keep only number
        if 'ExpertTrack' in df.columns or 'ExpertTracks' in df.columns:
            col_name = 'ExpertTrack' if 'ExpertTrack' in df.columns else 'ExpertTracks'
            df[col_name] = df[col_name].apply(lambda x: re.sub(r'[^\d]', '', str(x)) if pd.notna(x) else x)
        
        # Weekly study - remove 'hours' and 'hour' and add to column name
        if 'Weekly study' in df.columns:
            df['Weekly study'] = df['Weekly study'].str.replace('hours', '', regex=False, case=False)
            df['Weekly study'] = df['Weekly study'].str.replace('hour', '', regex=False, case=False)
            df['Weekly study'] = df['Weekly study'].str.strip()
            df = df.rename(columns={'Weekly study': 'Weekly study (hours)'})
        
        # Premium course - extract only the price (39)
        if 'Premium course' in df.columns:
            df['Premium course'] = df['Premium course'].apply(extract_premium_price)
        
        # Delete specific columns
        cols_to_delete = ['FAQs', "What's include", 'Price']
        for col in cols_to_delete:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Remove columns with no meaningful data
        cols_to_drop = [col for col in df.columns if not is_column_meaningful(df[col])]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
    
    # Simplilearn specific cleaning
    elif 'simplilearn' in site_name.lower():
        # Remove specific columns
        cols_to_delete = ['Rank', 'Created by']
        for col in cols_to_delete:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Extract duration
        if 'Duration' in df.columns:
            df['Duration'] = df['Duration'].apply(extract_duration_months)
        
        # Remove columns with no meaningful data
        cols_to_drop = [col for col in df.columns if not is_column_meaningful(df[col])]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
    
    else:
        # Generic cleaning for other sites
        if 'Duration' in df.columns:
            df['Duration'] = df['Duration'].apply(extract_duration_months)
        
        # Remove columns with no meaningful data
        cols_to_drop = [col for col in df.columns if not is_column_meaningful(df[col])]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
    
    return df

def apply_common_cleaning(df):
    """Apply cleaning steps common to all sites"""
    df = df.copy()
    
    # Get list of columns at start to avoid issues with column renaming
    # Use pd.api.types.is_object_dtype to safely check column types
    text_columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]
    
    # Clean all text columns - remove _x000D_ and extra spaces
    for col in text_columns:
        try:
            df[col] = clean_text_column(df[col])
        except Exception as e:
            print(f"Warning: Could not clean column '{col}': {e}")
            continue
    
    # Convert numeric columns to proper types
    if 'Number of viewers' in df.columns and pd.api.types.is_object_dtype(df['Number of viewers']):
        df['Number of viewers'] = clean_number_column(df['Number of viewers'])
    
    if 'Number of Reviews' in df.columns and pd.api.types.is_object_dtype(df['Number of Reviews']):
        df['Number of Reviews'] = clean_number_column(df['Number of Reviews'])
    
    if 'Rating' in df.columns and pd.api.types.is_object_dtype(df['Rating']):
        df['Rating'] = df['Rating'].apply(clean_rating)
    
    if 'Duration' in df.columns and pd.api.types.is_object_dtype(df['Duration']):
        df['Duration'] = df['Duration'].apply(extract_duration_months)
    
    if 'Price' in df.columns:
        df['Price'] = df['Price'].apply(clean_price)
    
    return df

# Main processing loop
print("Starting dataset cleaning...")
print(f"Total courses: {len(df)}")

# Group by Site and process each website separately
sites = df['Site'].unique()
print(f"\nFound {len(sites)} sites: {', '.join(sites)}")

# Create Excel writer object for output
output_file = 'All_Courses_Cleaned.xlsx'
writer = pd.ExcelWriter(output_file, engine='openpyxl')

for site in sites:
    print(f"\n{'='*50}")
    print(f"Processing: {site}")
    print(f"{'='*50}")
    
    # Filter data for this site
    site_df = df[df['Site'] == site].copy()
    print(f"Courses from {site}: {len(site_df)}")
    
    # Apply site-specific cleaning
    if 'coursera' in site.lower():
        site_df = clean_coursera(site_df)
    elif 'udacity' in site.lower():
        site_df = clean_udacity(site_df)
    else:
        site_df = clean_other_sites(site_df)
    
    # Apply common cleaning to all sites
    site_df = apply_common_cleaning(site_df)
    
    # No need for additional empty column removal here - already done in site-specific functions
    print(f"Final columns: {len(site_df.columns)}")
    print(f"Columns: {', '.join(site_df.columns[:10])}{'...' if len(site_df.columns) > 10 else ''}")
    
    # Create proper sheet name (keep original site name, replace spaces with underscores)
    # Excel has 31 char limit
    sheet_name = site.replace(' ', '_')
    if len(sheet_name) > 31:
        sheet_name = sheet_name[:31]
    
    # Write to Excel sheet
    site_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"✓ Added to Excel sheet: '{sheet_name}'")

# Save and close the Excel file
writer.close()

print("\n" + "="*50)
print(f"✓ Dataset cleaning completed successfully!")
print(f"✓ All cleaned datasets saved in '{output_file}'")
print(f"✓ Total sheets: {len(sites)}")
print("="*50)