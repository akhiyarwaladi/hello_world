#!/usr/bin/env python3
"""
Script to remove all Unicode emoticons from Python files in the repository
"""

import os
import re
import glob

def remove_emoticons_from_file(file_path):
    """Remove emoticons from a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Pattern to match Unicode emoticons and symbols
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002600-\U000026FF"  # Miscellaneous Symbols
            "\U00002700-\U000027BF"  # Dingbats
            "\U0000FE00-\U0000FE0F"  # Variation Selectors
            "\U0001F018-\U0001F270"  # Various asian characters
            "]+",
            flags=re.UNICODE
        )

        # Remove emoticons
        cleaned_content = emoji_pattern.sub('', content)

        # Only write if content changed
        if cleaned_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"Cleaned: {file_path}")
            return True
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all Python files"""
    python_files = []

    # Find all Python files
    for root, dirs, files in os.walk('.'):
        # Skip .git directory
        if '.git' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    print(f"Found {len(python_files)} Python files")

    cleaned_count = 0
    for file_path in python_files:
        if remove_emoticons_from_file(file_path):
            cleaned_count += 1

    print(f"Cleaned {cleaned_count} files")

if __name__ == "__main__":
    main()