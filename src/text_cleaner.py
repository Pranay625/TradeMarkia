"""
Text Cleaner Module

This module provides text preprocessing and cleaning functionality.

Responsibilities:
- Remove email headers (From:, Subject:, Organization:, etc.)
- Remove quoted replies (lines starting with >)
- Remove excessive blank lines
- Normalize whitespace
- Handle noisy newsgroup text

Key Functions:
- clean(): Main cleaning pipeline
- remove_headers(): Remove email headers
- remove_quotes(): Remove quoted text
- normalize_whitespace(): Clean up spacing
"""

import re
from typing import List


class TextCleaner:
    """Cleans and preprocesses text data for embedding generation."""
    
    def __init__(self, lowercase: bool = False, remove_numbers: bool = False):
        """
        Initialize text cleaner with configuration.
        
        Args:
            lowercase: Convert text to lowercase
            remove_numbers: Remove numeric characters
        """
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        
        # Common email header patterns
        self.header_patterns = [
            r'^From:.*$',
            r'^Subject:.*$',
            r'^Organization:.*$',
            r'^Lines:.*$',
            r'^Distribution:.*$',
            r'^NNTP-Posting-Host:.*$',
            r'^Keywords:.*$',
            r'^Summary:.*$',
            r'^Expires:.*$',
            r'^Sender:.*$',
            r'^Reply-To:.*$',
            r'^Followup-To:.*$',
            r'^Date:.*$',
            r'^Article-I\.D\.?:.*$',
            r'^Posted:.*$',
            r'^Posting-Version:.*$',
            r'^Relay-Version:.*$',
            r'^X-.*:.*$',  # X- headers
        ]
    
    def clean(self, text: str) -> str:
        """
        Clean a single text document.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Step 1: Remove email headers
        # Reasoning: Headers contain metadata, not content
        text = self.remove_headers(text)
        
        # Step 2: Remove quoted replies
        # Reasoning: Quotes are duplicated content from previous messages
        text = self.remove_quotes(text)
        
        # Step 3: Remove signature blocks
        # Reasoning: Signatures are boilerplate, not relevant content
        text = self.remove_signatures(text)
        
        # Step 4: Remove URLs and email addresses
        # Reasoning: URLs/emails are not semantic content
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        
        # Step 5: Normalize whitespace
        # Reasoning: Clean up excessive spacing and blank lines
        text = self.normalize_whitespace(text)
        
        # Optional: lowercase
        if self.lowercase:
            text = text.lower()
        
        # Optional: remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def remove_headers(self, text: str) -> str:
        """
        Remove email headers from text.
        
        Headers appear at the beginning of newsgroup posts and contain
        metadata like From:, Subject:, Organization:, etc.
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Check if line matches any header pattern
            is_header = False
            for pattern in self.header_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_header = True
                    break
            
            if not is_header:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def remove_quotes(self, text: str) -> str:
        """
        Remove quoted replies (lines starting with >).
        
        In newsgroups, quoted text from previous messages starts with >
        These are not original content and should be removed.
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that start with > (quoted text)
            if not line.strip().startswith('>'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def remove_signatures(self, text: str) -> str:
        """
        Remove signature blocks.
        
        Signatures typically start with -- or ___ and contain
        contact information, disclaimers, etc.
        """
        # Common signature delimiters
        sig_patterns = [
            r'\n--\s*\n',  # Standard signature delimiter
            r'\n_{3,}\n',  # Underscores
            r'\n={3,}\n',  # Equal signs
        ]
        
        for pattern in sig_patterns:
            # Split on signature delimiter and keep only first part
            parts = re.split(pattern, text, maxsplit=1)
            if len(parts) > 1:
                text = parts[0]
        
        return text
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        URLs don't contribute to semantic meaning in most cases.
        """
        # Match http://, https://, ftp://, www.
        url_pattern = r'https?://\S+|ftp://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    def remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text.
        
        Email addresses are PII and don't contribute to content semantics.
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace characters.
        
        - Replace multiple spaces with single space
        - Replace multiple newlines with single newline
        - Remove leading/trailing whitespace from lines
        """
        # Split into lines and strip each line
        lines = [line.strip() for line in text.split('\n')]
        
        # Remove empty lines
        lines = [line for line in lines if line]
        
        # Join with single newline
        text = '\n'.join(lines)
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean multiple documents efficiently.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of cleaned text strings
        """
        return [self.clean(text) for text in texts]
