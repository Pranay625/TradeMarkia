"""
Example: Text Cleaning Demonstration

This script shows the before/after effect of text cleaning
on a sample newsgroup post.
"""

from src.text_cleaner import TextCleaner


# Sample raw newsgroup post with typical noise
SAMPLE_RAW_TEXT = """From: john.doe@university.edu (John Doe)
Subject: Re: Best programming languages for AI
Organization: University Computer Science Dept
Lines: 42
NNTP-Posting-Host: cs.university.edu
X-Newsreader: TIN [version 1.2 PL2]

> What are the best programming languages for AI development?
> I've heard Python is popular but what about others?

I think Python is definitely the most popular choice these days.
The ecosystem is fantastic with libraries like TensorFlow, PyTorch,
and scikit-learn.

> Is C++ still relevant for AI?

Yes, C++ is still used for performance-critical applications.
Many deep learning frameworks have C++ backends for speed.

You might also want to check out:
- Julia for numerical computing
- R for statistical analysis
- Java for enterprise applications

For more info, visit: https://www.example.com/ai-languages

Feel free to email me at john.doe@university.edu if you have questions.

--
John Doe
PhD Student, Computer Science
University of Technology
Email: john.doe@university.edu
Web: www.example.com/~johndoe
"""


def main():
    print("\n" + "="*70)
    print("TEXT CLEANING DEMONSTRATION")
    print("="*70)
    
    # Initialize cleaner
    cleaner = TextCleaner(lowercase=False, remove_numbers=False)
    
    # Show original text
    print("\n[BEFORE CLEANING]")
    print("-"*70)
    print(SAMPLE_RAW_TEXT)
    print("-"*70)
    print(f"Length: {len(SAMPLE_RAW_TEXT)} characters")
    print(f"Lines: {len(SAMPLE_RAW_TEXT.split(chr(10)))}")
    
    # Clean the text
    cleaned_text = cleaner.clean(SAMPLE_RAW_TEXT)
    
    # Show cleaned text
    print("\n[AFTER CLEANING]")
    print("-"*70)
    print(cleaned_text)
    print("-"*70)
    print(f"Length: {len(cleaned_text)} characters")
    print(f"Lines: {len(cleaned_text.split(chr(10)))}")
    
    # Show what was removed
    print("\n[CLEANING SUMMARY]")
    print("-"*70)
    print("[+] Removed email headers (From, Subject, Organization, etc.)")
    print("[+] Removed quoted replies (lines starting with >)")
    print("[+] Removed signature block (after --)")
    print("[+] Removed URLs and email addresses")
    print("[+] Normalized whitespace and blank lines")
    print(f"[+] Size reduction: {((len(SAMPLE_RAW_TEXT) - len(cleaned_text)) / len(SAMPLE_RAW_TEXT) * 100):.1f}%")
    print("-"*70)
    
    print("\n[SUCCESS] The cleaned text contains only the actual content!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
