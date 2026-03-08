# Data Ingestion Pipeline

## Overview

The data ingestion pipeline loads and cleans the 20 Newsgroups dataset for semantic processing.

## Components

### 1. Data Loader (`src/data_loader.py`)

**Purpose**: Load documents from disk and organize by category

**Features**:
- Iterates through category folders
- Reads documents with encoding error handling (`errors='ignore'`)
- Extracts category names from folder structure
- Applies text cleaning automatically
- Returns structured document list

**Usage**:
```python
from src.data_loader import NewsgroupsDataLoader

loader = NewsgroupsDataLoader("data/20_newsgroups", apply_cleaning=True)
documents = loader.load()
loader.print_stats()

# Output format:
# [
#   {"category": "rec.autos", "text": "cleaned document text..."},
#   {"category": "sci.space", "text": "cleaned document text..."},
#   ...
# ]
```

### 2. Text Cleaner (`src/text_cleaner.py`)

**Purpose**: Remove noise from newsgroup posts

**Cleaning Steps**:

1. **Remove Email Headers** ✂️
   - Removes: `From:`, `Subject:`, `Organization:`, `Lines:`, `X-*:`, etc.
   - Reasoning: Headers are metadata, not content

2. **Remove Quoted Replies** ✂️
   - Removes: Lines starting with `>`
   - Reasoning: Quotes are duplicated content from previous messages

3. **Remove Signatures** ✂️
   - Removes: Text after `--`, `___`, or `===`
   - Reasoning: Signatures are boilerplate (contact info, disclaimers)

4. **Remove URLs** ✂️
   - Removes: `http://`, `https://`, `ftp://`, `www.`
   - Reasoning: URLs don't contribute to semantic meaning

5. **Remove Email Addresses** ✂️
   - Removes: `user@domain.com` patterns
   - Reasoning: Email addresses are PII and not semantic content

6. **Normalize Whitespace** 🧹
   - Removes: Excessive blank lines and spaces
   - Reasoning: Clean formatting for consistent processing

**Usage**:
```python
from src.text_cleaner import TextCleaner

cleaner = TextCleaner(lowercase=False, remove_numbers=False)
cleaned_text = cleaner.clean(raw_text)
```

## Testing

### Run the demo cleaning script:
```bash
python demo_cleaning.py
```

This shows before/after cleaning on a sample newsgroup post.

### Run the full pipeline test:
```bash
python test_data_pipeline.py
```

This loads the entire dataset and displays:
- Total documents loaded
- Category distribution
- Cleaning impact statistics
- Sample cleaned text

## Expected Output

### Dataset Statistics Example:
```
============================================================
20 NEWSGROUPS DATASET STATISTICS
============================================================
Total documents: 11,314
Total categories: 20

Category Distribution:
------------------------------------------------------------
  alt.atheism                      480 ( 4.24%)
  comp.graphics                    584 ( 5.16%)
  comp.os.ms-windows.misc          591 ( 5.22%)
  comp.sys.ibm.pc.hardware         590 ( 5.21%)
  comp.sys.mac.hardware            578 ( 5.11%)
  comp.windows.x                   593 ( 5.24%)
  misc.forsale                     585 ( 5.17%)
  rec.autos                        594 ( 5.25%)
  rec.motorcycles                  598 ( 5.29%)
  rec.sport.baseball               597 ( 5.28%)
  rec.sport.hockey                 600 ( 5.30%)
  sci.crypt                        595 ( 5.26%)
  sci.electronics                  591 ( 5.22%)
  sci.med                          594 ( 5.25%)
  sci.space                        593 ( 5.24%)
  soc.religion.christian           599 ( 5.29%)
  talk.politics.guns               546 ( 4.83%)
  talk.politics.mideast            564 ( 4.99%)
  talk.politics.misc               465 ( 4.11%)
  talk.religion.misc               377 ( 3.33%)
============================================================
```

### Cleaning Impact:
- **Size reduction**: ~30-40% on average
- **Noise removed**: Headers, quotes, signatures, URLs, emails
- **Result**: Clean, semantic-rich text ready for embedding

## Integration

The data loader is integrated into the main pipeline:

```
Data Loader → Text Cleaner → Embedding Model → Clustering → Cache
```

The cleaned documents are passed to the embedding model for vector generation.

## Configuration

Edit `config.py` to customize:
```python
# Data Configuration
NEWSGROUPS_SUBSET = "train"
NEWSGROUPS_CATEGORIES = None  # None = all categories

# Text Cleaning Configuration
LOWERCASE = False
REMOVE_NUMBERS = False
```

## Error Handling

- **Encoding errors**: Handled with `errors='ignore'`
- **Missing files**: Skipped gracefully
- **Empty documents**: Filtered out after cleaning
- **Invalid directories**: Ignored during iteration

## Performance

- **Loading time**: ~2-5 seconds for full dataset
- **Cleaning overhead**: ~10-15% additional time
- **Memory usage**: ~50-100 MB for full dataset

## Next Steps

After data ingestion:
1. ✅ Documents are loaded and cleaned
2. ⏭️ Generate embeddings with `embedding_model.py`
3. ⏭️ Perform clustering with `clustering.py`
4. ⏭️ Build semantic cache with `semantic_cache.py`
