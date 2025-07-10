# Embeddings Retrieval Tool - Usage Guide

## Overview
This tool provides a unified interface to retrieve embeddings from your pgVector PostgreSQL database with various filtering and search options.

## Installation
```bash
# Install required dependencies
pip install psycopg2-binary numpy sentence-transformers

# Optional: For better performance with large datasets
pip install pandas tabulate
```

## Basic Usage

### 0. Using Configuration Files
```bash
# Use a configuration file (recommended for Docker environments)
python retrieve_embeddings.py -c conf_umrs_user1.json --table text --limit 10

# Use configuration file from Docker workdir
python retrieve_embeddings.py -c /workdir/conf_umrs_user1.json --list-tables
```

### 1. List Available Databases
```bash
python retrieve_embeddings.py --list-databases
```

### 2. List Tables with Vector Columns
```bash
# List tables for a specific user's database
python retrieve_embeddings.py --user ejeannot --list-tables

# List tables for a specific database
python retrieve_embeddings.py --list-tables --database "text_and_images_ejeannot"
```

### 3. Retrieve All Embeddings
```bash
# Get all embeddings for a user (limited to 100 by default)
python retrieve_embeddings.py --user ejeannot --limit 100

# Get embeddings from a specific table
python retrieve_embeddings.py --user ejeannot --table text --limit 50
```

### 4. Filter by Date Range
```bash
python retrieve_embeddings.py --user ejeannot --table text \
    --date-from 2025-01-01 --date-to 2025-07-10
```

### 5. Filter by S3 Object Name
```bash
# Find all embeddings for cat images
python retrieve_embeddings.py --user ejeannot --table image \
    --s3-filter "Cat_" --limit 20

# Find embeddings for specific file types
python retrieve_embeddings.py --user ejeannot --s3-filter ".jpg"
```

### 6. Similarity Search
```bash
# Find similar text embeddings
python retrieve_embeddings.py --user ejeannot --table text \
    --similarity "apple fruit delicious" --top-k 5

# Find similar image embeddings
python retrieve_embeddings.py --user ejeannot --table image \
    --similarity "cat animal pet" --top-k 10
```

### 7. Export Results
```bash
# Export to JSON
python retrieve_embeddings.py --user ejeannot --table text \
    --output-format json --save-to embeddings.json

# Export to CSV
python retrieve_embeddings.py --user ejeannot --table image \
    --output-format csv --save-to embeddings.csv
```

## Advanced Examples

### Multiple Filters Combined
```bash
python retrieve_embeddings.py --user ejeannot --table text \
    --date-from 2025-06-01 \
    --s3-filter "apple" \
    --limit 20 \
    --output-format json
```

### Different Similarity Models
```bash
# Use a different model for similarity search
python retrieve_embeddings.py --user ejeannot --table text \
    --similarity "technology computer" \
    --similarity-model "all-mpnet-base-v2" \
    --top-k 5
```

### Work with Different Experiments
```bash
# Retrieve from a specific experiment
python retrieve_embeddings.py --user ejeannot \
    --experiment "large_embeddings_db" \
    --table text_embeddings \
    --limit 10
```

## Database Schema Expected

The tool expects tables with the following structure:
- `id`: Primary key (SERIAL)
- `s3_object`: S3 object path (TEXT)
- `embedding`: Vector column (VECTOR(dimension))
- `date`: Date field (DATE) - optional
- `time`: Time field (TIME) - optional
- `text`: Text content (TEXT) - optional for text tables
- `model`: Model name (TEXT) - optional

## Common Use Cases

### 1. Data Exploration
```bash
# Get overview of what's in the database
python retrieve_embeddings.py --user ejeannot --list-tables
python retrieve_embeddings.py --user ejeannot --table text --limit 5
```

### 2. Quality Check
```bash
# Check recent embeddings
python retrieve_embeddings.py --user ejeannot --table text \
    --date-from 2025-07-01 --limit 10
```

### 3. Content Search
```bash
# Find specific content
python retrieve_embeddings.py --user ejeannot --table image \
    --s3-filter "Dog_" --limit 15
```

### 4. Similarity Analysis
```bash
# Find similar items
python retrieve_embeddings.py --user ejeannot --table text \
    --similarity "artificial intelligence machine learning" \
    --top-k 8
```

### 5. Data Export for Analysis
```bash
# Export for external analysis
python retrieve_embeddings.py --user ejeannot --table text \
    --output-format json --save-to analysis_data.json
```

## Troubleshooting

### Connection Issues
- Ensure PostgreSQL is running
- Check if the user has access to the database
- Verify pgvector extension is installed

### No Results Found
- Check if the table exists: `--list-tables`
- Verify the user and experiment names are correct
- Check date filters are not too restrictive

### Similarity Search Not Working
- Install sentence-transformers: `pip install sentence-transformers`
- Verify the model name is correct
- Check if embeddings exist in the table

## Performance Tips

1. **Use limits**: Always use `--limit` for large datasets
2. **Specific tables**: Specify `--table` to avoid scanning all tables
3. **Date filters**: Use date ranges to reduce search space
4. **Indexing**: Ensure your vector columns have appropriate indexes for similarity search

## Configuration

Edit `embeddings_config.json` to customize:
- Default database connection parameters
- Available models for similarity search
- Output formatting options
- Common experiment and table names
