import pandas as pd
from pathlib import Path
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
import logging

logger = logging.getLogger(__name__)

class CSVHandler:
    def __init__(self):
        pass
        
    def read_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {str(e)}")
            raise

def process_csv_files(file_paths):
    documents = []
    csv_handler = CSVHandler()
    
    for file_path in file_paths:
        try:
            df = csv_handler.read_csv(file_path)
            # Convert DataFrame to string representation
            text_content = df.to_string()
            # Create document with metadata
            doc = Document(text=text_content, metadata={"source": str(file_path)})
            documents.append(doc)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            continue
            
    return documents

def create_or_load_index(data_dir, persist_dir):
    if Path(persist_dir).exists():
        # Load existing index
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        # Create new index
        documents = []
        csv_files = Path(data_dir).glob("*.csv")
        documents = process_csv_files([str(f) for f in csv_files])
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
    
    return index

def update_index(data_dir, persist_dir):
    # Process all CSV files and create new index
    documents = []
    csv_files = Path(data_dir).glob("*.csv")
    documents = process_csv_files([str(f) for f in csv_files])
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    return index