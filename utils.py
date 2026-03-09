import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import chardet
import logging
import json
import os

logger = logging.getLogger(__name__)

# --- Embedding Model ---

def get_embeddings():
    """Return OpenAI Embeddings instance."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

# --- Qdrant Manager ---

COLLECTION_NAME = "juangpt"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small outputs 1536-dim vectors

class QdrantManager:
    """Manages connections and operations with Qdrant Cloud."""

    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.embeddings = get_embeddings()
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")

    def get_vector_store(self):
        """Return a LangChain QdrantVectorStore wrapping this collection."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )

    def reindex(self, documents: list[Document]):
        """Wipe the collection and re-add all documents."""
        import time

        # Recreate the collection to clear old data
        self.client.delete_collection(COLLECTION_NAME)
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Cleared collection {COLLECTION_NAME}")

        # Add documents in small batches with delays to avoid rate limits
        vector_store = self.get_vector_store()
        batch_size = 50
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1

            # Retry with exponential backoff on rate limit errors
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    vector_store.add_documents(batch)
                    logger.info(f"Indexed batch {batch_num}/{total_batches} ({len(batch)} docs)")
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate_limit" in str(e).lower():
                        wait_time = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32 seconds
                        logger.warning(f"Rate limited on batch {batch_num}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Error indexing batch {batch_num}: {str(e)}")
                        raise
            else:
                logger.error(f"Failed to index batch {batch_num} after {max_retries} retries")

            # Pause between batches to avoid hitting rate limits
            if i + batch_size < len(documents):
                time.sleep(0.5)

        logger.info(f"Reindexing complete. Total documents: {len(documents)}")


# --- CSV Handler (robust version from chatGroq.py) ---

class CSVHandler:
    def __init__(self):
        self.supported_encodings = ['utf-8', 'latin1', 'utf-16', 'ascii', 'iso-8859-1']
        self.possible_delimiters = [';', ',', '\t', '|']

    def detect_encoding(self, file_path: str) -> str:
        """Detect the file encoding."""
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding']

    def detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detect the CSV delimiter."""
        with open(file_path, 'r', encoding=encoding) as file:
            header = file.readline()
            for delimiter in self.possible_delimiters:
                if delimiter in header:
                    return delimiter
        return ','

    def read_file_with_description(self, file_path: str, encoding: str, delimiter: str):
        """Read CSV expecting first line=description, second line=link, rest=data."""
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                if len(lines) < 3:
                    raise ValueError("File must have at least 3 lines")
                description = lines[0].strip().replace('"', '')
                link = lines[1].strip().replace('"', '')

            chunks = []
            for chunk in pd.read_csv(
                file_path, encoding=encoding, sep=delimiter,
                skiprows=2, chunksize=10000
            ):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            return description, link, df
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names."""
        df.columns = df.columns.map(
            lambda x: str(x).strip().lower()
            .replace('"', '').replace("'", "")
            .replace(" ", "_").replace("-", "_")
        )
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data in all columns based on their data types."""
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(lambda x: self._clean_string(x))
            try:
                numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                if numeric_conversion.notna().all():
                    df[col] = numeric_conversion
            except Exception:
                pass

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _clean_string(self, value) -> str:
        """Clean individual string values."""
        if pd.isna(value):
            return pd.NA
        try:
            value = str(value).strip().replace('"', '')
            value = value.replace('\n', ' ').replace('\r', ' ')
            value = ' '.join(value.split())
            return value if value else pd.NA
        except Exception:
            return pd.NA

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type."""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else 'Unknown')

        return df


# --- Process CSV Files into LangChain Documents ---

def process_csv_files(file_paths: list[str]) -> list[Document]:
    """Process multiple CSV files into LangChain Document objects."""
    csv_handler = CSVHandler()
    documents = []
    metadata_dir = "metadata"
    os.makedirs(metadata_dir, exist_ok=True)

    for file_path in file_paths:
        try:
            encoding = csv_handler.detect_encoding(file_path)
            delimiter = csv_handler.detect_delimiter(file_path, encoding)
            logger.info(f"Processing {file_path} (encoding={encoding}, delimiter={delimiter})")

            description, link, df = csv_handler.read_file_with_description(
                file_path, encoding, delimiter
            )
            df = csv_handler.clean_column_names(df)
            df = csv_handler.clean_data(df)
            df = csv_handler.handle_missing_values(df)

            # Save metadata JSON
            detailed_metadata = {
                "source": file_path,
                "description": description,
                "encoding": encoding,
                "delimiter": delimiter,
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "column_names": list(df.columns),
                "link": link,
            }
            metadata_filename = os.path.join(
                metadata_dir, f"{Path(file_path).stem}_metadata.json"
            )
            with open(metadata_filename, "w") as f:
                json.dump(detailed_metadata, f, indent=4)

            # Create LangChain Documents by splitting large dataframes into chunks
            chunk_size_chars = 3000
            df_string = df.to_string(index=False)
            lines = df_string.split('\n')
            
            if not lines:
                continue

            header_line = lines[0]
            current_chunk_text = f"File Description: {description}\nSource Link: {link}\n\nData:\n{header_line}\n"
            
            docs_from_file = []
            
            for line in lines[1:]:
                # If adding this line exceeds chunk size and we already have data lines in the chunk
                if len(current_chunk_text) + len(line) > chunk_size_chars and current_chunk_text != f"File Description: {description}\nSource Link: {link}\n\nData:\n{header_line}\n":
                    docs_from_file.append(
                        Document(
                            page_content=current_chunk_text,
                            metadata={
                                "source": file_path,
                                "description": description[:200],
                                "link": link,
                            }
                        )
                    )
                    current_chunk_text = f"File Description: {description}\nSource Link: {link}\n\nData:\n{header_line}\n"
                
                current_chunk_text += line + "\n"
            
            # Add the last chunk if it has any data rows
            if current_chunk_text != f"File Description: {description}\nSource Link: {link}\n\nData:\n{header_line}\n":
                docs_from_file.append(
                    Document(
                        page_content=current_chunk_text,
                        metadata={
                            "source": file_path,
                            "description": description[:200],
                            "link": link,
                        }
                    )
                )

            documents.extend(docs_from_file)
            logger.info(f"Successfully processed {file_path} into {len(docs_from_file)} chunks")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue

    return documents