"""
Simple Embedding System for Turkcell SSS Data

This system only does:
1. Load CSV data
2. Preprocess (combine question+answer+source)
3. Create embeddings using sentence-transformers
That's it. Clean and simple.
"""

import csv
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class EmbeddingSystem:
    """
    Simple embedding system - just creates embeddings from CSV data.
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        """
        Initialize the embedding system.
        
        Args:
            model_name: Hugging Face model name for embeddings
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Embedding system initialized with model: {model_name}")
    
    def _load_model(self):
        """Load the embedding model if not already loaded"""
        if self.model is None:
            logger.info("Loading embedding model...")
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
    
    def load_csv_data(self, csv_file_path: str) -> List[Dict[str, str]]:
        """
        Load CSV data with questions, answers, and sources.
        
        Args:
            csv_file_path: Path to the CSV file
            
        Returns:
            List of dictionaries with question, answer, source
        """
        data = []
        
        try:
            # Try utf-8-sig first to handle BOM
            with open(csv_file_path, 'r', encoding='utf-8-sig') as file:
                reader = csv.DictReader(file)
                
                # Debug: print column names
                logger.info(f"CSV columns found: {reader.fieldnames}")
                
                for i, row in enumerate(reader):
                    # Handle different possible column names
                    question = (row.get('Question', '') or 
                              row.get('question', '') or 
                              row.get('QUESTION', '')).strip()
                    
                    answer = (row.get('Answer', '') or 
                             row.get('answer', '') or 
                             row.get('ANSWER', '')).strip()
                    
                    source = (row.get('Source', '') or 
                             row.get('source', '') or 
                             row.get('SOURCE', '')).strip()
                    
                    # More lenient check - just need question
                    if question and len(question) > 3:
                        data.append({
                            'question': question,
                            'answer': answer if answer else "No answer provided",
                            'source': source if source else "No source provided"
                        })
                    else:
                        if i < 5:  # Log first few problematic rows
                            logger.warning(f"Skipping row {i}: question='{question[:50]}', answer='{answer[:50]}'")
                
                logger.info(f"Loaded {len(data)} records from CSV")
                return data
                
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            # Try with different encoding
            try:
                logger.info("Trying with latin-1 encoding...")
                with open(csv_file_path, 'r', encoding='latin-1') as file:
                    reader = csv.DictReader(file)
                    
                    for row in reader:
                        question = row.get('Question', '').strip()
                        answer = row.get('Answer', '').strip()
                        source = row.get('Source', '').strip()
                        
                        if question and len(question) > 3:
                            data.append({
                                'question': question,
                                'answer': answer if answer else "No answer provided",
                                'source': source if source else "No source provided"
                            })
                    
                    logger.info(f"Loaded {len(data)} records with latin-1 encoding")
                    return data
                    
            except Exception as e2:
                logger.error(f"Error with latin-1 encoding: {e2}")
                return []
    
    def combine_text(self, question: str, answer: str, source: str) -> str:
        """
        Combine question, answer, and source into a single text string.
        Truncates if too long for the model (512 tokens ≈ 2000 characters).
        
        Args:
            question: The question text
            answer: The answer text
            source: The source URL
            
        Returns:
            Combined text string (truncated if needed)
        """
        # Start with question (most important)
        combined = f"Soru: {question}\n\n"
        
        # Estimate remaining space (rough: 512 tokens ≈ 2000 chars)
        question_source_length = len(combined) + len(f"\n\nKaynak: {source}")
        max_answer_length = 2000 - question_source_length
        
        if len(answer) > max_answer_length:
            answer = answer[:max_answer_length] + "..."
            logger.debug("Truncated answer to fit model limits")
        
        combined += f"Cevap: {answer}\n\nKaynak: {source}"
        
        return combined
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text using sentence-transformers.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array embedding (1024,)
        """
        self._load_model()
        
        try:
            # Use sentence-transformers which handles E5 models properly
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding  # Already a numpy array with shape (1024,)
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    def process_csv_to_embeddings(self, csv_file_path: str) -> List[Dict[str, Any]]:
        """
        Process CSV file and create embeddings for all records.
        
        Args:
            csv_file_path: Path to the CSV file
            
        Returns:
            List of dicts with embeddings and metadata
        """
        try:
            # Load CSV data
            data = self.load_csv_data(csv_file_path)
            
            if not data:
                logger.error("No data loaded from CSV")
                return []
            
            results = []
            logger.info("Creating embeddings...")
            
            for i, item in enumerate(data):
                # Combine text
                combined_text = self.combine_text(
                    item['question'], 
                    item['answer'], 
                    item['source']
                )
                
                # Create embedding
                embedding = self.create_embedding(combined_text)
                
                # Store result
                results.append({
                    'id': str(i),
                    'embedding': embedding,
                    'text': combined_text,
                    'metadata': {
                        'question': item['question'],
                        'answer': item['answer'],
                        'source': item['source'],
                        'index': i
                    }
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(data)} embeddings")
            
            logger.info(f"Successfully created {len(results)} embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return []


# Global instance for easy usage
embedding_system = EmbeddingSystem()


def create_embeddings_from_csv(csv_file_path: str) -> List[Dict[str, Any]]:
    """
    Simple function to create embeddings from CSV file.
    
    Args:
        csv_file_path: Path to CSV file
        
    Returns:
        List of embeddings with metadata
    """
    return embedding_system.process_csv_to_embeddings(csv_file_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python embedding_system.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    print(f"Creating embeddings from {csv_file}...")
    
    results = create_embeddings_from_csv(csv_file)
    
    if results:
        print(f"Successfully created {len(results)} embeddings")
        print(f"Sample embedding shape: {results[0]['embedding'].shape}")
        print(f"Sample question: {results[0]['metadata']['question']}")
    else:
        print("Failed to create embeddings")