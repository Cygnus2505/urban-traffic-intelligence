"""
Text Embedding Module
Converts text chunks into vector embeddings using OpenAI models.
"""
from typing import List
from openai import OpenAI
from loguru import logger
import time

from ..config import get_settings

settings = get_settings()


class OpenAIEmbedder:
    """Generates embeddings using OpenAI API"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embedding vectors (list of floats)
        """
        if not texts:
            return []
            
        try:
            # Clean texts (replace newlines) to improve performance
            cleaned_texts = [text.replace("\n", " ") for text in texts]
            
            response = self.client.embeddings.create(
                input=cleaned_texts,
                model=self.model
            )
            
            # Extract embeddings in order
            return [data.embedding for data in response.data]
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
