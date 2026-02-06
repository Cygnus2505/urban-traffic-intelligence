"""
Document Chunking Module
Splits long documents into smaller, overlapping chunks for embedding.
"""
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from ..config import get_settings

settings = get_settings()


class DocumentChunker:
    """
    Splits documents into smaller chunks ensuring context is preserved
    using chunk overlap.
    """
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split a list of document dicts into smaller chunk dicts.
        
        Args:
            documents: List of dicts with 'content', 'id', and 'metadata' keys
            
        Returns:
            List of chunk dicts ready for embedding
        """
        chunks = []
        
        for doc in documents:
            # Create LangChain document
            lc_doc = LangchainDocument(
                page_content=doc['content'],
                metadata=doc.get('metadata', {})
            )
            
            # Split
            lc_chunks = self.splitter.split_documents([lc_doc])
            
            # Convert back to dicts
            for i, chunk in enumerate(lc_chunks):
                chunks.append({
                    'chunk_id': f"{doc['id']}_{i}",
                    'parent_id': doc['id'],
                    'content': chunk.page_content,
                    'chunk_index': i,
                    'metadata': chunk.metadata
                })
                
        return chunks
