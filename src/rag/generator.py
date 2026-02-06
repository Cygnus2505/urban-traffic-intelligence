"""
RAG Generator Module
Generates answers using LLM based on retrieved context.
"""
from typing import List, Dict, Any
from openai import OpenAI
from loguru import logger

from ..config import get_settings

settings = get_settings()


class RAGGenerator:
    """Generates answers using OpenAI Chat Completion"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer given a query and retrieved context.
        
        Args:
            query: User's question
            context_chunks: List of relevant text chunks found in database
            
        Returns:
            Dict containing 'answer' and 'sources'
        """
        
        # 1. Construct the context string
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information in the traffic database to answer your question.",
                "sources": []
            }
            
        context_text = "\n\n---\n\n".join([c['content'] for c in context_chunks])
        
        # 2. Build the system prompt
        system_prompt = """You are the Urban Traffic Intelligence AI for Chicago. 
        Your goal is to answer questions about traffic conditions, crashes, and road construction based ONLY on the provided context.
        
        Rules:
        1. Use the Context provided below to answer.
        2. If the answer is not in the context, say you don't know. Do not make up information.
        3. Mention specific street names, times, or conditions if available.
        4. Keep answers concise and professional.
        """
        
        user_prompt = f"""
        Context Information:
        {context_text}
        
        Question: {query}
        """
        
        try:
            # 3. Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3, # Low temperature for factual consistency
            )
            
            answer = response.choices[0].message.content
            
            # 4. Format sources for citation
            sources = [
                {
                    "content_snippet": c['content'][:100] + "...", 
                    "score": c['score'],
                    "metadata": c['metadata']
                } 
                for c in context_chunks
            ]
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "Sorry, I encountered an error while processing your request.",
                "sources": []
            }
