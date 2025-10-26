from agno.knowledge.document.base import Document
from agno.knowledge.chunking.semantic import SemanticChunking
from typing import List

class CustomChunking(SemanticChunking):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def chunk(self, document: Document) -> List[Document]:
        chunked_documents = super().chunk(document)
        
        chunked_documents_without_references = []
        for chunk in chunked_documents:
            if "References" in chunk.content:
                break
            chunked_documents_without_references.append(chunk)
            
        return chunked_documents_without_references
    
