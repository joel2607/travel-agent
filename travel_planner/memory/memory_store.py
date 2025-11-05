from typing import List, Dict, Any, Optional
import json
import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import settings
from models.memory import ConversationMessage


class MemoryStore:
    """Manages recall and archival storage"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.core_memory_file = os.path.join(settings.CHROMA_PERSIST_DIR, "core_memory.json")
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GEMINI_API_KEY
        )
        
        # Initialize Chroma vector store
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create collections
        self.conversation_collection = self.client.get_or_create_collection(
            name=f"conversations_{user_id}",
            metadata={"type": "recall_storage"}
        )
        
        self.archival_collection = self.client.get_or_create_collection(
            name=f"archival_{user_id}",
            metadata={"type": "archival_storage"}
        )
        
        # Core memory (simple dict storage - use DB in production)
        self.core_memory_store = self._load_core_memory_from_file()
    
    def _load_core_memory_from_file(self) -> Dict:
        """Load core memory from JSON file"""
        if os.path.exists(self.core_memory_file):
            try:
                with open(self.core_memory_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def get_core_memory(self) -> Optional[Dict]:
        """Retrieve core memory"""
        return self.core_memory_store.get(self.user_id)
    
    def save_core_memory(self, core_memory: Dict):
        """Save core memory"""
        self.core_memory_store[self.user_id] = core_memory
        with open(self.core_memory_file, "w") as f:
            json.dump(self.core_memory_store, f, indent=4)
    
    def save_conversation_message(self, message: ConversationMessage):
        """Save message to recall storage"""
        embedding = self.embeddings.embed_query(message.content)
        
        self.conversation_collection.add(
            documents=[message.content],
            embeddings=[embedding],
            metadatas=[{
                "role": message.role,
                "timestamp": message.timestamp,
                **message.metadata
            }],
            ids=[f"{self.user_id}_{message.timestamp}"]
        )
    
    def search_conversations(
        self, 
        query: str, 
        page: int = 1, 
        page_size: int = 10
    ) -> List[Dict]:
        """Search conversation history"""
        query_embedding = self.embeddings.embed_query(query)
        
        offset = (page - 1) * page_size
        
        results = self.conversation_collection.query(
            query_embeddings=[query_embedding],
            n_results=page_size,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results['documents']:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                formatted_results.append({
                    "content": doc,
                    "metadata": metadata,
                    "relevance_score": 1 - distance  # Convert distance to similarity
                })
        
        return formatted_results
    
    def insert_archival(self, content: str, metadata: Dict):
        """Insert into archival storage"""
        embedding = self.embeddings.embed_query(content)
        
        doc_id = f"{self.user_id}_{metadata.get('trip_id', 'doc')}_{len(self.archival_collection.get()['ids'])}"
        
        self.archival_collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
    
    def get_all_archival_memories(self) -> List[Dict]:
        """Retrieve all documents from archival storage."""
        results = self.archival_collection.get(include=["documents", "metadatas"])
        
        formatted_results = []
        if results['documents']:
            for doc, metadata in zip(results['documents'], results['metadatas']):
                formatted_results.append({"content": doc, "metadata": metadata})
        
        return formatted_results
        
    def search_archival(
        self,
        query: str,
        page: int = 1,
        page_size: int = 5
    ) -> List[Dict]:
        """Search archival storage (past trips)"""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.archival_collection.query(
            query_embeddings=[query_embedding],
            n_results=page_size,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results['documents']:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                formatted_results.append({
                    "content": doc,
                    "metadata": metadata,
                    "relevance_score": 1 - distance
                })
        
        return formatted_results