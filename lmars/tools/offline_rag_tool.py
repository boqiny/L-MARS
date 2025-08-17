"""
Offline RAG Tool for Local Legal Documents
Uses BM25 algorithm for better retrieval performance
"""
import os
import re
import json
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter
import hashlib


class BM25:
    """BM25 ranking algorithm implementation for document retrieval."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with tuning parameters.
        
        Args:
            k1: Controls term frequency saturation (typically 1.2-2.0)
            b: Controls length normalization (0-1, typically 0.75)
        """
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avgdl = 0
        self.doc_freqs = {}
        self.idf = {}
        self.doc_lengths = []
        self.documents = []
        
    def fit(self, documents: List[List[str]]):
        """
        Fit BM25 to a corpus of tokenized documents.
        
        Args:
            documents: List of tokenized documents (list of word lists)
        """
        self.documents = documents
        self.doc_count = len(documents)
        self.doc_lengths = [len(doc) for doc in documents]
        self.avgdl = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0
        
        # Calculate document frequencies
        df = {}
        for document in documents:
            seen_words = set()
            for word in document:
                if word not in seen_words:
                    df[word] = df.get(word, 0) + 1
                    seen_words.add(word)
        
        # Calculate IDF scores
        self.idf = {}
        for word, freq in df.items():
            self.idf[word] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
    
    def score(self, query: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a document given a query.
        
        Args:
            query: Tokenized query
            doc_idx: Index of document to score
            
        Returns:
            BM25 score
        """
        if doc_idx >= len(self.documents):
            return 0.0
            
        doc = self.documents[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        # Count term frequencies in document
        doc_freqs = Counter(doc)
        
        score = 0.0
        for term in query:
            if term in doc_freqs:
                tf = doc_freqs[term]
                idf = self.idf.get(term, 0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for top-k documents given a query.
        
        Args:
            query: Tokenized query
            top_k: Number of results to return
            
        Returns:
            List of (doc_idx, score) tuples
        """
        scores = [(i, self.score(query, i)) for i in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class DocumentChunk:
    """Represents a chunk of a document for granular retrieval."""
    
    def __init__(self, content: str, metadata: Dict[str, Any], chunk_id: str):
        self.content = content
        self.metadata = metadata
        self.chunk_id = chunk_id
        self.tokens = self._tokenize(content.lower())
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on non-alphanumeric characters."""
        # Remove special characters and split
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove stop words (basic list)
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this',
            'it', 'from', 'be', 'are', 'was', 'were', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'shall', 'must', 'if', 'when', 'where',
            'what', 'who', 'whom', 'whose', 'why', 'how'
        }
        return [t for t in tokens if t and len(t) > 2 and t not in stop_words]


class OfflineRAGTool:
    """Offline RAG search using BM25 for local markdown documents."""
    
    def __init__(self, input_dir: str = "inputs", chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the offline RAG tool.
        
        Args:
            input_dir: Directory containing markdown documents
            chunk_size: Size of document chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.input_dir = Path(input_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.bm25 = None
        
        # Load and index documents
        self._load_documents()
        if self.chunks:
            self._build_index()
    
    def _load_documents(self):
        """Load all markdown documents and create chunks."""
        if not self.input_dir.exists():
            print(f"Creating input directory: {self.input_dir}")
            self.input_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Load all .md files
        md_files = list(self.input_dir.glob("**/*.md"))
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract title
                title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                title = title_match.group(1) if title_match else file_path.stem
                
                # Create document metadata
                doc_metadata = {
                    'file_path': str(file_path),
                    'title': title,
                    'filename': file_path.name
                }
                
                # Create chunks with overlap
                chunks = self._create_chunks(content, doc_metadata)
                self.chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Loaded {len(self.chunks)} chunks from {len(md_files)} documents")
    
    def _create_chunks(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split document into overlapping chunks.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Split by sections (headers) first for better semantic chunking
        sections = re.split(r'\n(?=#)', content)
        
        for section in sections:
            if not section.strip():
                continue
                
            # If section is small enough, use it as a single chunk
            if len(section) <= self.chunk_size:
                chunk_id = hashlib.md5(section.encode()).hexdigest()[:8]
                chunk = DocumentChunk(section, metadata, chunk_id)
                chunks.append(chunk)
            else:
                # Split large sections into smaller chunks with overlap
                text = section
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    
                    # Try to find a good break point (sentence or paragraph)
                    if end < len(text):
                        # Look for paragraph break
                        break_point = text.rfind('\n\n', start, end)
                        if break_point == -1:
                            # Look for sentence break
                            break_point = text.rfind('. ', start, end)
                        if break_point != -1:
                            end = break_point + 1
                    
                    chunk_text = text[start:end].strip()
                    if chunk_text:
                        chunk_id = hashlib.md5(f"{metadata['file_path']}_{start}".encode()).hexdigest()[:8]
                        chunk = DocumentChunk(chunk_text, metadata, chunk_id)
                        chunks.append(chunk)
                    
                    # Move start position with overlap
                    start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    
    def _build_index(self):
        """Build BM25 index for all chunks."""
        # Extract tokens from all chunks
        tokenized_chunks = [chunk.tokens for chunk in self.chunks]
        
        # Build BM25 index
        self.bm25 = BM25(k1=1.5, b=0.75)
        self.bm25.fit(tokenized_chunks)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks using BM25.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with content and metadata
        """
        if not self.chunks or not self.bm25:
            return []
        
        # Tokenize query
        query_tokens = DocumentChunk(query, {}, "query").tokens
        
        if not query_tokens:
            return []
        
        # Search with BM25
        results = self.bm25.search(query_tokens, top_k)
        
        # Format results
        formatted_results = []
        seen_files = set()
        
        for chunk_idx, score in results:
            if score > 0:
                chunk = self.chunks[chunk_idx]
                
                # Deduplicate by file (keep best scoring chunk per file)
                file_path = chunk.metadata['file_path']
                if file_path in seen_files:
                    continue
                seen_files.add(file_path)
                
                formatted_results.append({
                    'title': chunk.metadata['title'],
                    'content': chunk.content[:500] + '...' if len(chunk.content) > 500 else chunk.content,
                    'full_content': chunk.content,
                    'file_path': file_path,
                    'relevance_score': score,
                    'source': 'Offline RAG (BM25)',
                    'chunk_id': chunk.chunk_id
                })
        
        return formatted_results
    
    def add_document(self, content: str, filename: str):
        """
        Add a new document to the RAG system.
        
        Args:
            content: Markdown content
            filename: Name for the file
        """
        # Ensure input directory exists
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # Save document
        file_path = self.input_dir / filename
        if not filename.endswith('.md'):
            file_path = self.input_dir / f"{filename}.md"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Reload documents and rebuild index
        self.chunks = []  # Clear existing chunks
        self._load_documents()
        if self.chunks:
            self._build_index()
    
    def list_documents(self) -> List[Dict[str, str]]:
        """List all available documents."""
        docs = {}
        for chunk in self.chunks:
            file_path = chunk.metadata['file_path']
            if file_path not in docs:
                docs[file_path] = {
                    'title': chunk.metadata['title'],
                    'filename': chunk.metadata['filename'],
                    'path': file_path
                }
        return list(docs.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents."""
        if not self.chunks:
            return {'status': 'No documents indexed'}
        
        unique_docs = len(set(c.metadata['file_path'] for c in self.chunks))
        total_tokens = sum(len(c.tokens) for c in self.chunks)
        avg_chunk_size = sum(len(c.content) for c in self.chunks) / len(self.chunks)
        
        return {
            'documents': unique_docs,
            'chunks': len(self.chunks),
            'total_tokens': total_tokens,
            'avg_chunk_size': f"{avg_chunk_size:.0f} chars",
            'algorithm': 'BM25',
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }


# Singleton instance
_rag_tool = None

def get_offline_rag_tool(input_dir: str = "inputs") -> OfflineRAGTool:
    """Get or create the offline RAG tool instance."""
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = OfflineRAGTool(input_dir)
    return _rag_tool


def search_offline_rag(query: str, top_k: int = 3) -> str:
    """
    Search offline RAG for relevant legal documents using BM25.
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        Formatted search results as string
    """
    tool = get_offline_rag_tool()
    results = tool.search(query, top_k)
    
    if not results:
        return "No relevant documents found in offline RAG."
    
    # Format results
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"**Result {i}: {result['title']}**")
        output.append(f"Source: {result['file_path']}")
        output.append(f"Relevance: {result['relevance_score']:.2f}")
        output.append(f"Content: {result['content']}")
        output.append("")
    
    return "\n".join(output)


def add_document_to_rag(content: str, filename: str) -> str:
    """
    Add a document to the offline RAG system.
    
    Args:
        content: Document content in markdown
        filename: Filename for the document
        
    Returns:
        Success message
    """
    tool = get_offline_rag_tool()
    tool.add_document(content, filename)
    return f"Document '{filename}' added successfully to offline RAG."