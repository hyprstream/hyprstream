# Application Developer Persona: File System RAG with Adaptive Models
## Building Retrieval-Augmented Generation Apps with Sparse Adaptive Layers

### Overview

As an application developer, you want to build a RAG system that indexes documents from your file system, generates embeddings, and uses a frozen base model with sparse adaptive layers to provide contextual responses. Hyprstream enables you to build this with minimal infrastructure while maintaining per-domain or per-user model adaptations.

---

## The Challenge

Traditional RAG pipelines:
```
Files → Embedding API → Vector DB → LLM API → Response
         ($$$)           (Separate)    ($$$)
```

**Problems:**
- Expensive API calls for embeddings and inference
- No ability to adapt model to your specific documents
- Separate systems for storage, retrieval, and generation
- No learning from user interactions
- Generic responses that don't improve over time

---

## The Hyprstream Solution

```
Files → Local Embedder → Hyprstream → Adapted Response
                           ↓
                   [VDB Storage + Base Model + Sparse Adapters]
                        (All integrated, learns from usage)
```

---

## Complete Example: Building a Documentation Assistant

### 1. File Indexing Pipeline

```python
from hyprstream import Client, EmbeddingStore, SparseAdaptiveModel
from pathlib import Path
import hashlib
from typing import List, Dict
import mimetypes

class FileIndexer:
    """Index files and generate embeddings for RAG"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.client = Client("http://localhost:8080")
        
        # Local embedding model (no API costs)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims
        
        # Document chunker
        self.chunker = DocumentChunker(
            chunk_size=512,
            overlap=50
        )
        
    async def index_directory(self, 
                             path: Path = None,
                             extensions: List[str] = ['.md', '.txt', '.py', '.js']):
        """Recursively index files in directory"""
        
        path = path or self.base_path
        indexed_count = 0
        
        for file_path in path.rglob('*'):
            if file_path.suffix in extensions:
                await self.index_file(file_path)
                indexed_count += 1
                
        print(f"Indexed {indexed_count} files")
        return indexed_count
    
    async def index_file(self, file_path: Path):
        """Index a single file with smart chunking"""
        
        # Check if already indexed
        file_hash = self.get_file_hash(file_path)
        if await self.is_indexed(file_hash):
            return
        
        # Read and chunk file
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        chunks = self.chunker.chunk(content, file_type=file_path.suffix)
        
        # Generate embeddings locally
        embeddings = self.embedder.encode(
            [chunk['text'] for chunk in chunks],
            batch_size=32,
            show_progress_bar=False
        )
        
        # Store in Hyprstream with metadata
        for chunk, embedding in zip(chunks, embeddings):
            await self.client.insert(
                embedding=embedding.tolist(),
                metadata={
                    'file_path': str(file_path),
                    'file_hash': file_hash,
                    'chunk_index': chunk['index'],
                    'chunk_text': chunk['text'],
                    'start_line': chunk.get('start_line'),
                    'end_line': chunk.get('end_line'),
                    'file_type': file_path.suffix,
                    'timestamp': file_path.stat().st_mtime
                },
                source='sentence-transformers'
            )
    
    def get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for deduplication"""
        return hashlib.md5(file_path.read_bytes()).hexdigest()
    
    async def is_indexed(self, file_hash: str) -> bool:
        """Check if file is already indexed"""
        results = await self.client.search_by_metadata(
            filters={'file_hash': file_hash},
            limit=1
        )
        return len(results) > 0


class DocumentChunker:
    """Smart document chunking based on file type"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: str, file_type: str) -> List[Dict]:
        """Chunk document based on type"""
        
        if file_type == '.py':
            return self.chunk_python(content)
        elif file_type == '.md':
            return self.chunk_markdown(content)
        else:
            return self.chunk_text(content)
    
    def chunk_python(self, content: str) -> List[Dict]:
        """Chunk Python code by functions/classes"""
        import ast
        chunks = []
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    # Extract the code
                    lines = content.split('\n')
                    code = '\n'.join(lines[start_line-1:end_line])
                    
                    chunks.append({
                        'text': code,
                        'index': len(chunks),
                        'start_line': start_line,
                        'end_line': end_line,
                        'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                        'name': node.name
                    })
        except:
            # Fallback to text chunking if parsing fails
            return self.chunk_text(content)
        
        return chunks if chunks else self.chunk_text(content)
    
    def chunk_markdown(self, content: str) -> List[Dict]:
        """Chunk Markdown by headers"""
        chunks = []
        current_chunk = []
        current_header = None
        
        for line in content.split('\n'):
            if line.startswith('#'):
                # Save previous chunk
                if current_chunk:
                    chunks.append({
                        'text': '\n'.join(current_chunk),
                        'index': len(chunks),
                        'header': current_header
                    })
                current_chunk = [line]
                current_header = line
            else:
                current_chunk.append(line)
                
                # Check size
                if len('\n'.join(current_chunk)) > self.chunk_size:
                    chunks.append({
                        'text': '\n'.join(current_chunk),
                        'index': len(chunks),
                        'header': current_header
                    })
                    current_chunk = []
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': '\n'.join(current_chunk),
                'index': len(chunks),
                'header': current_header
            })
        
        return chunks
    
    def chunk_text(self, content: str) -> List[Dict]:
        """Simple overlapping text chunks"""
        chunks = []
        lines = content.split('\n')
        
        for i in range(0, len(lines), self.chunk_size - self.overlap):
            chunk_lines = lines[i:i + self.chunk_size]
            chunks.append({
                'text': '\n'.join(chunk_lines),
                'index': len(chunks),
                'start_line': i + 1,
                'end_line': i + len(chunk_lines)
            })
        
        return chunks
```

### 2. RAG with Sparse Adaptive Layers

```python
class AdaptiveRAG:
    """RAG system with per-domain sparse adaptations"""
    
    def __init__(self, base_model_path: str = "gpt2"):
        # Frozen base model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_model.eval()  # Frozen
        
        # Hyprstream client for retrieval
        self.client = Client("http://localhost:8080")
        
        # Sparse adapter storage
        self.adapter_store = VDBWeightStore()
        
        # Active adapters per domain
        self.domain_adapters = {}
        
        # Local embedder (same as indexer)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def generate(self, 
                      query: str,
                      domain: str = "default",
                      k: int = 5,
                      adapt: bool = True) -> str:
        """Generate response with retrieval and adaptation"""
        
        # 1. Embed query
        query_embedding = self.embedder.encode(query)
        
        # 2. Retrieve relevant chunks
        results = await self.client.search(
            embedding=query_embedding.tolist(),
            k=k,
            filters={'domain': domain} if domain else {}
        )
        
        # 3. Build context
        context = self.build_context(results)
        
        # 4. Load/create domain adapter
        if adapt:
            adapter = await self.get_or_create_adapter(domain)
        else:
            adapter = None
        
        # 5. Generate with adapted model
        response = self.generate_with_adapter(
            query=query,
            context=context,
            adapter=adapter
        )
        
        # 6. Learn from interaction (async)
        if adapt:
            asyncio.create_task(
                self.update_adapter(domain, query, context, response)
            )
        
        return response
    
    def build_context(self, results: List[Dict]) -> str:
        """Build context from retrieved chunks"""
        
        context_parts = []
        seen_files = set()
        
        for result in results:
            metadata = result['metadata']
            file_path = metadata['file_path']
            
            # Add file header if first time seeing this file
            if file_path not in seen_files:
                context_parts.append(f"\n--- From {file_path} ---")
                seen_files.add(file_path)
            
            # Add chunk with location info
            if 'start_line' in metadata:
                context_parts.append(
                    f"[Lines {metadata['start_line']}-{metadata['end_line']}]:\n"
                    f"{metadata['chunk_text']}"
                )
            else:
                context_parts.append(metadata['chunk_text'])
        
        return "\n\n".join(context_parts)
    
    async def get_or_create_adapter(self, domain: str) -> SparseAdapter:
        """Get domain-specific adapter or create new one"""
        
        if domain in self.domain_adapters:
            return self.domain_adapters[domain]
        
        # Try to load from VDB
        adapter = await self.adapter_store.load_adapter(f"domain_{domain}")
        
        if adapter is None:
            # Create new sparse adapter for this domain
            adapter = SparseAdapter(
                base_model_config=self.base_model.config,
                sparsity=0.99,  # 99% sparse
                adapter_dim=16   # Low-rank dimension
            )
        
        self.domain_adapters[domain] = adapter
        return adapter
    
    def generate_with_adapter(self,
                             query: str,
                             context: str,
                             adapter: SparseAdapter = None) -> str:
        """Generate response with optional adapter"""
        
        # Build prompt
        prompt = f"""Context from your documents:
{context}

Question: {query}
Answer: """
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        
        # Apply adapter if provided
        if adapter:
            # Temporarily modify model weights with sparse deltas
            with self.apply_sparse_adapter(adapter):
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True
                )
        else:
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        
        return response
    
    async def update_adapter(self,
                           domain: str,
                           query: str,
                           context: str,
                           response: str):
        """Update adapter based on interaction (lightweight training)"""
        
        adapter = self.domain_adapters[domain]
        
        # Simple feedback loop: reinforce if response was good
        # In production, this would use explicit feedback or metrics
        
        # Compute lightweight gradient update
        # This is where sparse training happens
        loss = self.compute_adapter_loss(query, context, response, adapter)
        
        if loss < 0.5:  # Good response, reinforce
            # Update only sparse weights
            sparse_update = self.compute_sparse_gradient(loss, adapter)
            adapter.apply_sparse_update(sparse_update, lr=0.001)
            
            # Store updated adapter
            await self.adapter_store.store_adapter(
                f"domain_{domain}",
                adapter.get_sparse_weights(),
                metadata={'updates': adapter.update_count}
            )


class SparseAdapter:
    """Sparse weight modifications for base model"""
    
    def __init__(self, base_model_config, sparsity=0.99, adapter_dim=16):
        self.config = base_model_config
        self.sparsity = sparsity
        self.adapter_dim = adapter_dim
        
        # Initialize sparse weight deltas
        self.weight_deltas = {}
        
        # Create low-rank adapters for each layer
        for i in range(base_model_config.n_layer):
            # LoRA-style factorization
            self.weight_deltas[f'layer_{i}'] = {
                'A': torch.randn(base_model_config.n_embd, adapter_dim) * 0.01,
                'B': torch.randn(adapter_dim, base_model_config.n_embd) * 0.01
            }
        
        self.update_count = 0
    
    def get_sparse_weights(self) -> Dict:
        """Get sparse representation of adapter weights"""
        sparse_weights = {}
        
        for layer_name, matrices in self.weight_deltas.items():
            # Only store non-zero values
            for matrix_name, matrix in matrices.items():
                mask = torch.abs(matrix) > 1e-6
                indices = torch.nonzero(mask)
                values = matrix[mask]
                
                sparse_weights[f"{layer_name}_{matrix_name}"] = {
                    'indices': indices.tolist(),
                    'values': values.tolist(),
                    'shape': list(matrix.shape)
                }
        
        return sparse_weights
    
    def apply_sparse_update(self, update: Dict, lr: float = 0.001):
        """Apply sparse gradient update"""
        
        for layer_name, grad in update.items():
            if layer_name in self.weight_deltas:
                # Update only top-k values to maintain sparsity
                for matrix_name in ['A', 'B']:
                    if matrix_name in grad:
                        # Keep only top 1% of gradients
                        k = int(grad[matrix_name].numel() * (1 - self.sparsity))
                        top_k = torch.topk(torch.abs(grad[matrix_name].flatten()), k)
                        
                        # Zero out all but top-k
                        mask = torch.zeros_like(grad[matrix_name].flatten())
                        mask[top_k.indices] = 1
                        sparse_grad = grad[matrix_name].flatten() * mask
                        sparse_grad = sparse_grad.reshape(grad[matrix_name].shape)
                        
                        # Apply update
                        self.weight_deltas[layer_name][matrix_name] -= lr * sparse_grad
        
        self.update_count += 1
```

### 3. File Watcher for Auto-Indexing

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio

class FileWatcher(FileSystemEventHandler):
    """Watch filesystem for changes and auto-index"""
    
    def __init__(self, indexer: FileIndexer):
        self.indexer = indexer
        self.queue = asyncio.Queue()
        self.processed = set()
    
    def on_modified(self, event):
        if not event.is_directory:
            self.queue.put_nowait(('modified', event.src_path))
    
    def on_created(self, event):
        if not event.is_directory:
            self.queue.put_nowait(('created', event.src_path))
    
    def on_deleted(self, event):
        if not event.is_directory:
            self.queue.put_nowait(('deleted', event.src_path))
    
    async def process_events(self):
        """Process file events asynchronously"""
        while True:
            event_type, path = await self.queue.get()
            
            if event_type in ['modified', 'created']:
                # Re-index file
                await self.indexer.index_file(Path(path))
                print(f"Re-indexed {path}")
            
            elif event_type == 'deleted':
                # Remove from index
                await self.indexer.remove_file(Path(path))
                print(f"Removed {path} from index")


class AutoIndexer:
    """Automatic file indexing with watching"""
    
    def __init__(self, base_path: str):
        self.indexer = FileIndexer(base_path)
        self.watcher = FileWatcher(self.indexer)
        self.observer = Observer()
    
    async def start(self):
        """Start watching and initial indexing"""
        
        # Initial index
        print("Starting initial indexing...")
        await self.indexer.index_directory()
        
        # Start watching
        self.observer.schedule(
            self.watcher,
            self.indexer.base_path,
            recursive=True
        )
        self.observer.start()
        
        # Process events
        await self.watcher.process_events()
```

### 4. API Server for RAG Application

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Adaptive RAG API")

# Initialize components
indexer = FileIndexer("/path/to/docs")
rag = AdaptiveRAG(base_model_path="gpt2")
auto_indexer = AutoIndexer("/path/to/docs")

# Background task for auto-indexing
@app.on_event("startup")
async def startup():
    asyncio.create_task(auto_indexer.start())

class QueryRequest(BaseModel):
    query: str
    domain: Optional[str] = "default"
    k: Optional[int] = 5
    adapt: Optional[bool] = True

class IndexRequest(BaseModel):
    path: str
    extensions: Optional[List[str]] = ['.md', '.txt', '.py']

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query indexed documents with RAG"""
    
    try:
        response = await rag.generate(
            query=request.query,
            domain=request.domain,
            k=request.k,
            adapt=request.adapt
        )
        
        return {
            "query": request.query,
            "response": response,
            "domain": request.domain,
            "adapted": request.adapt
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/index")
async def index_path(request: IndexRequest):
    """Manually trigger indexing of a path"""
    
    try:
        count = await indexer.index_directory(
            Path(request.path),
            extensions=request.extensions
        )
        
        return {
            "path": request.path,
            "files_indexed": count
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/adapters")
async def list_adapters():
    """List all domain adapters"""
    
    adapters = []
    for domain, adapter in rag.domain_adapters.items():
        adapters.append({
            "domain": domain,
            "updates": adapter.update_count,
            "sparsity": adapter.sparsity,
            "parameters": adapter.adapter_dim * 2 * adapter.config.n_layer
        })
    
    return {"adapters": adapters}

@app.post("/feedback")
async def provide_feedback(query: str, response: str, rating: int, domain: str = "default"):
    """Provide feedback to improve adapter"""
    
    if rating > 3:  # Positive feedback
        # Reinforce this response pattern
        await rag.reinforce_adapter(domain, query, response)
    else:  # Negative feedback
        # Penalize this response pattern
        await rag.penalize_adapter(domain, query, response)
    
    return {"status": "feedback recorded"}
```

### 5. Usage Example

```python
import asyncio
from pathlib import Path

async def main():
    # Initialize system
    docs_path = "/home/user/projects/my-app/docs"
    indexer = FileIndexer(docs_path)
    rag = AdaptiveRAG()
    
    # Index documentation
    print("Indexing documentation...")
    await indexer.index_directory()
    
    # Query examples
    queries = [
        "How do I configure the database connection?",
        "What's the authentication flow?",
        "How do I deploy to production?",
        "Explain the caching strategy"
    ]
    
    for query in queries:
        print(f"\nQ: {query}")
        response = await rag.generate(
            query=query,
            domain="technical_docs",
            adapt=True  # Learn from this interaction
        )
        print(f"A: {response}")
    
    # After multiple queries, the adapter has learned
    print("\n\nAdapter statistics:")
    adapter = rag.domain_adapters["technical_docs"]
    print(f"Updates: {adapter.update_count}")
    print(f"Sparsity: {adapter.sparsity:.1%}")
    print(f"Parameters: {adapter.get_parameter_count():,}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Architectural Gaps Identified

### 1. **Missing: Base Model Management**
```python
# GAP: No unified interface for loading/managing different base models
# NEED: Model registry and loader
class ModelRegistry:
    """Manage different base models for RAG"""
    
    supported_models = {
        'gpt2': 'gpt2',
        'llama': 'meta-llama/Llama-2-7b',
        'mistral': 'mistralai/Mistral-7B-v0.1',
        'phi': 'microsoft/phi-2'
    }
    
    @staticmethod
    def load_model(model_name: str, quantization: str = None):
        # Handle different model types and quantization
        pass
```

### 2. **Missing: Embedding Model Integration**
```python
# GAP: No native embedding model management
# NEED: Unified embedding interface
class EmbeddingManager:
    """Manage multiple embedding models"""
    
    def __init__(self):
        self.models = {}
    
    def register_model(self, name: str, model):
        self.models[name] = model
    
    def encode(self, text: str, model_name: str = 'default'):
        # Unified encoding interface
        pass
```

### 3. **Missing: Chunking Strategy Framework**
```python
# GAP: No standardized chunking strategies
# NEED: Extensible chunking system
class ChunkingStrategy:
    """Base class for different chunking strategies"""
    
    def chunk(self, content: str, metadata: dict) -> List[Dict]:
        raise NotImplementedError

class SemanticChunker(ChunkingStrategy):
    """Chunk based on semantic boundaries"""
    pass

class SlidingWindowChunker(ChunkingStrategy):
    """Traditional sliding window"""
    pass
```

### 4. **Missing: Adapter Versioning & Rollback**
```python
# GAP: No adapter version control
# NEED: Versioning system for adapters
class AdapterVersionControl:
    """Version control for sparse adapters"""
    
    async def checkpoint(self, adapter_id: str, adapter: SparseAdapter):
        # Save versioned checkpoint
        pass
    
    async def rollback(self, adapter_id: str, version: int):
        # Rollback to previous version
        pass
    
    async def diff(self, adapter_id: str, v1: int, v2: int):
        # Show differences between versions
        pass
```

### 5. **Missing: Multi-Modal Support**
```python
# GAP: Text-only RAG
# NEED: Support for images, tables, diagrams
class MultiModalIndexer:
    """Index different content types"""
    
    async def index_image(self, image_path: Path):
        # OCR + Visual embeddings
        pass
    
    async def index_table(self, table_data):
        # Structured data embeddings
        pass
    
    async def index_diagram(self, diagram_path: Path):
        # Diagram understanding
        pass
```

### 6. **Missing: Feedback Loop Integration**
```python
# GAP: No systematic feedback collection
# NEED: Feedback-driven adapter training
class FeedbackLoop:
    """Collect and process user feedback"""
    
    async def implicit_feedback(self, query, response, dwell_time, copied):
        # Learn from user behavior
        pass
    
    async def explicit_feedback(self, query, response, rating, correction=None):
        # Learn from explicit ratings
        pass
```

### 7. **Missing: Context Window Management**
```python
# GAP: No smart context window handling
# NEED: Dynamic context selection
class ContextManager:
    """Manage context window efficiently"""
    
    def select_context(self, 
                       chunks: List[Dict],
                       query: str,
                       max_tokens: int = 2048) -> str:
        # Smart selection within token limits
        pass
    
    def compress_context(self, context: str, method='summary') -> str:
        # Compress long contexts
        pass
```

### 8. **Missing: Distributed Indexing**
```python
# GAP: Single-threaded indexing
# NEED: Parallel/distributed indexing
class DistributedIndexer:
    """Distributed file indexing"""
    
    async def index_parallel(self, paths: List[Path], workers: int = 4):
        # Parallel indexing
        pass
    
    async def index_distributed(self, paths: List[Path], nodes: List[str]):
        # Distributed across nodes
        pass
```

### 9. **Missing: Cache Layer**
```python
# GAP: No caching for embeddings/responses
# NEED: Multi-level cache
class RAGCache:
    """Cache for RAG operations"""
    
    def __init__(self):
        self.embedding_cache = LRUCache(1000)
        self.response_cache = TTLCache(100, ttl=3600)
        self.chunk_cache = DiskCache("/tmp/rag_cache")
```

### 10. **Missing: Security & Access Control**
```python
# GAP: No document-level access control
# NEED: Fine-grained permissions
class AccessControl:
    """Document and adapter access control"""
    
    async def check_access(self, user: str, document: str) -> bool:
        # Check document access
        pass
    
    async def filter_results(self, results: List, user: str) -> List:
        # Filter based on permissions
        pass
```

---

## Recommended Architecture Additions

### Priority 1: Core Infrastructure
1. **Model Registry**: Standardize base model management
2. **Adapter Versioning**: Track and rollback adapter changes
3. **Feedback Loop**: Systematic learning from usage

### Priority 2: Performance
1. **Caching Layer**: Multi-level caching system
2. **Distributed Indexing**: Scale to large document sets
3. **Context Management**: Smart context window handling

### Priority 3: Features
1. **Multi-Modal Support**: Images, tables, diagrams
2. **Security Layer**: Document-level access control
3. **Chunking Framework**: Extensible chunking strategies

---

## Getting Started

```bash
# Install dependencies
pip install hyprstream sentence-transformers watchdog fastapi

# Start Hyprstream
hyprstream start --enable-vdb --enable-adapters

# Index your documents
python -c "
import asyncio
from fsrag import FileIndexer

indexer = FileIndexer('/path/to/docs')
asyncio.run(indexer.index_directory())
"

# Start API server
uvicorn fsrag_api:app --reload

# Query your documents
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I deploy?", "domain": "docs"}'
```

---

## Summary

Hyprstream enables building adaptive RAG systems where:
1. **Local embeddings** eliminate API costs
2. **Sparse adapters** personalize responses per domain
3. **Unified system** combines storage, retrieval, and generation
4. **Continuous learning** improves responses over time

However, several architectural gaps need addressing for production use, particularly around model management, versioning, and multi-modal support.