import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from openai import OpenAI
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PineconeVectorStore:
    """Manages Pinecone vector database connection, embeddings, and similarity search."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        self.api_key = api_key or config.PINECONE_API_KEY
        if not self.api_key:
            raise ValueError("Pinecone API key not found. Please provide PINECONE_KEY in .env")

        self.index_name = index_name or config.PINECONE_INDEX_NAME
        self.namespace = namespace if namespace is not None else config.PINECONE_NAMESPACE

        # Initialize Pinecone client
        logger.info("Initializing Pinecone client...")
        self.pc = Pinecone(api_key=self.api_key)

        # Initialize OpenRouter client for embeddings
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
        self.openai_client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )

        # Connect to index and detect dimensions
        self.index = self._connect_index()
        self.dimension = self._detect_dimension()
        self.embedding_model = config.DIMENSION_EMBEDDING_MODELS.get(
            self.dimension, "openai/text-embedding-3-small"
        )
        logger.info(
            f"Connected to Pinecone index '{self.index_name}' (dim={self.dimension}, metric={self.metric}). "
            f"Using embedding model: {self.embedding_model}"
        )

    def _connect_index(self):
        """Validates index existence and returns the Index instance."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            raise ValueError(
                f"Index '{self.index_name}' not found. Available indexes: {existing_indexes}"
            )
        return self.pc.Index(self.index_name)

    def _detect_dimension(self) -> int:
        """Retrieves index dimension and metric."""
        desc = self.pc.describe_index(self.index_name)
        self.metric = desc.metric
        return desc.dimension

    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding vector for a single text chunk."""
        # Sanitize newline characters for cleaner tokenization
        clean_text = text.replace("\n", " ").strip()
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=clean_text,
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generates embeddings for a batch of texts."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ").strip() for t in texts[i : i + batch_size]]
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 50,
    ) -> int:
        """
        Calculates embeddings for chunks and uploads them to Pinecone.
        Each chunk item should be a dict: {'id': str, 'text': str, 'metadata': dict}
        """
        if not chunks:
            logger.warning("No chunks to upsert.")
            return 0

        target_ns = namespace if namespace is not None else self.namespace
        total_upserted = 0
        texts = [chunk["text"] for chunk in chunks]

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.get_embeddings_batch(texts)

        vectors_to_upsert = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = dict(chunk.get("metadata", {}))
            metadata["text"] = chunk["text"]  # Store raw text inside metadata for retrieval
            vectors_to_upsert.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": metadata,
            })

        logger.info(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone (namespace: '{target_ns}')...")
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=target_ns)
            total_upserted += len(batch)

        logger.info(f"Successfully upserted {total_upserted} vectors.")
        return total_upserted

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Searches for nearest vectors by query text and returns top matches with metadata."""
        target_ns = namespace if namespace is not None else self.namespace
        query_vector = self.get_embedding(query_text)

        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=target_ns,
            include_metadata=True,
            filter=filter_metadata,
        )

        matches = []
        for match in response.matches:
            matches.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata or {},
                "text": (match.metadata or {}).get("text", ""),
            })
        return matches

    def test_connection(self) -> Dict[str, Any]:
        """
        Executes a test query to verify connection, dimension consistency,
        and latency with Pinecone.
        """
        logger.info("Executing test query on Pinecone...")
        stats = self.index.describe_index_stats()
        test_text = "Тестовый запрос для проверки интеграции Pinecone и RAG"
        matches = self.query(query_text=test_text, top_k=2)

        result = {
            "status": "connected",
            "index_name": self.index_name,
            "dimension": self.dimension,
            "metric": self.metric,
            "total_vector_count": stats.total_vector_count,
            "namespaces": {k: v.vector_count for k, v in stats.namespaces.items()},
            "test_query": test_text,
            "matches_found": len(matches),
            "top_match": matches[0] if matches else None,
        }
        logger.info(f"Pinecone test successful! Total vectors: {stats.total_vector_count}")
        return result
