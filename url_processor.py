import hashlib
import logging
import re
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from vector_store import PineconeVectorStore

logger = logging.getLogger(__name__)


def extract_text_from_url(url: str, timeout: int = 15) -> Dict[str, Any]:
    """
    Downloads webpage HTML, extracts title and clean text content,
    stripping out scripts, styles, navigation and boilerplate tags.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    logger.info(f"Fetching URL: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        raise RuntimeError(f"Error fetching URL '{url}': {e}") from e

    # Detect proper encoding
    if response.encoding is None or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "svg"]):
        tag.decompose()

    # Prefer main content container if present
    main_content = soup.find("article") or soup.find("main") or soup.find("div", {"id": "content"}) or soup.body
    raw_text = main_content.get_text(separator="\n") if main_content else soup.get_text(separator="\n")

    # Clean text: normalize whitespace and blank lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]
    clean_text = "\n".join([line for line in lines if line])

    logger.info(f"Extracted {len(clean_text)} characters from '{title}' ({url})")
    return {
        "url": url,
        "title": title,
        "text": clean_text,
        "char_count": len(clean_text),
    }


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Splits text into chunks respecting paragraph / sentence boundaries where feasible,
    with configurable overlap.
    """
    if not text:
        return []

    # If text is shorter than chunk size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    # Split by double newlines (paragraphs) first
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 1 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If paragraph itself is larger than chunk_size, split by sentences/length
            if len(para) > chunk_size:
                step = max(1, chunk_size - chunk_overlap)
                for start in range(0, len(para), step):
                    sub = para[start : start + chunk_size].strip()
                    if sub:
                        chunks.append(sub)
                current_chunk = ""
            else:
                # Add overlap from end of previous text if applicable
                current_chunk = para

    if current_chunk and current_chunk not in chunks:
        chunks.append(current_chunk)

    return chunks


def process_url_to_chunks(
    url: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fetches URL, extracts text, and creates chunk items with metadata and unique IDs.
    """
    data = extract_text_from_url(url)
    text_chunks = chunk_text(data["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    chunks = []
    total = len(text_chunks)

    for idx, text in enumerate(text_chunks):
        chunk_id = f"url-{url_hash}-chk-{idx:04d}"
        chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": {
                "source": "url",
                "url": url,
                "title": data["title"],
                "chunk_index": idx,
                "total_chunks": total,
                "char_count": len(text),
            },
        })

    logger.info(f"Created {len(chunks)} chunks for URL '{url}'")
    return chunks


def index_url(
    url: str,
    vector_store: PineconeVectorStore,
    namespace: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Dict[str, Any]:
    """
    End-to-end pipeline:
    1. Parse URL content
    2. Divide into chunks
    3. Generate embeddings
    4. Save to Pinecone vector store
    """
    chunks = process_url_to_chunks(url, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return {"status": "empty", "url": url, "chunks_count": 0}

    upserted_count = vector_store.upsert_chunks(chunks, namespace=namespace)

    return {
        "status": "success",
        "url": url,
        "title": chunks[0]["metadata"]["title"],
        "chunks_count": len(chunks),
        "upserted_count": upserted_count,
        "namespace": namespace or vector_store.namespace,
    }
