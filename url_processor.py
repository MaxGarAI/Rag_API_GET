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
    with configurable overlap between chunks.
    """
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    # If text is shorter than chunk size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]

    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    # Split by double newlines (paragraphs) first
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Break down long paragraphs into smaller units (sentences or sliding window slices)
    units = []
    for p in raw_paragraphs:
        if len(p) <= chunk_size:
            units.append(p)
        else:
            # Split long paragraph by sentences
            sentences = re.split(r"(?<=[.!?…])\s+", p)
            current_unit = ""
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(s) > chunk_size:
                    # If a single sentence exceeds chunk_size, slice by window
                    step = max(1, chunk_size - chunk_overlap)
                    for start in range(0, len(s), step):
                        sub = s[start : start + chunk_size].strip()
                        if sub:
                            units.append(sub)
                    current_unit = ""
                elif len(current_unit) + len(s) + 1 <= chunk_size:
                    current_unit = f"{current_unit} {s}".strip()
                else:
                    if current_unit:
                        units.append(current_unit)
                    current_unit = s
            if current_unit:
                units.append(current_unit)

    if not units:
        return []

    # Assemble units into final chunks with proper inter-chunk overlap
    chunks = []
    current_chunk = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        if not current_chunk:
            current_chunk = unit
        elif len(current_chunk) + len(unit) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{unit}"
        else:
            chunks.append(current_chunk)

            # Build overlap from the tail of the previous chunk
            if chunk_overlap > 0:
                overlap_source = current_chunk
                if len(overlap_source) > chunk_overlap:
                    overlap_part = overlap_source[-chunk_overlap:]
                    first_space = overlap_part.find(" ")
                    if first_space != -1 and first_space < len(overlap_part) - 1:
                        overlap_part = overlap_part[first_space + 1:].strip()
                else:
                    overlap_part = overlap_source.strip()

                if overlap_part and len(overlap_part) + len(unit) + 2 <= chunk_size:
                    current_chunk = f"{overlap_part}\n\n{unit}"
                else:
                    current_chunk = unit
            else:
                current_chunk = unit

    if current_chunk and (not chunks or current_chunk != chunks[-1]):
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
