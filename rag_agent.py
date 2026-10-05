import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import config
from vector_store import PineconeVectorStore
from api_client import get_current_weather

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG Agent combining Pinecone vector store retrieval,
    external open API data integration, and LLM text generation.
    """

    def __init__(
        self,
        vector_store: Optional[PineconeVectorStore] = None,
        llm_model: Optional[str] = None,
    ):
        self.vector_store = vector_store or PineconeVectorStore()
        self.llm_model = llm_model or config.DEFAULT_LLM_MODEL
        self.openai_client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )
        logger.info(f"RAGAgent initialized with model: {self.llm_model}")

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves most relevant knowledge chunks from Pinecone."""
        return self.vector_store.query(query, top_k=top_k, namespace=namespace)

    def answer_query(
        self,
        query: str,
        top_k: int = 3,
        namespace: Optional[str] = None,
        include_live_weather: bool = False,
        weather_city: str = "Moscow",
    ) -> Dict[str, Any]:
        """
        Executes complete RAG cycle:
        1. Retrieves relevant documents from Pinecone
        2. Optionally augments with live open API data
        3. Formulates structured prompt with context
        4. Calls LLM via OpenRouter for final synthesis
        """
        logger.info(f"Processing query: '{query}'")

        # 1. Retrieve knowledge from Pinecone
        retrieved_docs = self.retrieve_context(query, top_k=top_k, namespace=namespace)

        # 2. Build context string
        context_parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            meta = doc.get("metadata", {})
            title = meta.get("title", "Документ")
            url = meta.get("url", "")
            source_info = f"[{i}] {title}" + (f" ({url})" if url else "")
            context_parts.append(f"{source_info}:\n{doc.get('text', '')}")

        # 3. Optional live API augmentation
        live_api_info = None
        if include_live_weather:
            weather_data = get_current_weather(city_name=weather_city)
            if weather_data.get("success"):
                live_api_info = weather_data["processed_data"]
                context_parts.append(
                    f"[Данные из Open-Meteo API]: Погода в {weather_city}: "
                    f"температура {live_api_info['temperature']}, влажность {live_api_info['relative_humidity']}, "
                    f"ветер {live_api_info['wind_speed']}."
                )

        context_text = "\n\n".join(context_parts) if context_parts else "Нет релевантного контекста в базе знаний."

        # 4. Formulate prompt for LLM
        system_instruction = (
            "Ты — интеллектуальный RAG-ассистент. Отвечай на вопросы пользователя точно, "
            "опираясь на предоставленный контекст из базы знаний и API данных. "
            "Если в контексте нет точного ответа, укажи на это честно. "
            "Указывай ссылки на источники [1], [2], если они есть в контексте."
        )

        user_prompt = (
            f"Контекст для ответа:\n"
            f"---------------------\n"
            f"{context_text}\n"
            f"---------------------\n\n"
            f"Вопрос пользователя: {query}\n\n"
            f"Ответ:"
        )

        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )

        answer_text = response.choices[0].message.content

        return {
            "query": query,
            "answer": answer_text,
            "sources": [
                {
                    "score": round(doc["score"], 4),
                    "title": doc.get("metadata", {}).get("title"),
                    "url": doc.get("metadata", {}).get("url"),
                    "text_preview": doc.get("text", "")[:150] + ("..." if len(doc.get("text", "")) > 150 else ""),
                }
                for doc in retrieved_docs
            ],
            "live_api_data": live_api_info,
        }
