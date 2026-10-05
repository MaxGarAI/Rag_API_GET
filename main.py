import sys
import json
import logging
from vector_store import PineconeVectorStore
from url_processor import extract_text_from_url, chunk_text, index_url
from api_client import fetch_open_api_data, get_current_weather
from rag_agent import RAGAgent

# Setup console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("demo")


def step1_pinecone_connection():
    print("\n" + "=" * 60)
    print(" ЭТАП 1: Подключение к Pinecone и тестовый запрос")
    print("=" * 60)
    
    store = PineconeVectorStore()
    test_result = store.test_connection()
    
    print("\n[Результат проверки подключения к Pinecone]:")
    print(f"- Индекс: {test_result['index_name']}")
    print(f"- Размерность векторов: {test_result['dimension']}")
    print(f"- Метрика: {test_result['metric']}")
    print(f"- Всего векторов в индексе: {test_result['total_vector_count']}")
    print(f"- Namespaces: {test_result['namespaces']}")
    print(f"- Тестовый запрос: '{test_result['test_query']}'")
    print(f"- Найдено совпадений: {test_result['matches_found']}")
    
    return store


def step2_url_processing_and_indexing(store: PineconeVectorStore, url: str):
    print("\n" + "=" * 60)
    print(" ЭТАП 2: Обработка URL (парсинг, чанкинг, эмбеддинги, сохранение)")
    print("=" * 60)
    print(f"Целевой URL: {url}")

    # 1. Извлечение текста
    page_data = extract_text_from_url(url)
    print(f"- Заголовок страницы: {page_data['title']}")
    print(f"- Извлечено символов: {page_data['char_count']}")
    print(f"- Превью текста: {page_data['text'][:200]}...")

    # 2. Чанкинг
    chunks = chunk_text(page_data["text"], chunk_size=500, chunk_overlap=50)
    print(f"- Сформировано чанков: {len(chunks)}")
    if chunks:
        print(f"- Пример 1-го чанка ({len(chunks[0])} симв.):\n  \"{chunks[0][:150]}...\"")

    # 3. Эмбеддинги и индексация в Pinecone
    index_result = index_url(url, vector_store=store, namespace="url_knowledge", chunk_size=500, chunk_overlap=50)
    print("\n[Результат индексации страницы]:")
    print(f"- Статус: {index_result['status']}")
    print(f"- Загружено векторов: {index_result['upserted_count']}")
    print(f"- Namespace в Pinecone: {index_result['namespace']}")

    return index_result


def step3_open_api_request():
    print("\n" + "=" * 60)
    print(" ЭТАП 3: GET-запрос к открытому API и обработка ответа")
    print("=" * 60)

    # 1. Тест с Open-Meteo API (погода в реальном времени)
    print("1) Запрос к открытому API погоды (Open-Meteo):")
    weather = get_current_weather(city_name="Москва", latitude=55.7558, longitude=37.6176)
    if weather["success"]:
        data = weather["processed_data"]
        print(f"  [OK] Статус: {weather['status_code']} (за {weather['latency_seconds']} сек)")
        print(f"  Город: {data['city']}")
        print(f"  Температура: {data['temperature']}")
        print(f"  Ощущается как: {data['apparent_temperature']}")
        print(f"  Влажность: {data['relative_humidity']}")
        print(f"  Скорость ветра: {data['wind_speed']}")
    else:
        print(f"  [ОШИБКА]: {weather['error']}")

    # 2. Универсальный GET-запрос к открытому REST API (JSONPlaceholder)
    print("\n2) Запрос к открытому REST API (JSONPlaceholder):")
    api_res = fetch_open_api_data("https://jsonplaceholder.typicode.com/todos/1")
    if api_res["success"]:
        print(f"  [OK] Статус: {api_res['status_code']} (за {api_res['latency_seconds']} сек)")
        print(f"  Полученные данные: {json.dumps(api_res['data'], indent=4, ensure_ascii=False)}")
    else:
        print(f"  [ОШИБКА]: {api_res['error']}")

    return weather


def step4_rag_demonstration(store: PineconeVectorStore, query: str):
    print("\n" + "=" * 60)
    print(" ЭТАП 4: RAG-агент — поиск контекста и генерация ответа")
    print("=" * 60)
    print(f"Вопрос пользователя: \"{query}\"")

    agent = RAGAgent(vector_store=store)
    result = agent.answer_query(
        query=query,
        top_k=3,
        namespace="url_knowledge",
        include_live_weather=True,
        weather_city="Москва",
    )

    print("\n--- Найденные источники в Pinecone ---")
    for idx, src in enumerate(result["sources"], start=1):
        print(f"[{idx}] Score: {src['score']} | {src['title']}")
        print(f"    Превью: {src['text_preview']}")

    print("\n--- Ответ RAG-агента ---")
    print(result["answer"])
    print("-" * 60)


if __name__ == "__main__":
    # URL для демонстрации: статья или документация
    demo_url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    
    # 1. Pinecone
    store = step1_pinecone_connection()
    
    # 2. URL parsing & vector upsert
    step2_url_processing_and_indexing(store, demo_url)
    
    # 3. Open API request
    step3_open_api_request()
    
    # 4. RAG Agent query
    step4_rag_demonstration(store, "What is Retrieval-augmented generation and how does it work?")
