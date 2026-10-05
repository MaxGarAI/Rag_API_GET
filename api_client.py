import logging
import time
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


def fetch_open_api_data(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    Executes a GET request to an open public API with comprehensive error handling.
    Safely captures status code, response time, and handles JSON or text payloads.
    """
    default_headers = {
        "User-Agent": "RAG-Agent-Demo/1.0",
        "Accept": "application/json",
    }
    if headers:
        default_headers.update(headers)

    start_time = time.time()
    result = {
        "success": False,
        "status_code": None,
        "url": url,
        "data": None,
        "error": None,
        "latency_seconds": 0.0,
    }

    try:
        logger.info(f"Sending GET request to API: {url} with params={params}")
        response = requests.get(url, params=params, headers=default_headers, timeout=timeout)
        result["status_code"] = response.status_code
        result["url"] = response.url
        result["latency_seconds"] = round(time.time() - start_time, 3)

        response.raise_for_status()

        # Parse JSON or fallback to text
        try:
            result["data"] = response.json()
        except ValueError:
            result["data"] = response.text

        result["success"] = True
        logger.info(f"API request succeeded in {result['latency_seconds']}s (status={response.status_code})")

    except requests.exceptions.Timeout:
        result["error"] = f"Request timed out after {timeout} seconds"
        logger.error(result["error"])
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP Error {response.status_code}: {e}"
        logger.error(result["error"])
    except requests.exceptions.ConnectionError as e:
        result["error"] = f"Connection failed: {e}"
        logger.error(result["error"])
    except requests.exceptions.RequestException as e:
        result["error"] = f"Request error: {e}"
        logger.error(result["error"])
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"
        logger.error(result["error"])

    return result


def get_current_weather(
    city_name: str = "Moscow",
    latitude: float = 55.7558,
    longitude: float = 37.6176,
) -> Dict[str, Any]:
    """
    Queries open API (Open-Meteo) for real-time weather and processes the response.
    Requires no API keys.
    """
    api_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m",
        "timezone": "auto",
    }

    res = fetch_open_api_data(api_url, params=params)
    if not res["success"]:
        return res

    raw = res["data"]
    current = raw.get("current", {})
    units = raw.get("current_units", {})

    processed = {
        "city": city_name,
        "temperature": f"{current.get('temperature_2m')} {units.get('temperature_2m', '°C')}",
        "apparent_temperature": f"{current.get('apparent_temperature')} {units.get('apparent_temperature', '°C')}",
        "relative_humidity": f"{current.get('relative_humidity_2m')} {units.get('relative_humidity_2m', '%')}",
        "wind_speed": f"{current.get('wind_speed_10m')} {units.get('wind_speed_10m', 'km/h')}",
        "precipitation": f"{current.get('precipitation')} {units.get('precipitation', 'mm')}",
        "time": current.get("time"),
        "raw_response": raw,
    }
    res["processed_data"] = processed
    return res


def get_sample_post(post_id: int = 1) -> Dict[str, Any]:
    """
    Queries JSONPlaceholder open REST API for sample post data.
    """
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"
    return fetch_open_api_data(url)
