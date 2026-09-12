"""Agent Tools (@tool definitions) for Weather and News querying."""

import logging
import os
import httpx
from dotenv import find_dotenv, load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from langfuse import observe

load_dotenv(find_dotenv())

logger = logging.getLogger("livekit.agent_tools")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

ddg_news_tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3, source="news"))
ddg_text_tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3, source="text"))


@tool
@observe(name="get_weather")
async def get_weather(city: str) -> str:
    """Get the current weather and temperature for a given city or location."""
    sanitized_city = (city or "").strip().strip("'\"")
    if not sanitized_city:
        logger.warning("get_weather received empty or whitespace city name.")
        return "Please specify a city name to check the weather."

    if not OPENWEATHER_API_KEY:
        logger.warning("OPENWEATHER_API_KEY is not configured.")
        return "Weather service is unavailable because the API key is not configured."

    logger.info("Executing get_weather tool for: '%s'", sanitized_city)
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                url,
                params={"q": sanitized_city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            )
            if resp.status_code == 200:
                data = resp.json()
                name = data.get("name", sanitized_city)
                country = data.get("sys", {}).get("country", "")
                loc_label = f"{name}, {country}" if country else name
                desc = data["weather"][0]["description"]
                temp = data["main"]["temp"]
                humidity = data["main"]["humidity"]
                result = f"Current weather in {loc_label}: {desc}, {temp}°C, humidity {humidity}%."
                logger.info("get_weather success for %s: %s, %s°C", loc_label, desc, temp)
                return result
            elif resp.status_code == 404:
                logger.info("get_weather: City '%s' not found (404)", sanitized_city)
                return f"I could not find weather data for '{sanitized_city}'. Please check the city name."
            else:
                logger.error("OpenWeather API error: HTTP %s", resp.status_code)
                return f"Weather service error: received HTTP {resp.status_code} from OpenWeather."
    except httpx.TimeoutException:
        logger.error("get_weather timed out for city: '%s'", sanitized_city)
        return "Weather request timed out. Please try again in a moment."
    except Exception as e:
        logger.exception("Unexpected error in get_weather for '%s': %s", sanitized_city, e)
        return f"Unable to retrieve weather data: {str(e)}"


@tool
@observe(name="get_news")
async def get_news(query: str) -> str:
    """Get the latest real-time news headlines and summaries on a specific topic or keyword."""
    sanitized_query = (query or "").strip().strip("'\"")
    if not sanitized_query:
        logger.warning("get_news received empty or whitespace query.")
        return "Please specify a topic or keyword to search for news."

    logger.info("Executing get_news tool for query: '%s'", sanitized_query)
    try:
        results = await ddg_news_tool.ainvoke(sanitized_query)
        if results and "No good DuckDuckGo search result" not in results and "403 Forbidden" not in results:
            logger.info("get_news success via news search for: '%s'", sanitized_query)
            return results
    except Exception as e:
        logger.warning("DuckDuckGo news search failed for '%s': %s. Falling back to text search.", sanitized_query, e)

    # Fallback to standard web text search if news endpoint rate limits or fails
    try:
        fallback_query = f"{sanitized_query} news"
        logger.info("Executing get_news fallback text search for: '%s'", fallback_query)
        results = await ddg_text_tool.ainvoke(fallback_query)
        if results and "No good DuckDuckGo search result" not in results:
            logger.info("get_news success via fallback text search for: '%s'", fallback_query)
            return results
    except Exception as e:
        logger.error("DuckDuckGo fallback search also failed for '%s': %s", sanitized_query, e)

    return f"I couldn't retrieve recent news for '{sanitized_query}' at this time."


agent_tools = [get_weather, get_news]
