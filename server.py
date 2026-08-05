#!/usr/bin/env python3
"""
multi-api-mcp (Python version)
---------------------------------------------------------
A single MCP server that exposes SEVERAL free public APIs
as separate tools. This is a direct Python port of server.js
using the official MCP Python SDK (FastMCP) — 15 tools,
same free APIs, same behavior.

Free APIs used here (no API key required unless noted):
 - get_coordinates      -> Open-Meteo Geocoding (place name -> lat/long, for the
                            weather tools below — avoids the model guessing coordinates)
 - get_weather          -> Open-Meteo (current conditions)
 - get_weather_forecast -> Open-Meteo (hourly forecast, up to 16 days ahead)
 - get_weather_history  -> Open-Meteo Archive (past daily weather)
 - get_exchange_rate    -> open.er-api.com
 - get_random_joke      -> JokeAPI
 - get_random_fact      -> uselessfacts.jsph.pl
 - get_nasa_apod        -> NASA (Astronomy Picture of the Day) — free DEMO_KEY works,
                            or set NASA_API_KEY env var for your own free key
 - get_wikipedia_summary -> Wikipedia REST API (no key)
 - get_github_user      -> GitHub REST API (no key needed for public reads)
 - generate_image       -> Pollinations.ai (AI image generation, no key)
 - create_diagram       -> Kroki.io (flowcharts/UML/diagrams, no key)
 - search_youtube       -> YouTube Data API v3 (needs YOUTUBE_API_KEY, free)
 - search_arxiv         -> Arxiv (academic papers, no key)
 - list_capabilities    -> built-in, no external API

Run:  python server.py
---------------------------------------------------------
"""

import base64
import os
import re
import secrets
import zlib
from typing import Optional
from urllib.parse import quote
from xml.etree import ElementTree

import httpx
import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

load_dotenv()

# YouTube Data API v3 key -- unlike most of the other free APIs here, this
# one genuinely requires a key (free, no billing needed at this quota tier,
# but not fully keyless like Wikipedia/weather/etc). Optional at startup,
# same pattern as NASA_API_KEY: the tool below gives a clear setup message
# rather than a confusing failure if it's missing.
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")

# GitHub REST API -- reads are genuinely public/keyless, but GitHub's
# ANONYMOUS rate limit is only 60 requests/hour PER IP, shared across every
# single caller hitting this server (all your own testing today, plus
# anyone else using it) -- easy to exhaust during a demo day of heavy
# testing. Adding a free personal access token (classic, no scopes needed
# for public reads) raises that same limit to 5,000 requests/hour -- same
# "optional env var, clear message if missing/exhausted" pattern as
# YOUTUBE_API_KEY and NASA_API_KEY above.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# Optional MCP-level authentication -- only used in "streamable-http" mode
# (see the __main__ block at the bottom). If MCP_AUTH_TOKEN is unset, the
# server behaves exactly as before: open, no auth required. If it IS set,
# every request must include "Authorization: Bearer <that token>" or it's
# rejected with 401 before it ever reaches any tool. This is a simple shared-
# secret gate, not full OAuth -- appropriate for a small number of trusted
# clients (like a colleague's chatbot) rather than public self-service signup.
MCP_AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN")

# host/port here only matter when running in "streamable-http" mode (see the
# __main__ block at the bottom) -- in the normal "stdio" mode (the default,
# unchanged from before), FastMCP ignores them entirely since stdio has no
# network address at all. Read from the environment so a hosting platform's
# assigned PORT is respected when this runs standalone.
mcp = FastMCP(
    "multi-api-mcp",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
)


def safe_fetch_json(url: str, headers: Optional[dict] = None) -> dict:
    """Small helper so every tool has consistent error handling."""
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            res = client.get(url, headers=headers or {})
    except httpx.ConnectError as err:
        raise RuntimeError(
            f"Couldn't connect to {url.split('/')[2]} at all (connection error: {err}). "
            "This usually means either there's no internet connection right now, or "
            "something on this network (a corporate firewall, VPN, or antivirus) is "
            "blocking that specific site. Try opening the URL in a normal browser tab "
            "to check."
        ) from err
    except httpx.TimeoutException as err:
        raise RuntimeError(
            f"The request to {url.split('/')[2]} timed out after 15s ({err}). "
            "The site may be slow or blocked by a network filter."
        ) from err
    if True:
        if res.status_code in (429, 403) and "api.nasa.gov" in url and "DEMO_KEY" in url:
            raise RuntimeError(
                "NASA's DEMO_KEY is rate-limited right now (it's a shared key used by "
                "everyone who hasn't registered their own — only 30 requests/hour, 50/day, "
                "total, for the whole world, so it runs out fast). Get a free personal key "
                "in about 10 seconds at https://api.nasa.gov, then set "
                "NASA_API_KEY=<your key> in your .env file and restart the server."
            )
        if res.status_code == 403 and "api.github.com" in url and res.headers.get("X-RateLimit-Remaining") == "0":
            # GitHub returns 403 (NOT 429) for its rate limit, and an
            # unauthenticated request only gets 60/hour, shared across every
            # caller hitting this server -- easy to exhaust during heavy
            # testing. Without this check it fell through to the generic
            # ">= 400" branch below and looked like "user not found" or a
            # vague failure, when the real cause (and fix) is completely
            # different and worth surfacing clearly.
            if GITHUB_TOKEN:
                raise RuntimeError(
                    "GitHub's API rate limit was hit even with a token configured "
                    "(5,000 requests/hour) -- unusual, but it can happen under very "
                    "heavy testing. Try again shortly."
                )
            raise RuntimeError(
                "GitHub's API rate limit was hit (60 requests/hour for unauthenticated "
                "requests, shared across everyone using this server -- easy to exhaust "
                "during testing). Get a free personal access token in about a minute at "
                "https://github.com/settings/tokens (no scopes needed for public reads), "
                "then set GITHUB_TOKEN=<your token> in .env and restart -- that raises "
                "the limit to 5,000 requests/hour."
            )
        if res.status_code == 429:
            raise RuntimeError("Too many requests to this API right now (rate-limited). Try again in a bit.")
        if res.status_code >= 400:
            safe_url = re.sub(r"([?&](?:api_key|key)=)[^&]+", r"\1***", url)
            raise RuntimeError(f"Request to {safe_url} failed: {res.status_code} {res.reason_phrase}")
        raw = res.text
        if not raw:
            raise RuntimeError(
                "The API returned no data for this request (empty response). "
                "This usually means the country/topic/date isn't supported by this free API."
            )
        try:
            return res.json()
        except ValueError:
            raise RuntimeError("The API returned a response that wasn't valid data. Try different inputs.")


# ---------------------------------------------------------
# TOOL 0: Geocoding — place name to coordinates (Open-Meteo — free, no key)
# ---------------------------------------------------------
@mcp.tool()
def get_coordinates(place: str) -> str:
    """Look up the latitude and longitude for a place name (city, town, or landmark)
    using Open-Meteo's free geocoding API. Call this FIRST whenever get_weather,
    get_weather_forecast, or get_weather_history need coordinates for a named place —
    use the returned latitude/longitude with those tools instead of guessing
    coordinates from memory, which can be inaccurate for smaller or less well-known
    places."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote(place)}&count=1&language=en&format=json"
    data = safe_fetch_json(url)
    results = data.get("results")
    if not results:
        raise RuntimeError(
            f"Could not find a location matching '{place}'. Try a different spelling "
            "or a nearby larger city/town."
        )
    r = results[0]
    name_parts = [r.get("name")]
    if r.get("admin1"):
        name_parts.append(r["admin1"])
    if r.get("country"):
        name_parts.append(r["country"])
    return (
        f"Location: {', '.join(p for p in name_parts if p)}\n"
        f"Latitude: {r['latitude']}\n"
        f"Longitude: {r['longitude']}"
    )


# ---------------------------------------------------------
# TOOL 1: Weather (Open-Meteo — free, no key)
# ---------------------------------------------------------
@mcp.tool()
def get_weather(latitude: float, longitude: float) -> str:
    """Get current weather for a latitude/longitude using Open-Meteo (free, no API key)."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,"
        "cloud_cover,uv_index,is_day,wind_speed_10m,weather_code"
        "&daily=sunrise,sunset&timezone=auto"
    )
    data = safe_fetch_json(url)
    c = data.get("current", {})
    daily = data.get("daily", {})
    lines = [
        f"Temperature: {c.get('temperature_2m', 'N/A')}°C (feels like {c.get('apparent_temperature', 'N/A')}°C)",
        f"Humidity: {c.get('relative_humidity_2m', 'N/A')}%",
        f"Cloud cover: {c.get('cloud_cover', 'N/A')}%",
        f"UV index: {c.get('uv_index', 'N/A')}",
        f"Precipitation right now: {c.get('precipitation', 'N/A')}mm",
        f"Wind speed: {c.get('wind_speed_10m', 'N/A')} km/h",
        f"Day or night: {'Day' if c.get('is_day') == 1 else 'Night'}",
        f"Sunrise: {(daily.get('sunrise') or ['N/A'])[0]}",
        f"Sunset: {(daily.get('sunset') or ['N/A'])[0]}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------
# TOOL 1b: Hourly weather forecast (Open-Meteo — free, no key)
# ---------------------------------------------------------
@mcp.tool()
def get_weather_forecast(latitude: float, longitude: float, days: int = 2) -> str:
    """Get an hourly weather forecast (temperature, rain chance, windspeed, humidity, UV,
    cloud cover) for a latitude/longitude, for the next 1-16 days, using Open-Meteo (free,
    no API key). Use this for 'next few hours' or 'next N days' style forecast questions —
    NOT for current-moment weather (use get_weather) and NOT for past dates (use
    get_weather_history)."""
    days = max(1, min(16, days))
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
        "&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,"
        "precipitation_probability,weathercode,windspeed_10m,cloud_cover,uv_index"
        f"&daily=sunrise,sunset&forecast_days={days}&timezone=auto"
    )
    data = safe_fetch_json(url)
    h = data["hourly"]
    lines = [
        f"{h['time'][i]} — {h['temperature_2m'][i]}°C (feels {h['apparent_temperature'][i]}°C), "
        f"humidity {h['relative_humidity_2m'][i]}%, rain chance {h['precipitation_probability'][i]}%, "
        f"wind {h['windspeed_10m'][i]} km/h, cloud {h['cloud_cover'][i]}%, UV {h['uv_index'][i]}"
        for i in range(len(h["time"]))
    ]
    daily = data.get("daily", {})
    sun_info = [
        f"{s[:10]}: sunrise {s[11:]} / sunset {daily['sunset'][i][11:]}"
        for i, s in enumerate(daily.get("sunrise", []))
    ]
    return "\n".join(lines) + "\n\nSunrise/Sunset:\n" + "\n".join(sun_info)


# ---------------------------------------------------------
# TOOL 1c: Past weather (Open-Meteo Archive — free, no key)
# ---------------------------------------------------------
@mcp.tool()
def get_weather_history(latitude: float, longitude: float, start_date: str, end_date: str) -> str:
    """Get actual recorded daily weather (max/min temperature, rainfall, max wind) for a
    latitude/longitude for a past date range (format YYYY-MM-DD), using Open-Meteo's Archive
    API (free, no API key). Use this for 'last N days' / historical weather questions —
    NOT for current or future weather."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode"
        "&timezone=auto"
    )
    data = safe_fetch_json(url)
    d = data["daily"]
    n = len(d["time"])
    if n > 45:
        header = "date,high_c,low_c,rain_mm,max_wind_kmh"
        lines = [
            f"{d['time'][i]},{d['temperature_2m_max'][i]},{d['temperature_2m_min'][i]},"
            f"{d['precipitation_sum'][i]},{d['windspeed_10m_max'][i]}"
            for i in range(n)
        ]
        return header + "\n" + "\n".join(lines)
    lines = [
        f"{d['time'][i]} — high {d['temperature_2m_max'][i]}°C, low {d['temperature_2m_min'][i]}°C, "
        f"rainfall {d['precipitation_sum'][i]}mm, max wind {d['windspeed_10m_max'][i]} km/h"
        for i in range(n)
    ]
    return "\n".join(lines)


# ---------------------------------------------------------
# TOOL 2: Exchange rate (open.er-api.com — free, no key)
# ---------------------------------------------------------
@mcp.tool()
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get the CURRENT (right-now) exchange rate from one currency to another
    (e.g. USD -> EUR). This only returns today's rate — it has no memory of
    past days. For 'last N days' / date-range / historical exchange rate
    questions, use get_exchange_rate_history instead of calling this
    repeatedly."""
    from_c = from_currency.upper()
    to_c = to_currency.upper()
    url = f"https://open.er-api.com/v6/latest/{from_c}"
    data = safe_fetch_json(url)
    rate = data.get("rates", {}).get(to_c)
    if rate is None:
        raise RuntimeError(f"Could not find rate for {to_c}")
    return f"1 {from_c} = {rate} {to_c}"


# ---------------------------------------------------------
# TOOL 2b: Historical exchange rates (Frankfurter / ECB — free, no key)
# ---------------------------------------------------------
@mcp.tool()
def get_exchange_rate_history(from_currency: str, to_currency: str, start_date: str, end_date: str) -> str:
    """Get the exchange rate from one currency to another for EACH DAY in a
    past date range (format YYYY-MM-DD), e.g. 'USD to INR for the last 3
    days'. Use this instead of get_exchange_rate for any question about past
    days, a date range, or a trend/chart over time — get_exchange_rate only
    has today's single rate and cannot answer these. Source: Frankfurter
    (European Central Bank reference rates, free, no key). Covers major
    world currencies (USD, EUR, GBP, INR, JPY, and ~30 others) — if a
    currency isn't covered, this will come back empty."""
    from_c = from_currency.upper()
    to_c = to_currency.upper()
    url = f"https://api.frankfurter.app/{start_date}..{end_date}?from={from_c}&to={to_c}"
    data = safe_fetch_json(url)
    rates = data.get("rates", {})
    if not rates:
        raise RuntimeError(
            f"No historical rate data for {from_c} -> {to_c} between {start_date} and "
            f"{end_date}. This free source (Frankfurter/ECB) only covers major world "
            "currencies and dates from 1999 onward on weekdays — check the currency "
            "codes and date range."
        )
    lines = [
        f"{date}: 1 {from_c} = {day_rates[to_c]} {to_c}"
        for date, day_rates in sorted(rates.items())
        if to_c in day_rates
    ]
    return "\n".join(lines)


# ---------------------------------------------------------
# TOOL 3: Random joke (JokeAPI — free, no key)
# ---------------------------------------------------------
@mcp.tool()
def get_random_joke(category: str = "Any") -> str:
    """Get a random clean joke from JokeAPI. Category must be one of:
    Any, Programming, Misc, Pun, Dark, Spooky, Christmas."""
    valid = {"Any", "Programming", "Misc", "Pun", "Dark", "Spooky", "Christmas"}
    if category not in valid:
        category = "Any"
    url = f"https://v2.jokeapi.dev/joke/{category}?safe-mode"
    data = safe_fetch_json(url)
    if data.get("type") == "single":
        return data["joke"]
    return f"{data['setup']}\n{data['delivery']}"


# ---------------------------------------------------------
# TOOL 5: Random fact (uselessfacts.jsph.pl — free, no key)
# ---------------------------------------------------------
@mcp.tool()
def get_random_fact() -> str:
    """Get a random trivia fact."""
    url = "https://uselessfacts.jsph.pl/api/v2/facts/random?language=en"
    data = safe_fetch_json(url)
    return data["text"]


# ---------------------------------------------------------
# TOOL 6: NASA Astronomy Picture of the Day
# ---------------------------------------------------------
@mcp.tool()
def get_nasa_apod(date: Optional[str] = None) -> str:
    """Get NASA's Astronomy Picture of the Day (title, explanation, and image/video URL),
    optionally for a specific past date (format YYYY-MM-DD). Defaults to today if omitted."""
    api_key = os.environ.get("NASA_API_KEY") or "DEMO_KEY"
    date_param = f"&date={date}" if date else ""
    url = f"https://api.nasa.gov/planetary/apod?api_key={api_key}{date_param}"
    data = safe_fetch_json(url)
    return (
        f"Title: {data['title']}\n"
        f"Date: {data['date']}\n"
        f"Media URL: {data['url']}\n\n"
        f"Explanation: {data['explanation']}"
    )


# ---------------------------------------------------------
# TOOL 7: Wikipedia summary (free, no key)
# ---------------------------------------------------------
@mcp.tool()
def get_wikipedia_summary(topic: str) -> str:
    """Get a short summary/intro of any topic from Wikipedia. Handles free-text
    topics (typos, wrong casing, extra words) by resolving to the correct page
    title first, so you don't need to know the exact canonical title."""
    headers = {"User-Agent": "multi-api-mcp-poc/1.0 (POC demo; contact: local-dev)"}

    resolved_title = topic
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php?action=opensearch"
            f"&search={quote(topic)}&limit=1&namespace=0&format=json"
        )
        search_data = safe_fetch_json(search_url, headers=headers)
        if isinstance(search_data, list) and len(search_data) > 1 and search_data[1]:
            resolved_title = search_data[1][0]
    except RuntimeError:
        pass

    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(resolved_title)}"
    try:
        data = safe_fetch_json(url, headers=headers)
    except RuntimeError as err:
        msg = str(err)
        is_network_error = "connect" in msg.lower() or "timed out" in msg.lower()
        if resolved_title != topic or is_network_error:
            raise
        raise RuntimeError(
            f"Couldn't find a Wikipedia page matching '{topic}' (tried resolving it via "
            "Wikipedia search first, but that didn't turn up a match either). Double-check "
            "the spelling, or try a more specific/complete name."
        )
    lines = [data["title"], "", data.get("extract", "")]
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page")
    if page_url:
        lines.append(f"\nRead more: {page_url}")
    return "\n".join(lines)


# ---------------------------------------------------------
# TOOL 8: GitHub public user info (free, no key)
# ---------------------------------------------------------
@mcp.tool()
def get_github_user(username: str) -> str:
    """Get public profile info for a GitHub username (bio, repo count, followers, etc.)."""
    url = f"https://api.github.com/users/{quote(username)}"
    headers = {"User-Agent": "multi-api-mcp"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    data = safe_fetch_json(url, headers=headers)
    return "\n".join(
        [
            f"Name: {data.get('name') or data.get('login')}",
            f"Bio: {data.get('bio') or 'N/A'}",
            f"Public repos: {data.get('public_repos')}",
            f"Followers: {data.get('followers')}",
            f"Following: {data.get('following')}",
            f"Profile: {data.get('html_url')}",
        ]
    )


# ---------------------------------------------------------
# TOOL 10: AI image generation (Pollinations.ai — free, no key)
# ---------------------------------------------------------
@mcp.tool()
def generate_image(prompt: str) -> str:
    """Generate an AI image from a text description/prompt using Pollinations.ai
    (free, no API key). Use this when the user asks to draw, generate, create, or
    show a picture/image/art of something -- NOT for real photos of real current
    events or real people (this generates AI art, not real photographs). Returns
    a direct image URL -- always include that exact URL as-is in your reply
    (don't paraphrase or drop it) so it can be displayed to the user."""
    if not prompt or not prompt.strip():
        raise RuntimeError("A description of the image to generate is required.")
    prompt = prompt.strip()
    # An overly long prompt embedded directly in a URL can trip a "request
    # too large" / URL-too-long failure at the HTTP layer -- same failure
    # class as any other API that receives unbounded text. Cap it defensively
    # rather than let a verbose model-generated description crash the tool.
    if len(prompt) > 1200:
        prompt = prompt[:1200]
    encoded = quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?nologo=true"
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            res = client.get(url)
    except httpx.TimeoutException as err:
        raise RuntimeError(
            f"Image generation timed out after 30s ({err}). Pollinations can be slow "
            "under load -- try again, or simplify the description."
        ) from err
    except httpx.ConnectError as err:
        raise RuntimeError(f"Couldn't connect to Pollinations' image service: {err}") from err
    if res.status_code >= 400:
        raise RuntimeError(f"Image generation failed: {res.status_code} {res.reason_phrase}")
    return f"Here's the generated image: {url}"


# ---------------------------------------------------------
# TOOL 11: Diagram generation (Kroki.io — free, no key)
# ---------------------------------------------------------
_KROKI_DIAGRAM_TYPES = {
    "mermaid", "graphviz", "plantuml", "blockdiag", "bpmn", "c4plantuml",
    "ditaa", "erd", "nomnoml", "pikchr", "structurizr", "svgbob", "vega",
    "vegalite", "wavedrom", "excalidraw", "d2",
}


@mcp.tool()
def create_diagram(diagram_type: str, diagram_source: str) -> str:
    """Generate a real diagram image (flowchart, sequence diagram, UML,
    mind map, ER diagram, network/architecture diagram, etc.) from text using
    Kroki.io (free, no API key). Write the diagram_source in the diagram
    language matching diagram_type -- most commonly 'mermaid' (flowcharts,
    sequence diagrams, mind maps, Gantt charts -- easiest to write free-text
    syntax for) or 'graphviz'/'d2' (network and architecture diagrams). Other
    supported diagram_type values: plantuml, blockdiag, bpmn, c4plantuml,
    ditaa, erd, nomnoml, pikchr, structurizr, svgbob, vega, vegalite,
    wavedrom, excalidraw. Use this whenever the user asks to draw, diagram,
    visualize, or map out a process, flow, structure, or relationship --
    NOT for charts of numeric data (use show_chart for that instead). Returns
    a direct image URL -- always include that exact URL as-is in your reply
    (don't paraphrase or drop it) so it can be displayed to the user."""
    if not diagram_source or not diagram_source.strip():
        raise RuntimeError("Diagram content (diagram_source) is required.")
    diagram_source = diagram_source.strip()
    # Diagrams can legitimately be verbose, but not unbounded -- cap defensively
    # so an oversized diagram_source can't produce a URL long enough to fail,
    # same principle as generate_image's prompt cap above.
    if len(diagram_source) > 8000:
        diagram_source = diagram_source[:8000]
    dtype = (diagram_type or "").strip().lower()
    if dtype not in _KROKI_DIAGRAM_TYPES:
        raise RuntimeError(
            f"'{diagram_type}' isn't a supported diagram type. Use one of: "
            + ", ".join(sorted(_KROKI_DIAGRAM_TYPES))
        )
    compressed = zlib.compress(diagram_source.encode("utf-8"), 9)
    encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
    url = f"https://kroki.io/{dtype}/svg/{encoded}"
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            res = client.get(url)
    except httpx.TimeoutException as err:
        raise RuntimeError(f"Diagram generation timed out after 30s ({err}). Try again.") from err
    except httpx.ConnectError as err:
        raise RuntimeError(f"Couldn't connect to Kroki's diagram service: {err}") from err
    if res.status_code >= 400:
        detail = res.text[:300] if res.text else f"{res.status_code} {res.reason_phrase}"
        raise RuntimeError(f"Diagram generation failed (likely invalid {dtype} syntax): {detail}")
    return f"Here's the generated diagram: {url}"


# ---------------------------------------------------------
# TOOL 12: YouTube video search (YouTube Data API v3 -- free, but DOES need a
# key, unlike most tools in this file -- see YOUTUBE_API_KEY above)
# ---------------------------------------------------------
@mcp.tool()
def search_youtube(query: str, max_results: int = 5) -> str:
    """Search YouTube for videos matching a query -- returns each result's
    title, channel name, publish date, and a direct watch URL. Use this for
    'find a video about X' / 'youtube search for Y' style requests. NOT for
    looking up a specific video the user already named or linked -- there's
    nothing to search for in that case, just use what they gave you
    directly."""
    if not YOUTUBE_API_KEY:
        raise RuntimeError(
            "YouTube search isn't set up yet -- add YOUTUBE_API_KEY to the .env "
            "file (a free key from Google Cloud Console -- enable 'YouTube Data "
            "API v3', then create an API key under Credentials) and restart the "
            "server."
        )
    if not query or not query.strip():
        raise RuntimeError("A search query is required.")
    max_results = max(1, min(10, max_results))
    url = (
        "https://www.googleapis.com/youtube/v3/search"
        f"?part=snippet&type=video&maxResults={max_results}"
        f"&q={quote(query.strip())}&key={YOUTUBE_API_KEY}"
    )
    data = safe_fetch_json(url)
    items = data.get("items", [])
    lines = []
    for it in items:
        video_id = (it.get("id") or {}).get("videoId")
        if not video_id:
            continue
        snippet = it.get("snippet", {})
        title = snippet.get("title", "Untitled")
        channel = snippet.get("channelTitle", "Unknown channel")
        published = (snippet.get("publishedAt") or "")[:10]
        lines.append(
            f"{title} — {channel} ({published})\n"
            f"https://www.youtube.com/watch?v={video_id}"
        )
    if not lines:
        raise RuntimeError(f"No YouTube videos found for '{query}'.")
    return "\n\n".join(lines)


# ---------------------------------------------------------
# TOOL 13: Arxiv paper search (arxiv.org — free, no key)
# ---------------------------------------------------------
@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search Arxiv for real academic papers (physics, CS, math, and more)
    matching a query -- returns each paper's title, authors, publish date,
    a short abstract, and a direct link to the paper. Use this for research
    papers, academic/scientific topics, or 'find a paper about X' style
    requests -- for general topic summaries use get_wikipedia_summary
    instead, since not every topic has a relevant academic paper."""
    if not query or not query.strip():
        raise RuntimeError("A search query is required.")
    max_results = max(1, min(10, max_results))
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=all:{quote(query.strip())}&start=0&max_results={max_results}"
    )
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            res = client.get(url)
    except httpx.TimeoutException as err:
        raise RuntimeError(f"Arxiv search timed out after 20s ({err}). Try again.") from err
    except httpx.ConnectError as err:
        raise RuntimeError(f"Couldn't connect to Arxiv: {err}") from err
    if res.status_code >= 400:
        raise RuntimeError(f"Arxiv search failed: {res.status_code} {res.reason_phrase}")

    try:
        root = ElementTree.fromstring(res.text)
    except ElementTree.ParseError as err:
        raise RuntimeError(f"Arxiv returned a response that couldn't be parsed: {err}") from err

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    if not entries:
        raise RuntimeError(f"No Arxiv papers found for '{query}'.")

    lines = []
    for entry in entries:
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        id_el = entry.find("atom:id", ns)
        authors = [
            (a.find("atom:name", ns).text or "").strip()
            for a in entry.findall("atom:author", ns)
            if a.find("atom:name", ns) is not None
        ]
        title = " ".join((title_el.text or "").split()) if title_el is not None else "Untitled"
        summary = " ".join((summary_el.text or "").split()) if summary_el is not None else ""
        if len(summary) > 400:
            summary = summary[:400].rsplit(" ", 1)[0] + "…"
        published = (published_el.text or "")[:10] if published_el is not None else ""
        link = id_el.text.strip() if id_el is not None and id_el.text else ""
        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        lines.append(
            f"{title}\n{author_str} ({published})\n{summary}\n{link}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------
# TOOL 9: List capabilities (built-in, no external API)
# ---------------------------------------------------------
@mcp.tool()
def list_capabilities() -> str:
    """Lists every tool this MCP server provides, which free API each uses, and its known
    limitations. Use this when asked 'what can you do' or 'what tools do you have'."""
    return "\n".join(
        [
            "This MCP server (multi-api-mcp) provides 15 tools, each backed by a free public API:",
            "",
            "1. get_coordinates — look up latitude/longitude for a place name, for use with the weather tools. Source: Open-Meteo Geocoding.",
            "2. get_weather — current weather (temp, humidity, feels-like, UV, sunrise/sunset) for a latitude/longitude. Source: Open-Meteo.",
            "3. get_weather_forecast — hourly forecast for the next 1-16 days. Source: Open-Meteo.",
            "4. get_weather_history — past recorded daily weather for a date range. Source: Open-Meteo Archive.",
            "5. get_exchange_rate — TODAY's currency conversion rate between two currency codes. Source: open.er-api.com.",
            "6. get_exchange_rate_history — daily currency conversion rates over a past date range. Source: Frankfurter (ECB).",
            "7. get_random_joke — a random joke from a fixed category list. Source: JokeAPI.",
            "8. get_random_fact — a random trivia fact. Source: uselessfacts.jsph.pl.",
            "9. get_nasa_apod — NASA's Astronomy Picture of the Day, for today or a specific past date. Source: NASA.",
            "10. get_wikipedia_summary — a short summary of any topic (typo/casing tolerant). Source: Wikipedia.",
            "11. get_github_user — public profile info for an exact GitHub username. Source: GitHub.",
            "12. generate_image — generate an AI image from a text description. Source: Pollinations.ai.",
            "13. create_diagram — generate a real flowchart/UML/mind-map/architecture diagram from text. Source: Kroki.io.",
            "14. search_youtube — search YouTube for videos matching a query. Source: YouTube Data API v3 (requires a free key).",
            "15. search_arxiv — search real academic papers by topic (title, authors, abstract, link). Source: Arxiv.",
            "",
            "Limitations: each tool only answers what its underlying free API supports — no broad/open-ended "
            "questions, no data outside the listed inputs, and some APIs have partial geographic or topical "
            "coverage.",
        ]
    )


class _BearerAuthMiddleware(BaseHTTPMiddleware):
    """Simple shared-secret gate for the streamable-http transport.

    Only active when MCP_AUTH_TOKEN is set. Every incoming request must send
    an "Authorization: Bearer <token>" header matching MCP_AUTH_TOKEN exactly,
    checked with secrets.compare_digest to avoid leaking timing information
    about how much of the token was guessed correctly. Anything missing or
    wrong gets a 401 immediately -- it never reaches any tool.
    """

    async def dispatch(self, request, call_next):
        # The health-check route (added below, used by uptime monitors like
        # UptimeRobot to keep this free-tier service awake) must stay
        # reachable WITHOUT the bearer token -- monitoring services can't be
        # given a secret, and requiring auth on it would make every "keep
        # this awake" ping fail with 401, defeating its entire purpose.
        if request.url.path in ("/", "/health"):
            return await call_next(request)
        auth_header = request.headers.get("authorization", "")
        expected = f"Bearer {MCP_AUTH_TOKEN}"
        if not secrets.compare_digest(auth_header, expected):
            return JSONResponse(
                {"error": "Unauthorized: missing or invalid bearer token"},
                status_code=401,
            )
        return await call_next(request)


if __name__ == "__main__":
    # Two ways this can run, controlled by one environment variable:
    #
    # MCP_TRANSPORT unset or "stdio": local pipe mode (not used for hosting).
    #
    # MCP_TRANSPORT=streamable-http: this file becomes its own independent,
    # always-on network service with a real URL (ending in /mcp), reachable
    # by ANY chatbot/client over HTTP. This is the mode used when hosted on
    # a platform like Render.
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        port = int(os.environ.get("PORT", 8000))
        # Build the same Starlette app FastMCP would run internally, but
        # with the auth middleware layered on top (if enabled), then serve
        # it the same way FastMCP's own run_streamable_http_async does
        # (identical host/port/log_level) -- this keeps behavior identical
        # to before except for the added 401 gate and the health route below.
        app = mcp.streamable_http_app()

        # Plain health-check route -- this server otherwise only exposes
        # /mcp (the real MCP protocol endpoint, which needs a proper MCP
        # client to talk to, not a plain GET). Uptime monitors (UptimeRobot,
        # cron-job.org, etc.) used to keep this free-tier service from
        # spinning down after 15 min idle need a simple GET that returns
        # 200 -- without this, every ping to "/" was a 404, which made the
        # monitor falsely report the service as "Down" even when it was
        # perfectly healthy and running.
        async def _health(request):
            return JSONResponse({"status": "ok", "service": "multi-api-mcp"})

        app.router.routes.insert(0, Route("/", _health, methods=["GET"]))
        app.router.routes.insert(0, Route("/health", _health, methods=["GET"]))

        if MCP_AUTH_TOKEN:
            print(
                f"Starting multi-api-mcp as a standalone HTTP service on port "
                f"{port} (path: /mcp) -- AUTHENTICATION ENABLED (bearer token required)...",
                flush=True,
            )
            app.add_middleware(_BearerAuthMiddleware)
        else:
            print(
                f"Starting multi-api-mcp as a standalone HTTP service on port "
                f"{port} (path: /mcp) -- no authentication (MCP_AUTH_TOKEN not set)...",
                flush=True,
            )
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    else:
        mcp.run(transport="stdio")
