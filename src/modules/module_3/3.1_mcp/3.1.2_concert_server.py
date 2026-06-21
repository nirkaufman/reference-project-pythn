"""Custom MCP server exposing mocked concert search, booking, a catalog resource, and a prompt.

Run this server on its own before starting `langgraph dev` / Studio, so the
client (3.1.3_concert_client.py) can connect to it over HTTP:

    python src/modules/module_3/3.1_mcp/3.1.2_concert_server.py
"""

import json
import time

from fastmcp import FastMCP

mcp = FastMCP("Concerts")

# Hardcoded concert data
CONCERTS = [
    {
        "id": "1",
        "artist": "Taylor Swift",
        "venue": "Madison Square Garden",
        "date": "2025-06-15",
        "price": 150,
        "available": 50,
    },
    {
        "id": "2",
        "artist": "Coldplay",
        "venue": "Wembley Stadium",
        "date": "2025-07-20",
        "price": 120,
        "available": 200,
    },
    {
        "id": "3",
        "artist": "The Weeknd",
        "venue": "SoFi Stadium",
        "date": "2025-08-10",
        "price": 180,
        "available": 75,
    },
]

# Track bookings (in-memory for demo)
bookings: list[dict] = []


@mcp.tool()
def search_concerts(query: str) -> str:
    """Search for available concerts by artist name or venue."""
    matches = [
        c
        for c in CONCERTS
        if query.lower() in c["artist"].lower() or query.lower() in c["venue"].lower()
    ]
    if not matches:
        artists = ", ".join(c["artist"] for c in CONCERTS)
        return f'No concerts found matching "{query}". Available artists: {artists}'
    lines = [
        f"ID: {c['id']} | {c['artist']} at {c['venue']} on {c['date']} - ${c['price']} ({c['available']} tickets left)"
        for c in matches
    ]
    return f"Found {len(matches)} concert(s):\n" + "\n".join(lines)


@mcp.tool()
def book_tickets(concert_id: str, quantity: int) -> str:
    """Book tickets for a specific concert and return a confirmation code."""
    concert = next((c for c in CONCERTS if c["id"] == concert_id), None)
    if concert is None:
        return f'Concert with ID "{concert_id}" not found. Use search_concerts to find available concerts.'
    if concert["available"] < quantity:
        return f"Sorry, only {concert['available']} tickets available for {concert['artist']}. Requested: {quantity}"

    concert["available"] -= quantity
    confirmation_code = f"CONF-{int(time.time())}"
    bookings.append(
        {
            "concert_id": concert_id,
            "quantity": quantity,
            "confirmation_code": confirmation_code,
        }
    )

    total_cost = concert["price"] * quantity
    return (
        f"Booking confirmed!\n"
        f"Confirmation Code: {confirmation_code}\n"
        f"Event: {concert['artist']} at {concert['venue']}\n"
        f"Date: {concert['date']}\n"
        f"Tickets: {quantity}\n"
        f"Total: ${total_cost}"
    )


@mcp.resource("concerts://catalog")
def concert_catalog() -> str:
    """Return the full catalog of available concerts with all details."""
    return json.dumps(CONCERTS, indent=2)


@mcp.prompt()
def concert_assistant(query: str) -> str:
    """Build a helpful prompt template for concert booking assistance."""
    return (
        "You are a helpful concert booking assistant. Help the user with their request.\n\n"
        "Available actions:\n"
        "- Search for concerts by artist or venue using search_concerts\n"
        "- Book tickets using book_tickets (requires concert ID and quantity)\n"
        "- View the full catalog via the concerts://catalog resource\n\n"
        f"User request: {query}"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
