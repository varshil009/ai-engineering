"""Temporary test script to verify the fix."""
import asyncio
import sys
sys.path.insert(0, ".")

from tool_calling_concepts.main import run_agent


async def main():
    result = await run_agent("get latest 1000 values for SGT1 table")
    print(f"\nResponse: {result.get('response', 'N/A')[:200]}")
    print(f"Error: {result.get('error', 'None')}")
    return result


if __name__ == "__main__":
    asyncio.run(main())