"""
Test deep research streaming functionality.

Run with:
    cd .github/skills/deep-research/mcp-server
    uv run pytest tests/test_streaming.py -v -s

Or quick test:
    uv run python tests/test_streaming.py
"""

import asyncio
import os
import sys

import pytest

pytestmark = pytest.mark.e2e

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gemini_research_mcp.deep import deep_research, deep_research_stream
from gemini_research_mcp.types import DeepResearchProgress

# Aliases for test compatibility
deep_research_async = deep_research
deep_research_stream_async = deep_research_stream


async def test_streaming_raw():
    """Test raw streaming that yields progress events."""
    # Skip if no API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not set, skipping streaming test")
        return

    print("\n🧪 Testing raw streaming (deep_research_stream_async)...")
    print("=" * 60)
    
    query = "What is the capital of France? Give a brief answer."
    
    events_received = []
    text_parts = []
    thoughts = []
    
    try:
        async for progress in deep_research_stream_async(query):
            events_received.append(progress.event_type)
            
            if progress.event_type == "start":
                print(f"🚀 Started: {progress.interaction_id}")
            elif progress.event_type == "thought":
                thoughts.append(progress.content)
                print(f"🧠 Thought: {progress.content[:100]}...")
            elif progress.event_type == "text":
                text_parts.append(progress.content or "")
                # Print text as it streams
                print(progress.content, end="", flush=True)
            elif progress.event_type == "complete":
                print("\n✅ Complete!")
            elif progress.event_type == "error":
                print(f"\n❌ Error: {progress.content}")
                
        print("=" * 60)
        print(f"\n📊 Summary:")
        print(f"   Events: {events_received}")
        print(f"   Thoughts: {len(thoughts)}")
        print(f"   Text length: {len(''.join(text_parts))}")
        
        assert "start" in events_received, "Should have start event"
        
    except Exception as e:
        print(f"\n❌ Streaming test failed: {e}")
        raise


async def test_deep_research_async():
    """Test the main deep_research_async function (uses streaming internally)."""
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not set, skipping result test")
        return

    print("\n🧪 Testing deep_research_async (main entry point)...")
    print("=" * 60)
    
    progress_log = []
    
    def log_progress(p: DeepResearchProgress):
        progress_log.append(p.event_type)
        if p.event_type == "thought":
            print(f"🧠 {p.content[:80]}...")
        elif p.event_type == "start":
            print(f"🚀 {p.interaction_id}")
        elif p.event_type == "complete":
            print("✅ Complete")
    
    result = await deep_research_async(
        "What are the main features of Python 3.12? Be brief.",
        on_progress=log_progress,
    )
    
    print("=" * 60)
    print(f"\n📋 Result:")
    print(f"   Text: {result.text[:200]}..." if len(result.text) > 200 else f"   Text: {result.text}")
    print(f"   Thoughts collected: {len(result.thinking_summaries)}")
    print(f"   Interaction ID: {result.interaction_id}")
    print(f"   Progress events: {progress_log}")
    
    assert result.text, "Should have text"
    assert result.interaction_id, "Should have interaction_id"


async def test_real_deep_research():
    """Test a real deep research query that requires multi-step analysis."""
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not set, skipping real research test")
        return

    print("\n🔬 Testing REAL deep research (this will take 3-15 minutes)...")
    print("=" * 60)
    
    query = """
    Compare FastMCP and the standard MCP Python SDK for building Model Context Protocol servers.
    Focus on:
    1. Developer experience and ease of use
    2. Performance considerations
    3. Feature completeness
    Keep the comparison concise but thorough.
    """
    
    thought_count = 0
    
    def log_progress(p: DeepResearchProgress):
        nonlocal thought_count
        if p.event_type == "thought":
            thought_count += 1
            # Truncate for readability
            content = (p.content or "")[:150]
            print(f"🧠 [{thought_count}] {content}...")
        elif p.event_type == "start":
            print(f"🚀 Research started: {p.interaction_id}")
        elif p.event_type == "complete":
            print("✅ Research complete!")
    
    result = await deep_research_async(
        query,
        format_instructions="Use markdown formatting with clear sections. Include a comparison table.",
        on_progress=log_progress,
    )
    
    print("=" * 60)
    print("\n📋 RESEARCH REPORT:")
    print("=" * 60)
    print(result.text)
    print("=" * 60)
    print(f"\n📊 Stats:")
    print(f"   Total thoughts streamed: {thought_count}")
    print(f"   Thoughts captured: {len(result.thinking_summaries)}")
    print(f"   Report length: {len(result.text)} chars")
    print(f"   Interaction ID: {result.interaction_id}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Run real deep research test (takes 3-15 min)")
    args = parser.parse_args()
    
    print("🔬 Deep Research Streaming Test")
    print("================================\n")
    
    if args.real:
        # Run the full deep research test
        asyncio.run(test_real_deep_research())
    else:
        # Run quick streaming test
        asyncio.run(test_streaming_raw())
