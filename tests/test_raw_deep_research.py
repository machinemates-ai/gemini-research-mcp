#!/usr/bin/env python3
"""
Raw test script for Gemini Deep Research API.
Tests streaming + polling for in_progress case without MCP overhead.

Usage:
    cd .github/skills/deep-research/mcp-server
    uv run python test_raw_deep_research.py "your query here"
"""

import asyncio
import os
import sys
import time
from datetime import datetime

from google import genai

# Configuration
DEEP_RESEARCH_AGENT = "deep-research-preview-04-2026"
MAX_POLL_TIME = 3600.0  # 60 minutes
STREAM_POLL_INTERVAL = 10.0  # seconds between polls after stream ends


def log(msg: str) -> None:
    """Print timestamped log message."""
    elapsed = time.time() - START_TIME
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{elapsed:6.1f}s] {msg}")


async def run_deep_research(query: str) -> None:
    """Test Deep Research API with streaming + polling."""
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    
    log("="*70)
    log("🔬 DEEP RESEARCH RAW TEST")
    log(f"   Query: {query[:100]}...")
    log(f"   Agent: {DEEP_RESEARCH_AGENT}")
    log("="*70)
    
    # Counters
    chunk_count = 0
    thought_count = 0
    text_chunks = []
    thinking_summaries = []
    interaction_id = None
    final_status = None
    errors = []
    
    # Start streaming
    log("📡 Starting stream (background=True, stream=True)...")
    
    try:
        # Use create() with stream=True, not create_stream() which doesn't exist in async API
        stream = await client.aio.interactions.create(
            input=query,
            agent=DEEP_RESEARCH_AGENT,
            background=True,
            stream=True,
            agent_config={
                "type": "deep-research",
                "thinking_summaries": "auto",
            },
        )
        
        log("📥 Stream created, iterating events...")
        
        async for event in stream:
            chunk_count += 1
            event_type = getattr(event, 'event_type', type(event).__name__)
            
            # Log raw event type
            log(f"📦 CHUNK #{chunk_count}: {event_type}")
            
            # Handle interaction.start - get interaction_id
            if event_type == "interaction.start":
                if hasattr(event, 'interaction'):
                    interaction_id = getattr(event.interaction, 'id', None)
                    log(f"   interaction_id: {interaction_id}")
                continue
            
            # Handle content.delta - extract thinking summaries and text
            if event_type == "content.delta":
                delta = getattr(event, 'delta', None)
                if delta:
                    delta_type = getattr(delta, 'type', None)
                    if delta_type == 'thought_summary':
                        content = getattr(delta, 'content', None)
                        if content:
                            thought_text = getattr(content, 'text', None)
                            if thought_text:
                                thought_count += 1
                                thinking_summaries.append(thought_text)
                                summary = (
                                    thought_text[:100] + "..."
                                    if len(thought_text) > 100
                                    else thought_text
                                )
                                log(f"   🧠 Thought #{thought_count}: {summary}")
                    elif delta_type == 'text':
                        content = getattr(delta, 'content', None)
                        if content:
                            text_content = getattr(content, 'text', None)
                            if text_content:
                                text_chunks.append(text_content)
                                log(f"   📝 Text: {len(text_content)} chars")
                continue
            
            # Handle interaction.complete - get final status
            if event_type == "interaction.complete":
                if hasattr(event, 'interaction'):
                    interaction = event.interaction
                    final_status = getattr(interaction, 'status', None)
                    log(f"   status: {final_status}")
                    
                    # Try to extract output if present
                    output = getattr(interaction, 'output', None)
                    if output:
                        for item in output:
                            parts = getattr(item, 'parts', [])
                            for part in parts:
                                part_text = getattr(part, 'text', None)
                                if part_text:
                                    text_chunks.append(part_text)
                                    log(f"   📝 Final text: {len(part_text)} chars")
                continue
            
            # Handle error
            if event_type == "error":
                error_obj = getattr(event, 'error', None)
                if error_obj:
                    error_str = str(error_obj)
                    errors.append(error_str)
                    log(f"   ❌ ERROR: {error_str}")
                continue
            
            # Log unknown event types
            log("   (unknown event type)")
        
        log(f"📡 Stream ended after {chunk_count} chunks")
        
    except Exception as e:
        log(f"❌ Stream exception: {e}")
        errors.append(str(e))
    
    # Check if we need to poll
    final_text = "".join(text_chunks)
    
    log("="*70)
    log("📊 STREAM SUMMARY")
    log(f"   Chunks: {chunk_count}")
    log(f"   Thoughts: {thought_count}")
    log(f"   Text length: {len(final_text)} chars")
    log(f"   Status: {final_status}")
    log(f"   Errors: {len(errors)}")
    log("="*70)
    
    # If stream ended with in_progress or no text, poll
    if interaction_id and (final_status == "in_progress" or not final_text):
        log(
            f"⏳ Status is '{final_status}' with {len(final_text)} chars text "
            "- starting polling..."
        )
        
        poll_start = time.time()
        poll_count = 0
        
        while True:
            poll_count += 1
            poll_elapsed = time.time() - poll_start
            
            if poll_elapsed > MAX_POLL_TIME:
                log(f"❌ Polling timeout after {poll_elapsed:.1f}s")
                break
            
            try:
                interaction = await client.aio.interactions.get(interaction_id)
                status = getattr(interaction, 'status', 'unknown')
                
                log(f"🔄 Poll #{poll_count}: status={status}")
                
                if status == "completed":
                    # Debug: show interaction structure
                    log(f"   🔍 Interaction type: {type(interaction)}")
                    interaction_attrs = [
                        attr for attr in dir(interaction) if not attr.startswith('_')
                    ]
                    log(f"   🔍 Interaction attrs: {interaction_attrs}")
                    
                    # Try outputs (plural) - the correct attribute
                    if hasattr(interaction, 'outputs') and interaction.outputs:
                        outputs_len = (
                            len(interaction.outputs)
                            if hasattr(interaction.outputs, '__len__')
                            else 'N/A'
                        )
                        log(
                            f"   🔍 outputs type: {type(interaction.outputs)}, "
                            f"len: {outputs_len}"
                        )
                        for i, item in enumerate(interaction.outputs):
                            item_type = type(item).__name__
                            log(f"   🔍 outputs[{i}] type: {item_type}")
                            
                            # Check for direct .text attribute (TextContent has this!)
                            if hasattr(item, 'text') and item.text:
                                final_text = item.text
                                log(
                                    f"   ✅ Got text from outputs[{i}].text: "
                                    f"{len(final_text)} chars"
                                )
                                log(f"   📜 First 500 chars: {final_text[:500]}...")
                                break  # Found it!
                            
                            # Check for summary (ThoughtContent)
                            if hasattr(item, 'summary') and item.summary:
                                log(
                                    f"   🧠 outputs[{i}].summary (thought): "
                                    f"{item.summary[:100]}..."
                                )
                            
                            # Fallback: check parts (older API?)
                            if hasattr(item, 'parts') and item.parts:
                                for j, part in enumerate(item.parts):
                                    part_type = type(part).__name__
                                    log(
                                        f"   🔍 outputs[{i}].parts[{j}] type: {part_type}"
                                    )
                                    if hasattr(part, 'text') and part.text:
                                        final_text = part.text
                                        log(
                                            "   ✅ Got text from outputs.parts.text: "
                                            f"{len(final_text)} chars"
                                        )
                    else:
                        log("   🔍 No outputs attribute or empty")
                    
                    # Try output (singular) as fallback
                    if hasattr(interaction, 'output') and interaction.output:
                        output_len = (
                            len(interaction.output)
                            if hasattr(interaction.output, '__len__')
                            else 'N/A'
                        )
                        log(
                            f"   🔍 output type: {type(interaction.output)}, "
                            f"len: {output_len}"
                        )
                        for i, item in enumerate(interaction.output):
                            item_attrs = [attr for attr in dir(item) if not attr.startswith('_')]
                            log(
                                f"   🔍 output[{i}] type: {type(item)}, "
                                f"attrs: {item_attrs}"
                            )
                            if hasattr(item, 'parts') and item.parts:
                                for j, part in enumerate(item.parts):
                                    part_attrs = [
                                        attr for attr in dir(part) if not attr.startswith('_')
                                    ]
                                    log(
                                        f"   🔍 output[{i}].parts[{j}] type: {type(part)}, "
                                        f"attrs: {part_attrs}"
                                    )
                                    if hasattr(part, 'text') and part.text:
                                        final_text = part.text
                                        log(
                                            "   ✅ Got text from output.parts.text: "
                                            f"{len(final_text)} chars"
                                        )
                    else:
                        log("   🔍 No output attribute or empty")
                    
                    # Try response attribute
                    if hasattr(interaction, 'response') and interaction.response:
                        log(f"   🔍 response type: {type(interaction.response)}")
                        resp = interaction.response
                        if hasattr(resp, 'candidates') and resp.candidates:
                            for c, cand in enumerate(resp.candidates):
                                candidate_attrs = [
                                    attr for attr in dir(cand) if not attr.startswith('_')
                                ]
                                log(
                                    f"   🔍 response.candidates[{c}] attrs: "
                                    f"{candidate_attrs}"
                                )
                                if hasattr(cand, 'content') and cand.content:
                                    content = cand.content
                                    content_attrs = [
                                        attr for attr in dir(content) if not attr.startswith('_')
                                    ]
                                    log(f"   🔍 candidate.content attrs: {content_attrs}")
                                    if hasattr(content, 'parts') and content.parts:
                                        for p, part in enumerate(content.parts):
                                            part_attrs = [
                                                attr
                                                for attr in dir(part)
                                                if not attr.startswith('_')
                                            ]
                                            log(
                                                f"   🔍 candidate.content.parts[{p}] attrs: "
                                                f"{part_attrs}"
                                            )
                                            if hasattr(part, 'text') and part.text:
                                                final_text = part.text
                                                log(
                                                    "   ✅ Got text from response.candidates."
                                                    f"content.parts.text: {len(final_text)} chars"
                                                )
                    else:
                        log("   🔍 No response attribute or empty")
                    
                    # Try direct text attribute
                    if hasattr(interaction, 'text') and interaction.text:
                        final_text = interaction.text
                        log(f"   ✅ Got text from interaction.text: {len(final_text)} chars")
                    
                    break
                    
                elif status == "failed":
                    error_msg = str(getattr(interaction, 'error', 'Unknown error'))
                    log(f"   ❌ Research failed: {error_msg}")
                    errors.append(error_msg)
                    break
                
                await asyncio.sleep(STREAM_POLL_INTERVAL)
                
            except Exception as e:
                log(f"   ❌ Poll error: {e}")
                errors.append(str(e))
                break
    
    # Final report
    log("="*70)
    log("📋 FINAL REPORT")
    log("="*70)
    
    total_time = time.time() - START_TIME
    log(f"⏱️  Total time: {total_time:.1f}s")
    log(f"📦 Total chunks: {chunk_count}")
    log(f"🧠 Total thoughts: {thought_count}")
    log(f"📝 Final text: {len(final_text)} chars")
    log(f"❌ Errors: {len(errors)}")
    
    if errors:
        log("\n❌ ERRORS:")
        for i, err in enumerate(errors, 1):
            log(f"   {i}. {err}")
    
    if thinking_summaries:
        log("\n🧠 THINKING SUMMARIES:")
        for i, thought in enumerate(thinking_summaries, 1):
            summary = thought[:200] + "..." if len(thought) > 200 else thought
            log(f"   {i}. {summary}")
    
    if final_text:
        log("\n📝 FINAL TEXT:")
        log("-"*70)
        # Print first 2000 chars
        if len(final_text) > 2000:
            print(final_text[:2000])
            log(f"... [truncated, total {len(final_text)} chars]")
        else:
            print(final_text)
        log("-"*70)
    else:
        log("\n⚠️  NO FINAL TEXT RECEIVED")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        query = "What are the main differences between FastMCP and vanilla MCP SDK for Python?"
    else:
        query = " ".join(sys.argv[1:])
    
    START_TIME = time.time()
    asyncio.run(run_deep_research(query))
