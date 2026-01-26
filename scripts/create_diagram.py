#!/usr/bin/env python3
"""Create enriched workflow diagram via Excalidraw API and save to .excalidraw file."""

import json
from pathlib import Path

import requests

API = "http://localhost:3031/api/elements"
OUTPUT_FILE = Path(__file__).parent.parent / "docs" / "workflow.excalidraw"


def create(el):
    """Create an element on the canvas."""
    r = requests.post(API, json=el)
    return r.json()


def clear_canvas():
    """Clear all elements from canvas."""
    r = requests.get(API)
    for el in r.json().get("elements", []):
        requests.delete(f"{API}/{el['id']}")
    print("Canvas cleared")


def export_to_file():
    """Export canvas elements to .excalidraw file."""
    r = requests.get(API)
    data = r.json()
    elements = data.get("elements", [])
    
    excalidraw_doc = {
        "type": "excalidraw",
        "version": 2,
        "source": "gemini-research-mcp",
        "elements": elements,
        "appState": {
            "viewBackgroundColor": "#ffffff"
        }
    }
    
    OUTPUT_FILE.write_text(json.dumps(excalidraw_doc, indent=2))
    print(f"Exported to {OUTPUT_FILE}")


def main():
    clear_canvas()

    # ============================================================
    # FINAL DESIGN v12 - UX Optimized with Time Expectations
    # ============================================================
    # Fixes from UX analysis:
    # - Wider boxes to prevent text truncation
    # - Removed redundant arrow within Explore box
    # - Removed "Done" (export IS the completion)
    # - Removed verbose "back to Explore" labels
    # - Shorter examples (2-3 words max)
    # - Consistent visual flow indicators
    # ============================================================

    # ============================================================
    # HEADER - Power User style
    # ============================================================
    create({"type": "text", "x": 20, "y": 15, "strokeColor": "#1971c2", 
            "text": "🚀 Gemini Research MCP - Power User Workflow", "fontSize": 20})

    # ============================================================
    # TOP ROW: Main Happy Path
    # [Start] → [Explore] → [Export]
    # ============================================================
    
    # --- START (Green) - wider for text ---
    create({"type": "rectangle", "x": 20, "y": 55, "width": 210, "height": 150,
            "backgroundColor": "#ebfbee", "strokeColor": "#b2f2bb"})
    create({"type": "text", "x": 30, "y": 63, "strokeColor": "#2f9e44", 
            "text": "🚀 Start", "fontSize": 11})

    # research_web with time expectation and example
    create({"type": "rectangle", "x": 30, "y": 83, "width": 190, "height": 45,
            "backgroundColor": "#a5d8ff", "strokeColor": "#1971c2"})
    create({"type": "text", "x": 40, "y": 89, "strokeColor": "#1971c2", 
            "text": "research_web", "fontSize": 10})
    create({"type": "text", "x": 40, "y": 103, "strokeColor": "#868e96", 
            "text": '"What is RAG?"', "fontSize": 7})
    create({"type": "text", "x": 40, "y": 115, "strokeColor": "#ced4da", 
            "text": "⚡ 5-30 sec", "fontSize": 7})

    # "or"
    create({"type": "text", "x": 115, "y": 133, "strokeColor": "#ced4da", "text": "or", "fontSize": 8})

    # research_deep with time expectation, example, and clarification feature
    create({"type": "rectangle", "x": 30, "y": 145, "width": 190, "height": 50,
            "backgroundColor": "#ffc9c9", "strokeColor": "#e03131"})
    create({"type": "text", "x": 40, "y": 149, "strokeColor": "#e03131", 
            "text": "research_deep", "fontSize": 10})
    create({"type": "text", "x": 40, "y": 163, "strokeColor": "#868e96", 
            "text": '"Compare vector DBs"', "fontSize": 7})
    create({"type": "text", "x": 40, "y": 175, "strokeColor": "#ced4da", 
            "text": "🔬 3-20 min", "fontSize": 7})
    create({"type": "text", "x": 120, "y": 175, "strokeColor": "#ced4da", 
            "text": "❓ Clarifies first", "fontSize": 7})

    # Arrow to Explore
    create({"type": "arrow", "x": 230, "y": 130, "width": 25, "height": 0, "strokeColor": "#495057"})

    # --- EXPLORE (Purple) - wider ---
    create({"type": "rectangle", "x": 260, "y": 55, "width": 210, "height": 150,
            "backgroundColor": "#f8f0fc", "strokeColor": "#e599f7"})
    create({"type": "text", "x": 270, "y": 63, "strokeColor": "#9c36b5", 
            "text": "💬 Explore", "fontSize": 11})

    # Results
    create({"type": "rectangle", "x": 270, "y": 83, "width": 190, "height": 45,
            "backgroundColor": "#b2f2bb", "strokeColor": "#2f9e44"})
    create({"type": "text", "x": 280, "y": 89, "strokeColor": "#2f9e44", 
            "text": "✅ Report + citations", "fontSize": 9})
    create({"type": "text", "x": 280, "y": 103, "strokeColor": "#868e96", 
            "text": "Structured findings", "fontSize": 7})
    create({"type": "text", "x": 280, "y": 115, "strokeColor": "#ced4da", 
            "text": "with sources", "fontSize": 7})

    # research_followup
    create({"type": "rectangle", "x": 270, "y": 135, "width": 190, "height": 55,
            "backgroundColor": "#d0bfff", "strokeColor": "#7048e8"})
    create({"type": "text", "x": 280, "y": 141, "strokeColor": "#7048e8", 
            "text": "research_followup ↻", "fontSize": 9})
    create({"type": "text", "x": 280, "y": 155, "strokeColor": "#868e96", 
            "text": '"More on section 3"', "fontSize": 7})
    create({"type": "text", "x": 280, "y": 167, "strokeColor": "#868e96", 
            "text": '"Compare to AWS option"', "fontSize": 7})
    create({"type": "text", "x": 280, "y": 179, "strokeColor": "#ced4da", 
            "text": "Iterate on findings", "fontSize": 7})

    # Arrow to Export
    create({"type": "arrow", "x": 470, "y": 130, "width": 25, "height": 0, "strokeColor": "#495057"})

    # --- SHARE (Pink) - wider ---
    create({"type": "rectangle", "x": 500, "y": 55, "width": 175, "height": 150,
            "backgroundColor": "#fff0f6", "strokeColor": "#fcc2d7"})
    create({"type": "text", "x": 510, "y": 63, "strokeColor": "#c2255c", 
            "text": "📤 Share", "fontSize": 11})

    # export_research_session
    create({"type": "rectangle", "x": 510, "y": 83, "width": 155, "height": 45,
            "backgroundColor": "#ffe8cc", "strokeColor": "#e8590c"})
    create({"type": "text", "x": 520, "y": 89, "strokeColor": "#e8590c", 
            "text": "export_research_session", "fontSize": 8})
    create({"type": "text", "x": 520, "y": 103, "strokeColor": "#868e96", 
            "text": '"Send report"', "fontSize": 7})
    create({"type": "text", "x": 520, "y": 115, "strokeColor": "#ced4da", 
            "text": "Save & share", "fontSize": 7})

    # Formats
    create({"type": "text", "x": 515, "y": 138, "strokeColor": "#495057", 
            "text": "📄 DOCX", "fontSize": 8})
    create({"type": "text", "x": 515, "y": 153, "strokeColor": "#495057", 
            "text": "📝 Markdown", "fontSize": 8})
    create({"type": "text", "x": 515, "y": 168, "strokeColor": "#495057", 
            "text": "🔧 JSON", "fontSize": 8})
    create({"type": "text", "x": 515, "y": 188, "strokeColor": "#2f9e44", "text": "✓ Ready", "fontSize": 8})

    # ============================================================
    # BOTTOM ROW: Recovery Flow
    # [Interruption] → [Auto-save] → [Resume] ↗
    # ============================================================
    
    # --- INTERRUPTION (Orange) ---
    create({"type": "rectangle", "x": 20, "y": 220, "width": 210, "height": 75,
            "backgroundColor": "#fff4e6", "strokeColor": "#ffe8cc"})
    create({"type": "text", "x": 30, "y": 228, "strokeColor": "#e8590c", 
            "text": "⚠️ Interrupted", "fontSize": 10})
    create({"type": "text", "x": 35, "y": 248, "strokeColor": "#c77700", 
            "text": "VS Code closes", "fontSize": 8})
    create({"type": "text", "x": 35, "y": 261, "strokeColor": "#c77700", 
            "text": "Network drops", "fontSize": 8})
    create({"type": "text", "x": 35, "y": 274, "strokeColor": "#c77700", 
            "text": "Laptop sleeps", "fontSize": 8})

    # Arrow to Auto-save
    create({"type": "arrow", "x": 230, "y": 257, "width": 25, "height": 0, "strokeColor": "#495057"})

    # --- SAVED (Yellow) ---
    create({"type": "rectangle", "x": 260, "y": 220, "width": 210, "height": 75,
            "backgroundColor": "#fff9db", "strokeColor": "#ffe066"})
    create({"type": "text", "x": 270, "y": 228, "strokeColor": "#c77700", 
            "text": "💾 Saved", "fontSize": 10})
    create({"type": "text", "x": 275, "y": 248, "strokeColor": "#495057", 
            "text": "Session persisted", "fontSize": 8})
    create({"type": "text", "x": 275, "y": 263, "strokeColor": "#868e96", 
            "text": "Gemini keeps working", "fontSize": 8})
    create({"type": "text", "x": 275, "y": 278, "strokeColor": "#ced4da", 
            "text": "55 days retention", "fontSize": 7})

    # Arrow to Resume
    create({"type": "arrow", "x": 470, "y": 257, "width": 25, "height": 0, "strokeColor": "#495057"})

    # --- RESUME (Blue) - wider ---
    create({"type": "rectangle", "x": 500, "y": 220, "width": 175, "height": 75,
            "backgroundColor": "#e7f5ff", "strokeColor": "#a5d8ff"})
    create({"type": "text", "x": 510, "y": 228, "strokeColor": "#1971c2", 
            "text": "🔄 Resume", "fontSize": 10})

    # resume_research
    create({"type": "rectangle", "x": 510, "y": 248, "width": 155, "height": 38,
            "backgroundColor": "#d0ebff", "strokeColor": "#1971c2"})
    create({"type": "text", "x": 520, "y": 254, "strokeColor": "#1971c2", 
            "text": "resume_research", "fontSize": 9})
    create({"type": "text", "x": 520, "y": 268, "strokeColor": "#868e96", 
            "text": "Recover session", "fontSize": 7})
    create({"type": "text", "x": 600, "y": 268, "strokeColor": "#ced4da", 
            "text": "⚡", "fontSize": 7})

    # Arrow from Resume back to Explore - shows recovery returns to exploration
    # Resume is Row 2 Col 3, Explore is Row 1 Col 2 - diagonal return path
    create({"type": "arrow", "x": 500, "y": 235, "width": -25, "height": -25, "strokeColor": "#495057"})

    # ============================================================
    # BOTTOM: LATER (Cyan) - wider
    # ============================================================
    create({"type": "rectangle", "x": 20, "y": 310, "width": 655, "height": 50,
            "backgroundColor": "#e3fafc", "strokeColor": "#99e9f2"})
    create({"type": "text", "x": 30, "y": 318, "strokeColor": "#0c8599", 
            "text": "📅 Later", "fontSize": 10})

    # Question
    create({"type": "text", "x": 35, "y": 338, "strokeColor": "#495057", 
            "text": '"Find my quantum research"', "fontSize": 8})

    # Arrow
    create({"type": "arrow", "x": 250, "y": 338, "width": 25, "height": 0, "strokeColor": "#495057"})

    # list_research_sessions with dual-path indicators
    create({"type": "rectangle", "x": 280, "y": 323, "width": 170, "height": 32,
            "backgroundColor": "#d0bfff", "strokeColor": "#7048e8"})
    create({"type": "text", "x": 290, "y": 331, "strokeColor": "#7048e8", 
            "text": "list_research_sessions", "fontSize": 8})
    create({"type": "text", "x": 290, "y": 345, "strokeColor": "#868e96", 
            "text": "Browse history", "fontSize": 7})

    # Dual-path arrows from list_sessions
    create({"type": "text", "x": 460, "y": 328, "strokeColor": "#9c36b5", 
            "text": "↑ followup", "fontSize": 8})
    create({"type": "text", "x": 460, "y": 342, "strokeColor": "#c2255c", 
            "text": "↗ export", "fontSize": 8})

    print("Diagram created successfully!")
    
    # Export to .excalidraw file
    export_to_file()


if __name__ == "__main__":
    main()
