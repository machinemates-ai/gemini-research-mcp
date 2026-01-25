#!/usr/bin/env python3
"""Create enriched workflow diagram via Excalidraw API."""

import requests

API = "http://localhost:3031/api/elements"


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


def main():
    clear_canvas()

    # ============================================================
    # FINAL DESIGN v8 - UX Optimized
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
    # HEADER
    # ============================================================
    create({"type": "text", "x": 20, "y": 15, "strokeColor": "#1971c2", 
            "text": "Gemini Research MCP", "fontSize": 18})
    create({"type": "text", "x": 20, "y": 38, "strokeColor": "#868e96",
            "text": "User Journey", "fontSize": 10})

    # ============================================================
    # TOP ROW: Main Happy Path
    # [Start] → [Explore] → [Export]
    # ============================================================
    
    # --- START (Green) - widened ---
    create({"type": "rectangle", "x": 20, "y": 55, "width": 175, "height": 130,
            "backgroundColor": "#ebfbee", "strokeColor": "#b2f2bb"})
    create({"type": "text", "x": 30, "y": 63, "strokeColor": "#2f9e44", 
            "text": "🚀 Start", "fontSize": 11})

    # research_web
    create({"type": "rectangle", "x": 30, "y": 83, "width": 155, "height": 38,
            "backgroundColor": "#a5d8ff", "strokeColor": "#1971c2"})
    create({"type": "text", "x": 40, "y": 91, "strokeColor": "#1971c2", 
            "text": "research_web", "fontSize": 10})
    create({"type": "text", "x": 40, "y": 105, "strokeColor": "#868e96", 
            "text": "⚡ Quick facts", "fontSize": 7})

    # "or"
    create({"type": "text", "x": 95, "y": 125, "strokeColor": "#ced4da", "text": "or", "fontSize": 8})

    # research_deep
    create({"type": "rectangle", "x": 30, "y": 138, "width": 155, "height": 38,
            "backgroundColor": "#ffc9c9", "strokeColor": "#e03131"})
    create({"type": "text", "x": 40, "y": 146, "strokeColor": "#e03131", 
            "text": "research_deep", "fontSize": 10})
    create({"type": "text", "x": 40, "y": 160, "strokeColor": "#868e96", 
            "text": "🔬 Deep analysis", "fontSize": 7})

    # Arrow to Explore
    create({"type": "arrow", "x": 195, "y": 120, "width": 25, "height": 0, "strokeColor": "#495057"})

    # --- EXPLORE (Purple) - widened, NO internal arrow ---
    create({"type": "rectangle", "x": 225, "y": 55, "width": 180, "height": 130,
            "backgroundColor": "#f8f0fc", "strokeColor": "#e599f7"})
    create({"type": "text", "x": 235, "y": 63, "strokeColor": "#9c36b5", 
            "text": "💬 Explore", "fontSize": 11})

    # Results - larger to show it's the main output
    create({"type": "rectangle", "x": 235, "y": 83, "width": 160, "height": 38,
            "backgroundColor": "#b2f2bb", "strokeColor": "#2f9e44"})
    create({"type": "text", "x": 245, "y": 91, "strokeColor": "#2f9e44", 
            "text": "✅ Report + citations", "fontSize": 9})
    create({"type": "text", "x": 245, "y": 105, "strokeColor": "#868e96", 
            "text": "Structured findings", "fontSize": 7})

    # research_followup - with loop symbol only, no extra arrow
    create({"type": "rectangle", "x": 235, "y": 128, "width": 160, "height": 48,
            "backgroundColor": "#d0bfff", "strokeColor": "#7048e8"})
    create({"type": "text", "x": 245, "y": 136, "strokeColor": "#7048e8", 
            "text": "research_followup ↻", "fontSize": 9})
    create({"type": "text", "x": 245, "y": 152, "strokeColor": "#868e96", 
            "text": '"Explain more..."', "fontSize": 7})
    create({"type": "text", "x": 245, "y": 164, "strokeColor": "#ced4da", 
            "text": "Drill deeper", "fontSize": 7})

    # Arrow to Export
    create({"type": "arrow", "x": 405, "y": 110, "width": 25, "height": 0, "strokeColor": "#495057"})

    # --- EXPORT (Pink) - NO "Done" ellipse ---
    create({"type": "rectangle", "x": 435, "y": 55, "width": 145, "height": 130,
            "backgroundColor": "#fff0f6", "strokeColor": "#fcc2d7"})
    create({"type": "text", "x": 445, "y": 63, "strokeColor": "#c2255c", 
            "text": "📤 Share", "fontSize": 11})

    # export_session
    create({"type": "rectangle", "x": 445, "y": 83, "width": 125, "height": 38,
            "backgroundColor": "#ffe8cc", "strokeColor": "#e8590c"})
    create({"type": "text", "x": 455, "y": 91, "strokeColor": "#e8590c", 
            "text": "export_session", "fontSize": 9})
    create({"type": "text", "x": 455, "y": 105, "strokeColor": "#868e96", 
            "text": "Save & share", "fontSize": 7})

    # Formats - cleaner list with completion indicator
    create({"type": "text", "x": 450, "y": 128, "strokeColor": "#495057", 
            "text": "📄 DOCX", "fontSize": 8})
    create({"type": "text", "x": 450, "y": 143, "strokeColor": "#495057", 
            "text": "📝 Markdown", "fontSize": 8})
    create({"type": "text", "x": 450, "y": 158, "strokeColor": "#495057", 
            "text": "🔧 JSON", "fontSize": 8})
    create({"type": "text", "x": 450, "y": 173, "strokeColor": "#2f9e44", "text": "✓ Ready", "fontSize": 8})

    # ============================================================
    # BOTTOM ROW: Recovery Flow
    # [Interruption] → [Auto-save] → [Resume] ↗
    # ============================================================
    
    # --- INTERRUPTION (Orange) ---
    create({"type": "rectangle", "x": 20, "y": 200, "width": 175, "height": 75,
            "backgroundColor": "#fff4e6", "strokeColor": "#ffe8cc"})
    create({"type": "text", "x": 30, "y": 208, "strokeColor": "#e8590c", 
            "text": "⚠️ Interrupted", "fontSize": 10})
    create({"type": "text", "x": 35, "y": 228, "strokeColor": "#c77700", 
            "text": "VS Code closes", "fontSize": 8})
    create({"type": "text", "x": 35, "y": 241, "strokeColor": "#c77700", 
            "text": "Network drops", "fontSize": 8})
    create({"type": "text", "x": 35, "y": 254, "strokeColor": "#c77700", 
            "text": "Laptop sleeps", "fontSize": 8})

    # Arrow to Auto-save
    create({"type": "arrow", "x": 195, "y": 237, "width": 25, "height": 0, "strokeColor": "#495057"})

    # --- AUTO-SAVE (Yellow) - simplified ---
    create({"type": "rectangle", "x": 225, "y": 200, "width": 180, "height": 75,
            "backgroundColor": "#fff9db", "strokeColor": "#ffe066"})
    create({"type": "text", "x": 235, "y": 208, "strokeColor": "#c77700", 
            "text": "💾 Saved", "fontSize": 10})
    create({"type": "text", "x": 240, "y": 228, "strokeColor": "#495057", 
            "text": "Session persisted", "fontSize": 8})
    create({"type": "text", "x": 240, "y": 243, "strokeColor": "#868e96", 
            "text": "Gemini keeps working", "fontSize": 8})
    create({"type": "text", "x": 240, "y": 258, "strokeColor": "#ced4da", 
            "text": "55 days retention", "fontSize": 7})

    # Arrow to Resume
    create({"type": "arrow", "x": 405, "y": 237, "width": 25, "height": 0, "strokeColor": "#495057"})

    # --- RESUME (Blue) ---
    create({"type": "rectangle", "x": 435, "y": 200, "width": 145, "height": 75,
            "backgroundColor": "#e7f5ff", "strokeColor": "#a5d8ff"})
    create({"type": "text", "x": 445, "y": 208, "strokeColor": "#1971c2", 
            "text": "🔄 Resume", "fontSize": 10})

    # resume_research
    create({"type": "rectangle", "x": 445, "y": 228, "width": 125, "height": 38,
            "backgroundColor": "#d0ebff", "strokeColor": "#1971c2"})
    create({"type": "text", "x": 455, "y": 236, "strokeColor": "#1971c2", 
            "text": "resume_research", "fontSize": 9})
    create({"type": "text", "x": 455, "y": 251, "strokeColor": "#868e96", 
            "text": "Recover session", "fontSize": 7})

    # ============================================================
    # BOTTOM: MONTHS LATER (Cyan)
    # ============================================================
    create({"type": "rectangle", "x": 20, "y": 290, "width": 560, "height": 50,
            "backgroundColor": "#e3fafc", "strokeColor": "#99e9f2"})
    create({"type": "text", "x": 30, "y": 298, "strokeColor": "#0c8599", 
            "text": "📅 Later", "fontSize": 10})

    # Question - shorter
    create({"type": "text", "x": 35, "y": 318, "strokeColor": "#495057", 
            "text": '"Find my quantum research"', "fontSize": 8})

    # Arrow
    create({"type": "arrow", "x": 215, "y": 318, "width": 25, "height": 0, "strokeColor": "#495057"})

    # list_sessions
    create({"type": "rectangle", "x": 245, "y": 303, "width": 140, "height": 32,
            "backgroundColor": "#d0bfff", "strokeColor": "#7048e8"})
    create({"type": "text", "x": 255, "y": 311, "strokeColor": "#7048e8", 
            "text": "list_sessions", "fontSize": 9})
    create({"type": "text", "x": 255, "y": 325, "strokeColor": "#868e96", 
            "text": "Browse history", "fontSize": 7})

    print("Diagram created successfully!")


if __name__ == "__main__":
    main()
