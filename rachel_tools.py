"""
rachel_tools.py — OpenAI-format tool definitions for Rachel's MCP server tools.

Import TOOLS wherever a /v1/chat/completions call is made:
    from rachel_tools import TOOLS
    json={"tools": TOOLS, "tool_choice": "auto", ...}

Mirrors the avatar/memory tools registered in server.py.

NOTE — deliberately excluded:
  • speak_text   — spoken output is the chat `content`, relayed to TTS by the
                   orchestrator. Exposing it as a tool made the model "plan to
                   speak" inside its reasoning and end the turn without emitting
                   any content (silent thinks, chars=0).
  • look_at_scene — screenshots are fed inline every turn via /v1/chat/completions,
                   so the model already sees the screen. The tool made it plan a
                   redundant screenshot fetch and stall.
Rachel's tool surface is expressions, animations, and memory only.
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "set_expression",
            "description": "Set Rachel's facial expression on the VTuber avatar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "enum": ["neutral", "thinking", "joy", "anger", "sadness",
                                 "surprise", "concern", "excitement"],
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_expression",
            "description": "Reset Rachel's face to neutral expression.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_animation",
            "description": "Play a one-shot body animation on Rachel's avatar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "animation": {
                        "type": "string",
                        "enum": ["wave", "nod", "shrug", "head_tilt",
                                 "bounce", "dance_1", "dance_2", "clap"],
                    },
                },
                "required": ["animation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_idle_animation",
            "description": "Change Rachel's base idle animation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "animation": {
                        "type": "string",
                        "enum": ["standing", "sitting", "leaning", "relaxed"],
                    },
                },
                "required": ["animation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Search Rachel's autobiographical memory for relevant past events. "
                "Use when someone references a past event, says 'remember when', or when "
                "the current context might connect to something that happened before."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (natural language).",
                    },
                    "emotion": {
                        "type": "string",
                        "description": "Rachel's current emotion for mood-congruent recall.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": (
                "Explicitly save a notable moment in Rachel's memory. "
                "Use when something genuinely notable just happened — a clutch play, a funny "
                "chat moment, a stream milestone. Write in past tense as Rachel would remember it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "First-person memory sentence. e.g. 'Ben clutched a 1v4 and I screamed so loud.'",
                    },
                    "emotion": {
                        "type": "string",
                        "enum": ["joy", "sadness", "anger", "fear", "disgust",
                                 "surprise", "love", "neutral"],
                    },
                    "intensity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": ["text", "emotion", "intensity"],
            },
        },
    },
]