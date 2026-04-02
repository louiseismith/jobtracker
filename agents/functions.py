"""
agents/functions.py

Multi-agent orchestration helpers, adapted from:
dsai/08_function_calling/functions.py

Changes from original:
- Supports cloud model routing: OpenAI and Ollama Cloud models.
- Model/provider loaded from config/preferences.yaml.
- agent() and agent_run() accept an explicit tool_funcs dict for dispatching
  tool calls — the original used globals() which only finds functions defined
  in this module's namespace.
- df_as_text() unchanged.
"""

import json
import os
import yaml
import pandas as pd
import ollama
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PREFS_PATH = Path(__file__).parent.parent / "config" / "preferences.yaml"


def _load_config() -> dict:
    return yaml.safe_load(PREFS_PATH.read_text())


def get_ollama_client() -> ollama.Client:
    """Return an Ollama Cloud client. Local fallback is intentionally disabled."""
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("OLLAMA_API_KEY is not set for Ollama cloud model usage.")
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )


def get_chat_url_and_headers() -> tuple[str, dict]:
    """
    Legacy helper kept for agents/tools.py direct requests calls.
    Returns (url, headers) for raw requests.post usage.
    """
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("OLLAMA_API_KEY is not set for Ollama cloud model usage.")
    return (
        "https://ollama.com/api/chat",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )


def _get_model() -> str:
    cfg = _load_config()
    return cfg.get("model", "gpt-4o")


def _get_provider() -> str:
    cfg = _load_config()
    return str(cfg.get("provider", "openai")).strip().lower()


def _is_openai_model(model: str) -> bool:
    return model.startswith("gpt-") or model.startswith("o")


def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


def _resolve_provider(model: str, provider: str | None) -> str:
    if provider:
        return provider.strip().lower()
    if _is_openai_model(model):
        return "openai"
    if _is_anthropic_model(model):
        return "anthropic"
    return "ollama"


def _openai_chat(model: str, messages: list, tools: list | None = None) -> dict:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set for selected OpenAI model.")

    client = OpenAI(api_key=api_key)
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    tool_calls = []
    for tc in (msg.tool_calls or []):
        tool_calls.append(
            {
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                }
            }
        )

    return {
        "message": {
            "content": msg.content or "",
            "tool_calls": tool_calls,
        }
    }


def _openai_tools_to_anthropic(tools: list) -> list:
    """Convert OpenAI tool schema format to Anthropic format."""
    result = []
    for tool in tools:
        fn = tool.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _anthropic_chat(model: str, messages: list, tools: list | None = None) -> dict:
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    client = Anthropic(api_key=api_key)

    # Anthropic takes system prompt as a separate parameter, not a message.
    system = None
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            chat_messages.append(msg)

    kwargs = {
        "model": model,
        "max_tokens": 4096,
        "messages": chat_messages,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = _openai_tools_to_anthropic(tools)

    response = client.messages.create(**kwargs)

    # Normalize to the same shape as _openai_chat.
    text_content = ""
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_content += block.text
        elif block.type == "tool_use":
            tool_calls.append({
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input),
                }
            })

    return {
        "message": {
            "content": text_content,
            "tool_calls": tool_calls,
        }
    }


# ---------------------------------------------------------------------------
# Multi-round agent loops (used by cmd_ask)
# ---------------------------------------------------------------------------

def _openai_agent_loop(messages: list, model: str, tools: list, tool_funcs: dict, max_rounds: int) -> str:
    """OpenAI multi-round tool loop. Keeps calling tools until the model returns text."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    msgs = list(messages)

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model, messages=msgs, tools=tools, tool_choice="auto", temperature=0
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        msgs.append(msg)
        for tc in msg.tool_calls:
            fn = (tool_funcs or {}).get(tc.function.name)
            result = fn(**json.loads(tc.function.arguments or "{}")) if fn else f"Unknown tool: {tc.function.name}"
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return "Max tool-call rounds reached without a final answer."


def _anthropic_agent_loop(messages: list, model: str, tools: list, tool_funcs: dict, max_rounds: int) -> str:
    """Anthropic multi-round tool loop using proper tool_use/tool_result blocks."""
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system = None
    chat_msgs = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            chat_msgs.append({"role": msg["role"], "content": msg["content"]})

    anthropic_tools = _openai_tools_to_anthropic(tools)

    for round_num in range(max_rounds):
        kwargs = {
            "model": model, "max_tokens": 4096,
            "messages": chat_msgs, "tools": anthropic_tools,
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text = "".join(b.text for b in response.content if b.type == "text")

        if not tool_use_blocks:
            return text

        # Add assistant turn — convert SDK ContentBlock objects to plain dicts.
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        chat_msgs.append({"role": "assistant", "content": assistant_content})
        tool_results = []
        for block in tool_use_blocks:
            fn = (tool_funcs or {}).get(block.name)
            result = fn(**block.input) if fn else f"Unknown tool: {block.name}"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
            })
        chat_msgs.append({"role": "user", "content": tool_results})

    return "Max tool-call rounds reached without a final answer."


def agent_loop(role: str, task: str, tools: list, tool_funcs: dict,
               model: str | None = None, provider: str | None = None,
               max_rounds: int = 12) -> str:
    """
    Multi-round agent: keeps calling tools until the model returns a plain text response.
    Use this instead of agent_run when the task may require chained tool calls.
    """
    if model is None:
        model = _get_model()
    provider = _resolve_provider(model, provider)

    messages = [
        {"role": "system", "content": role},
        {"role": "user",   "content": task},
    ]

    if provider == "openai":
        return _openai_agent_loop(messages, model, tools, tool_funcs, max_rounds)
    elif provider == "anthropic":
        return _anthropic_agent_loop(messages, model, tools, tool_funcs, max_rounds)
    else:
        # Ollama: fall back to single-round
        return agent(messages=messages, model=model, provider=provider,
                     tools=tools, tool_funcs=tool_funcs)


# ---------------------------------------------------------------------------
# Core agent function (single-round, used by Reporter)
# ---------------------------------------------------------------------------

def agent(messages, model=None, provider=None, output="text", tools=None, tool_funcs=None, all=False):
    """
    Run a single agent, with or without tools.

    Parameters
    ----------
    messages : list
        List of {"role": ..., "content": ...} dicts.
    model : str, optional
        Model name. Defaults to value in config/preferences.yaml.
    provider : str, optional
        "openai" or "ollama". Defaults to config value (or inferred for explicit model overrides).
    output : str
        "text" returns last text response or last tool output.
        "tools" returns the full tool_calls list with outputs attached.
    tools : list, optional
        Ollama tool metadata dicts (the JSON schema definitions).
    tool_funcs : dict, optional
        {function_name: callable} — used to dispatch tool calls.
    all : bool
        If True, return the full raw API response dict.
    """
    if model is None:
        model = _get_model()
        provider = provider or _get_provider()

    provider = _resolve_provider(model, provider)

    if provider == "openai":
        result = _openai_chat(model=model, messages=messages, tools=tools)
    elif provider == "anthropic":
        result = _anthropic_chat(model=model, messages=messages, tools=tools)
    elif provider == "ollama":
        client = get_ollama_client()
        kwargs = {"model": model, "messages": messages, "stream": False}
        if tools:
            kwargs["tools"] = tools
        response = client.chat(**kwargs)
        result = response if isinstance(response, dict) else response.model_dump()
    else:
        raise ValueError(f"Unsupported provider '{provider}'. Use 'openai', 'anthropic', or 'ollama'.")

    if tools is None:
        return result["message"]["content"]

    tool_calls = result.get("message", {}).get("tool_calls") or []
    if tool_calls:
        registry = tool_funcs or {}

        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            func_name = func.get("name", "")
            raw_args  = func.get("arguments", {})
            func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

            callable_fn = registry.get(func_name) or globals().get(func_name)
            if callable_fn:
                tool_call["output"] = callable_fn(**func_args)

        if all:
            return result
        if output == "tools":
            return tool_calls
        # Important: do not return raw tool output as the final user answer.
        # Ask the model to synthesize a final response from gathered tool results.
        tool_output_lines = []
        for tc in tool_calls:
            fn = tc.get("function", {}).get("name", "unknown_tool")
            out = tc.get("output", "")
            tool_output_lines.append(f"{fn} output:\n{out}")

        follow_up_messages = messages + [
            {
                "role": "assistant",
                "content": "I retrieved tool results and will now answer using them.",
            },
            {
                "role": "user",
                "content": (
                    "Tool results:\n\n"
                    + "\n\n".join(tool_output_lines)
                    + "\n\nNow provide the final answer to my original question. "
                    "Do not just repeat raw tables; summarize and explain."
                ),
            },
        ]
        return agent(
            messages=follow_up_messages,
            model=model,
            provider=provider,
            output="text",
            tools=None,
            tool_funcs=tool_funcs,
            all=False,
        )

    # No tool calls — return text content
    return result["message"]["content"]


def agent_run(role, task, tools=None, tool_funcs=None, output="text", model=None, provider=None):
    """
    Convenience wrapper: run an agent with a system prompt (role) and user message (task).

    Parameters
    ----------
    role : str
        System prompt defining the agent's role.
    task : str
        User message / task for the agent.
    tools : list, optional
        Ollama tool metadata dicts.
    tool_funcs : dict, optional
        {function_name: callable} for tool dispatch.
    output : str
        "text" or "tools".
    model : str, optional
        Model name. Defaults to config/preferences.yaml.
    provider : str, optional
        "openai" or "ollama". Defaults to config/preferences.yaml.
    """
    messages = [
        {"role": "system", "content": role},
        {"role": "user",   "content": task},
    ]
    return agent(messages=messages, model=model, provider=provider, output=output,
                 tools=tools, tool_funcs=tool_funcs)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def df_as_text(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a markdown table string."""
    return df.to_markdown(index=False)
