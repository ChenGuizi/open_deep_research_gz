import asyncio
import ast
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from open_deep_research.deep_researcher import deep_researcher_builder


load_dotenv()

st.set_page_config(
    page_title="Open Deep Research",
    page_icon="R",
    layout="wide",
)


@st.cache_resource
def get_graph():
    """Compile one graph per Streamlit process and keep checkpoints in memory."""
    return deep_researcher_builder.compile(checkpointer=MemorySaver())


def new_thread():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.report = None


def get_thread_config(model: str, search_api: str):
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "search_api": search_api,
            "research_model": model,
            "summarization_model": model,
            "compression_model": model,
            "final_report_model": model,
            "allow_clarification": True,
        }
    }


def message_content(message):
    content = getattr(message, "content", message)
    return extract_text(content)


def extract_text(content):
    """Extract displayable text from string or provider content blocks."""
    if isinstance(content, str):
        stripped_content = content.strip()
        if stripped_content.startswith("[") and stripped_content.endswith("]"):
            try:
                content = ast.literal_eval(stripped_content)
            except (SyntaxError, ValueError):
                return normalize_display_text(content)
        else:
            return normalize_display_text(content)

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(normalize_display_text(str(block.get("text", ""))))
            elif isinstance(block, str):
                text_parts.append(normalize_display_text(block))
        if text_parts:
            return "\n\n".join(part for part in text_parts if part)

    return normalize_display_text(str(content))


def normalize_display_text(text: str) -> str:
    """Convert escaped whitespace sequences into characters for Markdown rendering."""
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


async def run_research(graph, prompt: str, thread_config: dict):
    """Run the graph and return its latest state."""
    async for _ in graph.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=thread_config,
        stream_mode="updates",
    ):
        pass
    return graph.get_state(thread_config)


if "thread_id" not in st.session_state:
    new_thread()

st.title("Open Deep Research")
st.caption("Ask a question and get a research report with sources and analysis.")

graph = get_graph()

with st.sidebar:
    st.header("Configuration")
    model = st.text_input(
        "Model",
        value=os.getenv("RESEARCH_MODEL", "google_genai:gemini-3.7-flash"),
        help="The selected model must support tool calling and structured output.",
    )
    search_api = st.selectbox("Search API", ["tavily", "none"], index=0)

    if st.button("New research", use_container_width=True):
        new_thread()
        st.rerun()

    st.divider()
    st.caption("Credentials are loaded from your local .env file.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("is_report"):
            st.markdown(message["content"])
        else:
            st.write(message["content"])

if prompt := st.chat_input("What would you like me to research?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    thread_config = get_thread_config(model, search_api)
    with st.chat_message("assistant"):
        with st.spinner("Researching... This may take a few minutes."):
            try:
                state = asyncio.run(run_research(graph, prompt, thread_config))
                report = state.values.get("final_report")

                if report:
                    report = extract_text(report)
                    st.session_state.report = report
                    st.session_state.messages.append(
                        {"role": "assistant", "content": report, "is_report": True}
                    )
                    st.markdown(report)
                else:
                    assistant_messages = [
                        message
                        for message in state.values.get("messages", [])
                        if getattr(message, "type", "") == "ai"
                    ]
                    response = (
                        message_content(assistant_messages[-1])
                        if assistant_messages
                        else "The workflow ended without producing a response."
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    st.write(response)
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
                st.error(error_message)
