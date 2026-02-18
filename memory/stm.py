# memory/stm.py

from typing import Dict, List, Any


class STM:
    """
    Wrapper around ctx.stm dict.
    turn-level structure:
    messages = [{"query": "...", "response": "..."}]
    """

    def __init__(self, state: Dict[str, Any]):
        if not isinstance(state, dict):
            raise TypeError("STM must wrap dict.")

        self.state = state

        self.state.setdefault("messages", [])
        self.state.setdefault("recent_messages", [])
        self.state.setdefault("summary", [])
        self.state.setdefault("compressed_until", 0)

    # --------------------------
    # Turn Messages
    # --------------------------

    @property
    def messages(self) -> List[Dict]:
        return self.state["messages"]

    @property
    def recent_messages(self) -> List[Dict]:
        return self.state["recent_messages"]

    def append_turn(self, query: str, response: str):
        self.state["messages"].append({
            "query": query,
            "response": response
        })

    def update_recent_messages(self, window_size: int = 3):
        self.state["recent_messages"] = self.state["messages"][-window_size:]

    # --------------------------
    # Summary
    # --------------------------

    @property
    def summary(self) -> List[str]:
        return self.state["summary"]

    def append_summary(self, block: str):
        if block:
            self.state["summary"].append(block)

    def get_summary_text(self) -> str:
        return "\n\n".join(self.state["summary"])

    # --------------------------
    # Compression Logic
    # --------------------------

    @property
    def compressed_until(self) -> int:
        return self.state["compressed_until"]

    @compressed_until.setter
    def compressed_until(self, value: int):
        self.state["compressed_until"] = value

    def need_compress(self, threshold: int = 5) -> bool:
        return len(self.messages) - self.compressed_until >= threshold

    def get_chunk_to_compress(self, threshold: int = 5) -> List[Dict]:
        start = self.compressed_until
        end = start + threshold
        return self.messages[start:end]

    def mark_compressed(self, threshold: int = 5):
        self.compressed_until += threshold
