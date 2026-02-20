import os


def env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
