from pathlib import Path

from setuptools import find_packages, setup


def load_requirements() -> list[str]:
    req_file = Path(__file__).parent / "requirements.txt"
    requirements: list[str] = []

    for line in req_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirements.append(stripped)

    return requirements


setup(
    name="deep-learner",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=load_requirements(),
)
