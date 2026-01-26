# Deep-Learner/setup.py
from setuptools import setup, find_packages

setup(
    name="deep-learner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph",
        "chainlit",
        "pymilvus",
        "elasticsearch",
        "openai"
    ],
)