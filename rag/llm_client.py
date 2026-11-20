# -*- coding: utf-8 -*-
"""LLM 调用封装（OpenAI 兼容接口，默认走 DeepSeek）。"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")


def call_llm(messages: list[dict], timeout: int = 30) -> str:
    if not API_KEY or not BASE_URL or not MODEL:
        raise ValueError("LLM 配置缺失，请检查 .env 文件")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.1,
    }
    response = requests.post(BASE_URL, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
