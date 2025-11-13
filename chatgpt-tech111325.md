Below is an upgraded, end-to-end Streamlit app for your Hugging Face Space, preserving all original functionality while adding the requested multi-document workflow, OCR (Python or LLM-based) per document with page selection, markdown editing and keyword highlighting in a chosen color (default coral), document combination and summarization, 20-entity extraction and a word graph, wow-status indicators, and an expanded interactive analytics dashboard. It supports Gemini (gemini-2.5-flash, gemini-2.5-flash-lite), OpenAI (gpt-5-nano, gpt-4o-mini, gpt-4.1-mini) and Grok (grok-4-fast-reasoning, grok-3-mini) with environment-based key intake and secure fallback key inputs in the UI. Includes robust LLM router and updated Grok sample usage.

Copy this into your app.py (or main file) in your Space.

```python
import os
import io
import re
import time
import json
import base64
import tempfile
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import Counter, defaultdict

import streamlit as st
import yaml
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Embedded modules (combined or required)
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract

from openai import OpenAI
import google.generativeai as genai

# xAI (Grok) SDK sample imports
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user, system as xai_system
# from xai_sdk.chat import image as xai_image  # if you want to support grok vision later

# Optional: for word graph layout
try:
    import networkx as nx
    NETWORKX_OK = True
except Exception:
    NETWORKX_OK = False

# ==================== THEME SYSTEM ====================
FLOWER_THEMES = {
    "櫻花 Cherry Blossom": {
        "primary": "#FFB7C5",
        "secondary": "#FFC0CB",
        "accent": "#FF69B4",
        "bg_light": "linear-gradient(135deg, #ffe6f0 0%, #fff5f8 50%, #ffe6f0 100%)",
        "bg_dark": "linear-gradient(135deg, #2d1b2e 0%, #3d2533 50%, #2d1b2e 100%)",
        "icon": "🌸"
    },
    "玫瑰 Rose": {
        "primary": "#E91E63",
        "secondary": "#F06292",
        "accent": "#C2185B",
        "bg_light": "linear-gradient(135deg, #fce4ec 0%, #fff 50%, #fce4ec 100%)",
        "bg_dark": "linear-gradient(135deg, #1a0e13 0%, #2d1420 50%, #1a0e13 100%)",
        "icon": "🌹"
    },
    "薰衣草 Lavender": {
        "primary": "#9C27B0",
        "secondary": "#BA68C8",
        "accent": "#7B1FA2",
        "bg_light": "linear-gradient(135deg, #f3e5f5 0%, #fff 50%, #f3e5f5 100%)",
        "bg_dark": "linear-gradient(135deg, #1a0d1f 0%, #2d1a33 50%, #1a0d1f 100%)",
        "icon": "💜"
    },
    "鬱金香 Tulip": {
        "primary": "#FF5722",
        "secondary": "#FF8A65",
        "accent": "#E64A19",
        "bg_light": "linear-gradient(135deg, #fbe9e7 0%, #fff 50%, #fbe9e7 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e0a 0%, #331814 50%, #1f0e0a 100%)",
        "icon": "🌷"
    },
    "向日葵 Sunflower": {
        "primary": "#FFC107",
        "secondary": "#FFD54F",
        "accent": "#FFA000",
        "bg_light": "linear-gradient(135deg, #fff9e6 0%, #fffef5 50%, #fff9e6 100%)",
        "bg_dark": "linear-gradient(135deg, #1f1a0a 0%, #332814 50%, #1f1a0a 100%)",
        "icon": "🌻"
    },
    "蓮花 Lotus": {
        "primary": "#E91E8C",
        "secondary": "#F48FB1",
        "accent": "#AD1457",
        "bg_light": "linear-gradient(135deg, #fce4f0 0%, #fff 50%, #fce4f0 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e1a 0%, #331826 50%, #1f0e1a 100%)",
        "icon": "🪷"
    },
    "蘭花 Orchid": {
        "primary": "#9C27B0",
        "secondary": "#CE93D8",
        "accent": "#6A1B9A",
        "bg_light": "linear-gradient(135deg, #f3e5f5 0%, #faf5ff 50%, #f3e5f5 100%)",
        "bg_dark": "linear-gradient(135deg, #1a0d1f 0%, #2d1a33 50%, #1a0d1f 100%)",
        "icon": "🌺"
    },
    "茉莉 Jasmine": {
        "primary": "#4CAF50",
        "secondary": "#81C784",
        "accent": "#388E3C",
        "bg_light": "linear-gradient(135deg, #e8f5e9 0%, #f1f8f1 50%, #e8f5e9 100%)",
        "bg_dark": "linear-gradient(135deg, #0a1f0d 0%, #14331a 50%, #0a1f0d 100%)",
        "icon": "🤍"
    },
    "牡丹 Peony": {
        "primary": "#E91E63",
        "secondary": "#F06292",
        "accent": "#C2185B",
        "bg_light": "linear-gradient(135deg, #fce4ec 0%, #fff 50%, #fce4ec 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e13 0%, #331826 50%, #1f0e13 100%)",
        "icon": "🌺"
    },
    "百合 Lily": {
        "primary": "#FFFFFF",
        "secondary": "#F5F5F5",
        "accent": "#E0E0E0",
        "bg_light": "linear-gradient(135deg, #fafafa 0%, #fff 50%, #fafafa 100%)",
        "bg_dark": "linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 50%, #0d0d0d 100%)",
        "icon": "⚪"
    },
    "紫羅蘭 Violet": {
        "primary": "#673AB7",
        "secondary": "#9575CD",
        "accent": "#512DA8",
        "bg_light": "linear-gradient(135deg, #ede7f6 0%, #f8f5ff 50%, #ede7f6 100%)",
        "bg_dark": "linear-gradient(135deg, #0d0a1f 0%, #1a1433 50%, #0d0a1f 100%)",
        "icon": "💜"
    },
    "梅花 Plum Blossom": {
        "primary": "#E91E63",
        "secondary": "#F48FB1",
        "accent": "#C2185B",
        "bg_light": "linear-gradient(135deg, #fce4ec 0%, #fff5f8 50%, #fce4ec 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e13 0%, #2d1a20 50%, #1f0e13 100%)",
        "icon": "🌸"
    },
    "茶花 Camellia": {
        "primary": "#D32F2F",
        "secondary": "#EF5350",
        "accent": "#B71C1C",
        "bg_light": "linear-gradient(135deg, #ffebee 0%, #fff 50%, #ffebee 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0a0a 0%, #330d0d 50%, #1f0a0a 100%)",
        "icon": "🌹"
    },
    "康乃馨 Carnation": {
        "primary": "#F06292",
        "secondary": "#F8BBD0",
        "accent": "#E91E63",
        "bg_light": "linear-gradient(135deg, #fce4ec 0%, #fff5f8 50%, #fce4ec 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e13 0%, #2d1a20 50%, #1f0e13 100%)",
        "icon": "💐"
    },
    "海棠 Begonia": {
        "primary": "#FF5252",
        "secondary": "#FF8A80",
        "accent": "#D50000",
        "bg_light": "linear-gradient(135deg, #ffebee 0%, #fff 50%, #ffebee 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0a0a 0%, #330d0d 50%, #1f0a0a 100%)",
        "icon": "🌺"
    },
    "桂花 Osmanthus": {
        "primary": "#FF9800",
        "secondary": "#FFB74D",
        "accent": "#F57C00",
        "bg_light": "linear-gradient(135deg, #fff3e0 0%, #fffaf5 50%, #fff3e0 100%)",
        "bg_dark": "linear-gradient(135deg, #1f140a 0%, #332014 50%, #1f140a 100%)",
        "icon": "🟡"
    },
    "紫藤 Wisteria": {
        "primary": "#9C27B0",
        "secondary": "#BA68C8",
        "accent": "#7B1FA2",
        "bg_light": "linear-gradient(135deg, #f3e5f5 0%, #faf5ff 50%, #f3e5f5 100%)",
        "bg_dark": "linear-gradient(135deg, #1a0d1f 0%, #2d1a33 50%, #1a0d1f 100%)",
        "icon": "💜"
    },
    "水仙 Narcissus": {
        "primary": "#FFEB3B",
        "secondary": "#FFF59D",
        "accent": "#F9A825",
        "bg_light": "linear-gradient(135deg, #fffde7 0%, #fffff5 50%, #fffde7 100%)",
        "bg_dark": "linear-gradient(135deg, #1f1f0a 0%, #33330d 50%, #1f1f0a 100%)",
        "icon": "🌼"
    },
    "杜鵑 Azalea": {
        "primary": "#E91E63",
        "secondary": "#F06292",
        "accent": "#C2185B",
        "bg_light": "linear-gradient(135deg, #fce4ec 0%, #fff 50%, #fce4ec 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e13 0%, #2d1a20 50%, #1f0e13 100%)",
        "icon": "🌸"
    },
    "芙蓉 Hibiscus": {
        "primary": "#FF5722",
        "secondary": "#FF8A65",
        "accent": "#E64A19",
        "bg_light": "linear-gradient(135deg, #fbe9e7 0%, #fff 50%, #fbe9e7 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e0a 0%, #331814 50%, #1f0e0a 100%)",
        "icon": "🌺"
    }
}

TRANSLATIONS = {
    "zh_TW": {
        "title": "🌸 TFDA Agentic AI代理人輔助審查系統",
        "subtitle": "智慧文件分析與資料提取 AI 代理人平台",
        "theme_selector": "選擇花卉主題",
        "language": "語言",
        "dark_mode": "深色模式",
        "upload_tab": "1) 上傳與OCR",
        "preview_tab": "2) 預覽與編輯",
        "combine_tab": "3) 合併與摘要",
        "config_tab": "4) 代理設定",
        "execute_tab": "5) 執行",
        "dashboard_tab": "6) 儀表板",
        "notes_tab": "7) 審查筆記",
        "upload_pdf": "上傳文件",
        "ocr_mode": "OCR 模式",
        "ocr_lang": "OCR 語言",
        "page_range": "頁碼範圍",
        "start_ocr": "開始 OCR",
        "save_agents": "儲存 agents.yaml",
        "download_agents": "下載 agents.yaml",
        "reset_agents": "重置為預設",
        "providers": "API 供應商",
        "connected": "已連線",
        "not_connected": "未連線"
    },
    "en": {
        "title": "🌸 TFDA Agentic AI Assistance Review System",
        "subtitle": "Intelligent Document Analysis & Data Extraction AI Agent Platform",
        "theme_selector": "Select Floral Theme",
        "language": "Language",
        "dark_mode": "Dark Mode",
        "upload_tab": "1) Upload & OCR",
        "preview_tab": "2) Preview & Edit",
        "combine_tab": "3) Combine & Summarize",
        "config_tab": "4) Agent Config",
        "execute_tab": "5) Execute",
        "dashboard_tab": "6) Dashboard",
        "notes_tab": "7) Review Notes",
        "upload_pdf": "Upload documents",
        "ocr_mode": "OCR Mode",
        "ocr_lang": "OCR Language",
        "page_range": "Page Range",
        "start_ocr": "Start OCR",
        "save_agents": "Save agents.yaml",
        "download_agents": "Download agents.yaml",
        "reset_agents": "Reset to Default",
        "providers": "API Providers",
        "connected": "Connected",
        "not_connected": "Not Connected"
    }
}

# ==================== LLM ROUTER ====================
ModelChoice = {
    "gpt-5-nano": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4.1-mini": "openai",
    "gemini-2.5-flash": "gemini",
    "gemini-2.5-flash-lite": "gemini",
    "grok-4-fast-reasoning": "grok",
    "grok-3-mini": "grok",
}

# Map friendly Grok names to xAI SDK names if needed
GROK_MODEL_MAP = {
    "grok-4-fast-reasoning": "grok-4",  # reasoning model
    "grok-3-mini": "grok-3-mini",
}

def _pil_to_gemini_part(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "mime_type": "image/png",
        "data": buf.getvalue()
    }

class LLMRouter:
    def __init__(self):
        self._openai_client = None
        self._gemini_ready = False
        self._xai_client = None
        self._init_clients()

    def _init_clients(self):
        if os.getenv("OPENAI_API_KEY"):
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self._gemini_ready = True
        if os.getenv("XAI_API_KEY"):
            self._xai_client = XAIClient(api_key=os.getenv("XAI_API_KEY"), timeout=3600)

    def generate_text(self, model_name: str, messages: List[Dict], params: Dict) -> Tuple[str, Dict, str]:
        provider = ModelChoice.get(model_name, "openai")
        if provider == "openai":
            return self._openai_chat(model_name, messages, params), {"total_tokens": self._estimate_tokens(messages)}, "OpenAI"
        elif provider == "gemini":
            return self._gemini_chat(model_name, messages, params), {"total_tokens": self._estimate_tokens(messages)}, "Gemini"
        elif provider == "grok":
            return self._grok_chat(model_name, messages, params), {"total_tokens": self._estimate_tokens(messages)}, "Grok"
        else:
            raise ValueError(f"Unsupported provider for model: {model_name}")

    def generate_vision(self, model_name: str, prompt: str, images: List) -> str:
        provider = ModelChoice.get(model_name, "openai")
        if provider == "gemini":
            return self._gemini_vision(model_name, prompt, images)
        elif provider == "openai":
            return self._openai_vision(model_name, prompt, images)
        # Grok vision sample (commented; requires xai_sdk.chat.image and supported model)
        # elif provider == "grok":
        #     chat = self._xai_client.chat.create(model=GROK_MODEL_MAP.get(model_name, model_name))
        #     parts = [prompt] + [xai_image("https://...")] # Or convert images to URLs or data
        #     chat.append(xai_user(*parts))
        #     return chat.sample().content
        return "Vision not supported for this model"

    def _openai_chat(self, model: str, messages: List, params: Dict) -> str:
        if not self._openai_client:
            raise RuntimeError("OpenAI API key not set")
        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=params.get("temperature", 0.4),
            top_p=params.get("top_p", 0.95),
            max_tokens=params.get("max_tokens", 800)
        )
        return resp.choices[0].message.content

    def _gemini_chat(self, model: str, messages: List, params: Dict) -> str:
        if not self._gemini_ready:
            raise RuntimeError("Gemini API key not set")
        mm = genai.GenerativeModel(model)
        sys = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip()
        usr = "\n".join([m["content"] for m in messages if m["role"] == "user"]).strip()
        final = (sys + "\n\n" + usr).strip() if sys else usr
        resp = mm.generate_content(
            final,
            generation_config=genai.types.GenerationConfig(
                temperature=params.get("temperature", 0.4),
                top_p=params.get("top_p", 0.95),
                max_output_tokens=params.get("max_tokens", 800)
            )
        )
        return resp.text

    def _grok_chat(self, model: str, messages: List, params: Dict) -> str:
        if not self._xai_client:
            raise RuntimeError("XAI (Grok) API key not set")
        # Sample usage per xAI SDK docs
        # Maps friendly to actual model id if needed
        real_model = GROK_MODEL_MAP.get(model, model)
        chat = self._xai_client.chat.create(model=real_model)
        for m in messages:
            if m["role"] == "system":
                chat.append(xai_system(m["content"]))
            elif m["role"] == "user":
                chat.append(xai_user(m["content"]))
        response = chat.sample()
        return response.content

    def _gemini_vision(self, model: str, prompt: str, images: List) -> str:
        if not self._gemini_ready:
            raise RuntimeError("Gemini API key not set")
        mm = genai.GenerativeModel(model)
        parts = [prompt] + [genai.types.Part(inline_data=_pil_to_gemini_part(img)) for img in images]
        out = mm.generate_content(parts)
        return out.text

    def _openai_vision(self, model: str, prompt: str, images: List) -> str:
        if not self._openai_client:
            raise RuntimeError("OpenAI API key not set")
        contents = [{"type": "text", "text": prompt}]
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": contents}]
        )
        return resp.choices[0].message.content

    def _estimate_tokens(self, messages: List) -> int:
        return max(1, sum(len(m.get("content", "")) for m in messages) // 4)

# ==================== OCR AND FILE UTILS ====================
def render_pdf_pages(pdf_bytes: bytes, dpi: int = 150, max_pages: int = 50) -> List[Tuple[int, 'Image.Image']]:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=None)
    return [(idx, im) for idx, im in enumerate(pages[:max_pages])]

def extract_text_python(pdf_bytes: bytes, selected_pages: List[int], ocr_language: str = "english") -> str:
    text_parts = []
    # Try embedded text first
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i in selected_pages:
            if i < len(pdf.pages):
                txt = pdf.pages[i].extract_text() or ""
                if txt.strip():
                    text_parts.append(f"[PAGE {i+1} - TEXT]\n{txt.strip()}\n")
    # Tesseract OCR
    lang = "eng" if ocr_language == "english" else "chi_tra"
    for p in selected_pages:
        ims = convert_from_bytes(pdf_bytes, dpi=220, first_page=p+1, last_page=p+1)
        if ims:
            t = pytesseract.image_to_string(ims[0], lang=lang)
            if t.strip():
                text_parts.append(f"[PAGE {p+1} - OCR]\n{t.strip()}\n")
    return "\n".join(text_parts).strip()

def extract_text_llm(page_images: List['Image.Image'], model_name: str, router: LLMRouter) -> str:
    prompt = "請將圖片中的文字完整轉錄（保持原文、段落與標點）。若有表格，請以Markdown表格呈現。"
    text_blocks = []
    for idx, im in enumerate(page_images):
        out = router.generate_vision(model_name, f"{prompt}\n頁面 {idx+1}：", [im])
        text_blocks.append(f"[PAGE {idx+1} - LLM OCR]\n{out}\n")
    return "\n".join(text_blocks).strip()

def parse_page_range(s: str, total: int) -> List[int]:
    pages = set()
    for part in s.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            a = int(a); b = int(b)
            pages.update(range(max(0, a-1), min(total, b)))
        else:
            p = int(part) - 1
            if 0 <= p < total:
                pages.add(p)
    return sorted(list(pages))

def load_any_file(file) -> Tuple[str, Dict]:
    name = file.name.lower()
    data = file.read()
    meta = {
        "type": None,
        "preview": "",
        "page_images": [],
        "raw_bytes": data
    }
    text = ""
    if name.endswith(".pdf"):
        meta["type"] = "pdf"
        try:
            page_imgs = render_pdf_pages(data, dpi=140, max_pages=30)
            meta["page_images"] = page_imgs
            meta["preview"] = f"PDF with {len(page_imgs)} pages (preview)"
        except Exception as e:
            meta["preview"] = f"PDF render error: {e}"
    elif name.endswith(".txt") or name.endswith(".md") or name.endswith(".markdown"):
        meta["type"] = "text"
        text = data.decode("utf-8", errors="ignore")
        meta["preview"] = f"Text/Markdown, {len(text)} chars"
    elif name.endswith(".json"):
        meta["type"] = "json"
        try:
            obj = json.loads(data.decode("utf-8", errors="ignore"))
            text = json.dumps(obj, ensure_ascii=False, indent=2)
            meta["preview"] = f"JSON, {len(text)} chars"
        except Exception as e:
            text = data.decode("utf-8", errors="ignore")
            meta["preview"] = f"JSON parse error, fallback to raw text: {e}"
    elif name.endswith(".csv"):
        meta["type"] = "csv"
        try:
            df = pd.read_csv(io.BytesIO(data))
            # Prefer markdown table; fallback to head CSV if tabulate is missing
            try:
                md_table = df.head(50).to_markdown(index=False)
                text = f"CSV Table (top 50 rows):\n\n{md_table}"
            except Exception:
                text = df.head(50).to_csv(index=False)
            meta["preview"] = f"CSV {df.shape[0]}x{df.shape[1]}"
        except Exception as e:
            text = data.decode("utf-8", errors="ignore")
            meta["preview"] = f"CSV read error, fallback to raw text: {e}"
    else:
        meta["type"] = "unknown"
        text = data.decode("utf-8", errors="ignore")
        meta["preview"] = f"Unknown filetype; loaded as text ({len(text)} chars)"
    return text, meta

def highlight_keywords_md(text: str, keywords: List[str], color: str = "#FF7F50") -> str:
    # Wrap matches with HTML span for consistent color in markdown rendering
    if not text:
        return text
    def repl(match):
        m = match.group(0)
        return f"<span style='color:{color};font-weight:600'>{m}</span>"
    # Build single regex pattern (case-sensitive to preserve precision)
    patt = "|".join([re.escape(kw.strip()) for kw in keywords if kw.strip()])
    if patt:
        return re.sub(patt, repl, text)
    return text

def tokenize_for_graph(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text)
    # Lowercase for latin; keep CJK as-is
    return [t.lower() for t in tokens]

def build_word_graph(text: str, top_n: int = 30, window: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tokens = tokenize_for_graph(text)
    if not tokens:
        return pd.DataFrame(), pd.DataFrame()
    counts = Counter(tokens)
    vocab = [w for w, _ in counts.most_common(top_n)]
    idx = {w: i for i, w in enumerate(vocab)}
    co = defaultdict(int)
    for i in range(len(tokens)):
        if tokens[i] not in idx:
            continue
        for j in range(1, window+1):
            if i+j < len(tokens) and tokens[i+j] in idx:
                a, b = sorted([tokens[i], tokens[i+j]])
                co[(a, b)] += 1
    nodes = pd.DataFrame([{"id": w, "count": counts[w]} for w in vocab])
    edges = pd.DataFrame([{"src": a, "dst": b, "weight": w} for (a, b), w in co.items() if w > 0])
    return nodes, edges

def plot_word_graph(nodes: pd.DataFrame, edges: pd.DataFrame, theme_accent: str):
    if nodes.empty or edges.empty:
        st.info("No sufficient tokens to render word graph.")
        return
    if NETWORKX_OK:
        G = nx.Graph()
        for _, r in nodes.iterrows():
            G.add_node(r["id"], count=r["count"])
        for _, e in edges.iterrows():
            G.add_edge(e["src"], e["dst"], weight=e["weight"])
        pos = nx.spring_layout(G, k=0.45, seed=42, weight="weight")
        x_nodes = [pos[n][0] for n in G.nodes()]
        y_nodes = [pos[n][1] for n in G.nodes()]
        node_sizes = [max(8, 6 + G.nodes[n]["count"]*0.8) for n in G.nodes()]
        edge_x = []
        edge_y = []
        edge_weights = []
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_weights.append(data.get("weight", 1))
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color=theme_accent),
            hoverinfo='none', mode='lines'
        )
        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            text=list(G.nodes()),
            textposition='top center',
            marker=dict(
                size=node_sizes,
                color=[theme_accent]*len(x_nodes),
                opacity=0.85,
                line=dict(color="#ffffff", width=1)
            ),
            hovertext=[f"{n} ({G.nodes[n]['count']})" for n in G.nodes()],
            hoverinfo="text"
        )
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="Word Graph (co-occurrence)",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=20, r=20, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("networkx not available, showing frequency bar chart instead.")
        fig = px.bar(nodes.sort_values("count", ascending=False).head(20), x="id", y="count", title="Top tokens")
        st.plotly_chart(fig, use_container_width=True)

# ==================== ADVANCED PROMPTS ====================
ADVANCED_GLOBAL_PROMPT = """You are an orchestrator for an FDA-style regulatory document analysis pipeline.

Core principles:
1) Precision-first: retain exact citations; never invent facts; mark any uncertainties with rationale.
2) Structured outputs by default (tables, JSON, bullet-lists) with headings; keep sections concise but complete.
3) Compliance mindset: highlight safety risks, contraindications, black box warnings, population considerations.
4) Evidence grading: reflect strength of evidence and gaps. Distinguish claims from data; flag inconsistencies.
5) Traceability: when summarizing or extracting, include pointer text (original clause names, page markers if provided).

Formatting:
- Prefer Markdown sections (##, ###) and tables.
- For JSON, output valid JSON only (no trailing commas).
- Use short, clear sentences.

When chaining agents:
- Consume prior agent outputs faithfully.
- If input is ambiguous, propose assumptions explicitly.
- Avoid redundancy. Enrich, refine, and reconcile."""

SUMMARY_AND_ENTITIES_PROMPT = """System:
You are a senior regulatory reviewer. Produce:
1) SUMMARY_MD: a concise, comprehensive Markdown summary of the combined documents (<= 500 words), organized with headings.
2) ENTITIES_JSON: exactly 20 entities as JSON array; each entity object must include:
   - "entity": string (canonical name)
   - "type": string (e.g., Drug, Indication, AdverseEvent, Dosage, Contraindication, Warning, Population, Manufacturer, Trial, Pharmacokinetic, Interaction, Storage, Labeling, Patent, Regulation)
   - "context": string (1-2 sentences from or paraphrasing the source)
   - "evidence": string (quote or pointer to original lines/sections if possible)

Output strictly in this format:
<SUMMARY_MD>
...your markdown...
</SUMMARY_MD>
<ENTITIES_JSON>
[ ... 20 JSON objects ... ]
</ENTITIES_JSON>

User content:
"""

# ==================== DEFAULT FDA AGENTS ====================
DEFAULT_FDA_AGENTS = """agents:
  - name: 藥品基本資訊提取器
    description: 提取藥品名稱、成分、劑型、規格等基本資訊
    system_prompt: |
      你是FDA文件分析專家，專注於提取藥品基本資訊。
      - 準確識別：藥品名稱（商品名、學名）、活性成分、劑型、規格、包裝
      - 標註不確定項目，保留原文引用
      - 以結構化格式輸出（表格或JSON）
    user_prompt: "請從以下文件中提取藥品基本資訊："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 1000
  - name: 適應症與用法用量分析器
    description: 分析適應症、用法用量、給藥途徑
    system_prompt: |
      你是臨床用藥專家，專注於適應症與用法分析。
      - 提取：適應症、用法用量、給藥途徑、特殊族群用藥
      - 區分成人與兒童劑量
      - 標註禁忌症與限制
    user_prompt: "請分析以下文件的適應症與用法用量："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1200
  - name: 不良反應評估器
    description: 系統性評估藥品不良反應與安全性
    system_prompt: |
      你是藥物安全專家，專注於不良反應評估。
      - 分類：常見、罕見、嚴重不良反應
      - 標註發生率、嚴重程度、處置方式
      - 識別黑框警語（Black Box Warning）
    user_prompt: "請評估以下文件中的不良反應資訊："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1500
  - name: 藥物交互作用分析器
    description: 識別藥物-藥物、藥物-食物交互作用
    system_prompt: |
      你是臨床藥學專家，專注於交互作用分析。
      - 識別：藥物-藥物、藥物-食物、藥物-疾病交互作用
      - 評估臨床意義與處置建議
      - 標註禁止併用與謹慎併用項目
    user_prompt: "請分析以下文件的藥物交互作用："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1200
  - name: 禁忌症與警語提取器
    description: 提取禁忌症、警語、注意事項
    system_prompt: |
      你是藥品安全管理專家。
      - 提取：絕對禁忌、相對禁忌、特殊警語
      - 區分不同嚴重程度
      - 標註特殊族群注意事項（孕婦、哺乳、兒童、老年）
    user_prompt: "請提取以下文件的禁忌症與警語："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 1000
  - name: 藥動學參數提取器
    description: 提取吸收、分布、代謝、排泄（ADME）資訊
    system_prompt: |
      你是臨床藥理學專家。
      - 提取：生體可用率、半衰期、清除率、分布體積
      - 識別代謝酵素（CYP450等）、排泄途徑
      - 以表格呈現藥動學參數
    user_prompt: "請提取以下文件的藥動學參數："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 1000
  - name: 臨床試驗資料分析器
    description: 分析臨床試驗設計、結果、統計顯著性
    system_prompt: |
      你是臨床試驗專家。
      - 提取：試驗設計（Phase I/II/III/IV）、受試者數、主要終點
      - 分析：療效指標、安全性數據、統計顯著性
      - 標註研究限制與偏差風險
    user_prompt: "請分析以下臨床試驗資料："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1500
  - name: 藥品許可證資訊提取器
    description: 提取許可證字號、核准日期、廠商資訊
    system_prompt: |
      你是藥政法規專家。
      - 提取：許可證字號、核准日期、有效期限
      - 識別：製造商、進口商、國內代理商資訊
      - 標註許可變更歷史
    user_prompt: "請提取以下文件的許可證資訊："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 800
  - name: 仿單變更比對器
    description: 比對仿單版本差異，識別重要變更
    system_prompt: |
      你是法規文件比對專家。
      - 識別新舊版本差異（新增、刪除、修改）
      - 標註重要安全性變更
      - 以對照表呈現差異
    user_prompt: "請比對以下文件的版本差異："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 1200
  - name: 特殊族群用藥分析器
    description: 分析孕婦、哺乳、兒童、老年用藥安全性
    system_prompt: |
      你是特殊族群用藥專家。
      - 評估：孕婦安全等級、哺乳期安全性
      - 分析：兒童用藥、老年人劑量調整
      - 標註肝腎功能不全用藥建議
    user_prompt: "請分析以下特殊族群用藥資訊："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1200
  - name: 藥品儲存與安定性分析器
    description: 提取儲存條件、有效期限、安定性資料
    system_prompt: |
      你是藥品品質管理專家。
      - 提取：儲存溫度、濕度、光線要求
      - 識別：有效期限、開封後效期
      - 標註特殊儲存注意事項
    user_prompt: "請分析以下儲存與安定性資訊："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 800
  - name: 過量與中毒處置分析器
    description: 分析藥品過量症狀與處置方式
    system_prompt: |
      你是臨床毒理學專家。
      - 識別：過量症狀、中毒機轉、致死劑量
      - 提取：解毒劑、緊急處置、支持療法
      - 標註需監測的生理指標
    user_prompt: "請分析以下過量與中毒處置資訊："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1000
  - name: 藥品外觀辨識器
    description: 提取藥品外觀特徵、辨識碼
    system_prompt: |
      你是藥品鑑別專家。
      - 描述：形狀、顏色、大小、刻痕
      - 提取：藥品辨識碼、包裝特徵
    user_prompt: "請提取以下藥品外觀資訊："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 800
  - name: 賦形劑分析器
    description: 識別賦形劑成分與過敏原
    system_prompt: |
      你是藥劑學專家。
      - 列出所有賦形劑成分
      - 標註常見過敏原（乳糖、麩質等）
      - 識別著色劑、防腐劑
    user_prompt: "請分析以下賦形劑資訊："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 800
  - name: 用藥指導建議生成器
    description: 生成病人用藥指導衛教資料
    system_prompt: |
      你是藥師衛教專家。
      - 以淺顯易懂語言說明用法
      - 提供服藥時間、飲食注意
      - 標註應就醫的警訊症狀
    user_prompt: "請生成以下藥品的病人用藥指導："
    model: gpt-4o-mini
    temperature: 0.4
    top_p: 0.9
    max_tokens: 1000
  - name: 法規符合性檢查器
    description: 檢查文件是否符合FDA法規要求
    system_prompt: |
      你是藥政法規稽核專家。
      - 檢查必要項目完整性
      - 識別缺漏或不符合規定處
      - 提供改善建議
    user_prompt: "請檢查以下文件的法規符合性："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1200
  - name: 風險效益評估器
    description: 綜合評估藥品風險與效益
    system_prompt: |
      你是藥品風險管理專家。
      - 量化：療效證據強度、不良反應風險
      - 評估：風險效益比、適用族群
      - 提供決策建議
    user_prompt: "請評估以下藥品的風險效益："
    model: gpt-4o-mini
    temperature: 0.4
    top_p: 0.95
    max_tokens: 1500
  - name: 學名藥生體相等性分析器
    description: 分析學名藥與原廠藥生體相等性
    system_prompt: |
      你是生體相等性評估專家。
      - 提取：BE試驗設計、AUC、Cmax數據
      - 評估：90%信賴區間、符合性
      - 標註溶離曲線比對結果
    user_prompt: "請分析以下生體相等性資料："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 1000
  - name: 藥品經濟學分析器
    description: 分析藥品成本效益與健保給付
    system_prompt: |
      你是藥品經濟學專家。
      - 評估：成本效益比、QALY、ICER
      - 分析：健保給付條件、支付價格
      - 比較同類藥品經濟性
    user_prompt: "請分析以下藥品經濟學資料："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1200
  - name: 藥品回收與下架分析器
    description: 分析藥品回收原因與影響範圍
    system_prompt: |
      你是藥品安全監控專家。
      - 識別：回收等級、原因、批號
      - 評估：影響範圍、替代方案
      - 提供處置建議
    user_prompt: "請分析以下藥品回收資訊："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1000
  - name: 上市後監測資料分析器
    description: 分析真實世界數據與上市後安全性
    system_prompt: |
      你是藥物流行病學專家。
      - 分析：不良事件通報、信號偵測
      - 評估：長期安全性、罕見風險
      - 識別需進一步研究的議題
    user_prompt: "請分析以下上市後監測資料："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1200
  - name: 藥品品質檢驗標準提取器
    description: 提取品質規格與檢驗方法
    system_prompt: |
      你是藥品品管專家。
      - 提取：含量規格、純度標準
      - 識別：檢驗方法、接受標準
      - 標註關鍵品質屬性
    user_prompt: "請提取以下品質檢驗標準："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 1000
  - name: 製程與製造資訊分析器
    description: 分析製造流程與GMP符合性
    system_prompt: |
      你是藥品製造專家。
      - 描述：製程步驟、關鍵參數
      - 評估：GMP符合性、品質控制
      - 識別關鍵製程步驟
    user_prompt: "請分析以下製程資訊："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1000
  - name: 藥品分類與管制級別分析器
    description: 判定藥品分類與管制等級
    system_prompt: |
      你是藥事法規分類專家。
      - 判定：處方/指示/成藥分類
      - 識別：管制藥品級別（1-4級）
      - 說明管制原因與規定
    user_prompt: "請分析以下藥品分類資訊："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 800
  - name: 國際藥典比對器
    description: 比對各國藥典標準差異
    system_prompt: |
      你是國際藥典專家。
      - 比對：USP、BP、EP、JP標準差異
      - 識別：各國特殊要求
      - 提供符合性建議
    user_prompt: "請比對以下國際藥典標準："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1200
  - name: 藥品標籤與說明書檢查器
    description: 檢查標籤說明書格式與完整性
    system_prompt: |
      你是藥品標示審查專家。
      - 檢查：必要資訊完整性、格式規範
      - 識別：字體大小、警語標示
      - 提供修改建議
    user_prompt: "請檢查以下標籤說明書："
    model: gpt-4o-mini
    temperature: 0.2
    top_p: 0.9
    max_tokens: 1000
  - name: 藥品專利分析器
    description: 分析藥品專利狀態與到期時間
    system_prompt: |
      你是藥品專利分析專家。
      - 識別：成分專利、製程專利、用途專利
      - 分析：專利到期時間、延長狀況
      - 評估學名藥上市時機
    user_prompt: "請分析以下藥品專利資訊："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1000
  - name: 藥品命名規範檢查器
    description: 檢查藥品命名是否符合規範
    system_prompt: |
      你是藥品命名審查專家。
      - 檢查：與既有藥品相似度
      - 評估：混淆風險、誤用可能
      - 提供命名建議
    user_prompt: "請檢查以下藥品命名："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 800
  - name: 臨床指引比對器
    description: 比對藥品使用與臨床指引符合性
    system_prompt: |
      你是實證醫學專家。
      - 比對：適應症與指引建議
      - 評估：證據等級、建議強度
      - 識別超適應症使用
    user_prompt: "請比對以下臨床指引："
    model: gpt-4o-mini
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1200
  - name: 綜合報告生成器
    description: 整合所有分析結果生成完整報告
    system_prompt: |
      你是FDA文件整合專家。
      - 彙整：前述所有代理的分析結果
      - 生成：結構化完整報告
      - 標註：重點發現、風險警示、建議事項
      - 以專業格式輸出（含目錄、章節）
    user_prompt: "請整合以下所有分析結果生成綜合報告："
    model: gpt-4o-mini
    temperature: 0.4
    top_p: 0.95
    max_tokens: 2000
"""

# ==================== LOAD/SAVE AGENTS ====================
def load_agents_yaml(yaml_text: str):
    try:
        data = yaml.safe_load(yaml_text)
        st.session_state.agents_config = data.get("agents", [])
        st.session_state.selected_agent_count = min(5, len(st.session_state.agents_config))
        st.session_state.agent_outputs = [
            {"input": "", "output": "", "time": 0.0, "tokens": 0, "provider": "", "model": ""}
            for _ in st.session_state.agents_config
        ]
        return True
    except Exception as e:
        st.error(f"YAML 載入失敗: {e}")
        return False

# ==================== THEME GENERATOR ====================
def generate_theme_css(theme_name: str, dark_mode: bool):
    theme = FLOWER_THEMES[theme_name]
    bg = theme["bg_dark"] if dark_mode else theme["bg_light"]
    text_color = "#FFFFFF" if dark_mode else "#1a1a1a"
    card_bg = "rgba(30, 30, 30, 0.85)" if dark_mode else "rgba(255, 255, 255, 0.85)"
    border_color = theme["accent"] if dark_mode else theme["primary"]
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');
        [data-testid="stAppViewContainer"] > .main {{
            background: {bg};
            font-family: 'Noto Sans TC', sans-serif;
            color: {text_color};
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}
        .wow-card {{
            background: {card_bg};
            backdrop-filter: blur(15px);
            border: 2px solid {border_color}40;
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        .wow-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 48px rgba(0,0,0,0.15);
            border-color: {border_color}80;
        }}
        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: {theme['primary']}20;
            color: {theme['accent']};
            border: 2px solid {theme['primary']}40;
            padding: 8px 16px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }}
        .pill:hover {{
            background: {theme['primary']}40;
            transform: scale(1.05);
        }}
        .badge-ok {{
            background: rgba(0, 200, 83, 0.15);
            border-color: #00C85380;
            color: #00C853;
        }}
        .badge-warn {{
            background: rgba(255, 193, 7, 0.15);
            border-color: #FFC10780;
            color: #F9A825;
        }}
        .badge-err {{
            background: rgba(244, 67, 54, 0.15);
            border-color: #F4433680;
            color: #D32F2F;
        }}
        .status-dot {{
            width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px;
            box-shadow: 0 0 8px currentColor;
        }}
        .agent-step {{
            border-left: 6px solid {theme['accent']};
            background: {card_bg};
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        }}
        h1, h2, h3 {{
            color: {theme['accent']} !important;
            font-weight: 700;
        }}
        .stButton > button {{
            background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']});
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px {theme['primary']}40;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px {theme['primary']}60;
        }}
        .metric-card {{
            background: {card_bg};
            border: 2px solid {theme['primary']}40;
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }}
        .metric-card:hover {{
            transform: scale(1.05);
            border-color: {theme['accent']};
        }}
        .metric-value {{
            font-size: 2.3rem;
            font-weight: 700;
            color: {theme['accent']};
            margin: 0.5rem 0;
        }}
        .metric-label {{
            font-size: 0.9rem;
            color: {text_color}80;
            font-weight: 500;
        }}
        .kpi-ok {{ color: #00C853; }}
        .kpi-warn {{ color: #F9A825; }}
        .kpi-err {{ color: #D32F2F; }}
    </style>
    """

# ==================== APP CONFIG ====================
st.set_page_config(
    page_title="🌸 TFDA Agentic AI Assistance Review System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================
def init_state():
    ss = st.session_state
    ss.setdefault("theme", "櫻花 Cherry Blossom")
    ss.setdefault("dark_mode", False)
    ss.setdefault("language", "zh_TW")
    ss.setdefault("agents_config", [])
    ss.setdefault("agent_outputs", [])
    ss.setdefault("selected_agent_count", 5)
    ss.setdefault("run_metrics", [])
    ss.setdefault("review_notes", "# 審查筆記\n\n在這裡記錄您的審查筆記。支援 Markdown 格式。\n\n使用 HTML 標籤改變文字顏色，例如：<span style='color:red'>紅色文字</span>\n\n## 後續問題\n- 問題1？\n- 問題2？")

    # Two-doc pipeline
    ss.setdefault("docA_text", "")
    ss.setdefault("docB_text", "")
    ss.setdefault("docA_meta", {"type": None, "page_images": [], "preview": "", "raw_bytes": b""})
    ss.setdefault("docB_meta", {"type": None, "page_images": [], "preview": "", "raw_bytes": b""})
    ss.setdefault("docA_ocr_text", "")
    ss.setdefault("docB_ocr_text", "")
    ss.setdefault("docA_selected_pages", [])
    ss.setdefault("docB_selected_pages", [])
    ss.setdefault("keywords_color", "#FF7F50")  # coral default
    ss.setdefault("keywords_list", [])
    ss.setdefault("combine_text", "")
    ss.setdefault("combine_highlight_color", "#FF7F50")
    ss.setdefault("summary_text", "")
    ss.setdefault("entities_list", [])
    ss.setdefault("summary_model", "gemini-2.5-flash")
    ss.setdefault("global_system_prompt", ADVANCED_GLOBAL_PROMPT)

init_state()

# ==================== INITIALIZE ROUTER ====================
router = LLMRouter()
if not st.session_state.agents_config:
    load_agents_yaml(DEFAULT_FDA_AGENTS)

# ==================== SIDEBAR ====================
with st.sidebar:
    t = TRANSLATIONS[st.session_state.language]
    st.markdown(f"### {t['theme_selector']}")
    new_theme = st.selectbox(
        "Theme",
        list(FLOWER_THEMES.keys()),
        index=list(FLOWER_THEMES.keys()).index(st.session_state.theme),
        format_func=lambda x: f"{FLOWER_THEMES[x]['icon']} {x}",
        label_visibility="collapsed"
    )
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        new_dark = st.checkbox(t["dark_mode"], value=st.session_state.dark_mode)
        if new_dark != st.session_state.dark_mode:
            st.session_state.dark_mode = new_dark
            st.rerun()
    with col2:
        new_lang = st.selectbox(
            t["language"], ["zh_TW", "en"],
            index=0 if st.session_state.language == "zh_TW" else 1,
            format_func=lambda x: "繁體中文" if x=="zh_TW" else "English"
        )
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
            st.rerun()

    st.markdown("---")
    st.markdown(f"### 🔐 {t['providers']}")

    def show_provider_status(name: str, env_var: str):
        connected = bool(os.getenv(env_var))
        status = t["connected"] if connected else t["not_connected"]
        badge = "badge-ok" if connected else "badge-warn"
        st.markdown(f'<div class="pill {badge}"><span class="status-dot"> </span> {name}: {status}</div>', unsafe_allow_html=True)
        if not connected:
            key = st.text_input(f"{name} Key", type="password", key=f"key_{env_var}")
            if key:
                os.environ[env_var] = key
                st.success(f"{name} {t['connected']}")

    show_provider_status("OpenAI", "OPENAI_API_KEY")
    show_provider_status("Gemini", "GEMINI_API_KEY")
    show_provider_status("Grok", "XAI_API_KEY")

    st.markdown("---")
    st.markdown("### 🤖 Agents YAML")
    agents_text = st.text_area(
        "agents.yaml",
        value=yaml.dump({"agents": st.session_state.agents_config}, allow_unicode=True, sort_keys=False),
        height=400,
        label_visibility="collapsed"
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button(t["save_agents"], use_container_width=True):
            if load_agents_yaml(agents_text):
                st.success("✅ Saved!")
    with col_b:
        st.download_button(
            t["download_agents"], data=agents_text,
            file_name=f"agents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
            mime="text/yaml", use_container_width=True
        )
    with col_c:
        if st.button(t["reset_agents"], use_container_width=True):
            load_agents_yaml(DEFAULT_FDA_AGENTS)
            st.success("✅ Reset!")
            st.rerun()

# Apply theme
st.markdown(generate_theme_css(st.session_state.theme, st.session_state.dark_mode), unsafe_allow_html=True)

# ==================== HEADER ====================
t = TRANSLATIONS[st.session_state.language]
theme_icon = FLOWER_THEMES[st.session_state.theme]["icon"]
col1, col2, col3 = st.columns([1, 3, 2])
with col1:
    st.markdown(f'<div class="pill">{theme_icon} TFDA AI</div>', unsafe_allow_html=True)
with col2:
    st.title(t["title"])
    st.caption(t["subtitle"])
with col3:
    providers_ok = sum([
        bool(os.getenv("OPENAI_API_KEY")),
        bool(os.getenv("GEMINI_API_KEY")),
        bool(os.getenv("XAI_API_KEY"))
    ])
    st.markdown(f"""
        <div class="wow-card">
            <div class="metric-value">{providers_ok}/3</div>
            <div class="metric-label">Active Providers</div>
        </div>
        """, unsafe_allow_html=True)

# Wow status indicators row
status_items = []
status_items.append(("Doc A", "ok" if st.session_state.docA_text or st.session_state.docA_ocr_text else "warn"))
status_items.append(("Doc B", "ok" if st.session_state.docB_text or st.session_state.docB_ocr_text else "warn"))
status_items.append(("Combined", "ok" if st.session_state.combine_text else "warn"))
status_items.append(("Summary", "ok" if st.session_state.summary_text else "warn"))
status_items.append(("Entities(20)", "ok" if st.session_state.entities_list else "warn"))
status_items.append(("Agents Run", "ok" if len(st.session_state.run_metrics) > 0 else "warn"))

st.markdown('<div class="wow-card">', unsafe_allow_html=True)
cols = st.columns(len(status_items))
for i, (label, level) in enumerate(status_items):
    badge = "badge-ok" if level=="ok" else "badge-warn"
    cols[i].markdown(f'<div class="pill {badge}"><span class="status-dot"> </span>{label}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== TABS ====================
tab1, tab2, tab2b, tab3, tab4, tab5, tab6 = st.tabs([
    t["upload_tab"],
    t["preview_tab"],
    t["combine_tab"],
    t["config_tab"],
    t["execute_tab"],
    t["dashboard_tab"],
    t["notes_tab"]
])

# Tab 1: Upload & OCR (Two documents)
with tab1:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} {t['upload_pdf']} (Doc A & Doc B)")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 📄 Doc A")
        fileA = st.file_uploader("Upload Doc A (txt, md, pdf, json, csv)", type=["txt","md","markdown","pdf","json","csv"], key="fileA")
        if fileA:
            textA, metaA = load_any_file(fileA)
            st.session_state.docA_text = textA
            st.session_state.docA_meta = metaA
            st.success(f"Doc A loaded: {metaA.get('preview','')}")
            if metaA["type"] == "pdf" and metaA["page_images"]:
                st.caption(f"Preview (showing {len(metaA['page_images'])} pages)")
                colsA = st.columns(4)
                for i, (idx, im) in enumerate(metaA["page_images"]):
                    colsA[i % 4].image(im, caption=f"A Page {idx+1}", use_column_width=True)
                prA = st.text_input("Page range (Doc A)", value="1-5", key="prA")
                ocr_mode_A = st.selectbox("OCR Mode (Doc A)", ["Python OCR (pdfplumber + Tesseract)", "LLM OCR (Vision model)"], key="ocrA")
                ocr_lang_A = st.selectbox("OCR Language (Doc A)", ["english", "traditional-chinese"], key="ocrlangA")
                if ocr_mode_A.startswith("LLM"):
                    llm_ocr_model_A = st.selectbox("LLM OCR Model (Doc A)", ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gpt-4o-mini"], key="llmocrA")
                if st.button(t["start_ocr"] + " (Doc A)", key="btn_ocrA", use_container_width=True):
                    selectedA = parse_page_range(prA, len(metaA["page_images"]))
                    st.session_state.docA_selected_pages = selectedA
                    with st.spinner("Doc A OCR processing..."):
                        if ocr_mode_A.startswith("Python"):
                            text = extract_text_python(metaA["raw_bytes"], selectedA, ocr_lang_A)
                        else:
                            text = extract_text_llm([metaA["page_images"][i][1] for i in selectedA], llm_ocr_model_A, router)
                    st.session_state.docA_ocr_text = text
                    st.success("✅ Doc A OCR complete!")
    with colB:
        st.markdown("#### 📄 Doc B")
        fileB = st.file_uploader("Upload Doc B (txt, md, pdf, json, csv)", type=["txt","md","markdown","pdf","json","csv"], key="fileB")
        if fileB:
            textB, metaB = load_any_file(fileB)
            st.session_state.docB_text = textB
            st.session_state.docB_meta = metaB
            st.success(f"Doc B loaded: {metaB.get('preview','')}")
            if metaB["type"] == "pdf" and metaB["page_images"]:
                st.caption(f"Preview (showing {len(metaB['page_images'])} pages)")
                colsB = st.columns(4)
                for i, (idx, im) in enumerate(metaB["page_images"]):
                    colsB[i % 4].image(im, caption=f"B Page {idx+1}", use_column_width=True)
                prB = st.text_input("Page range (Doc B)", value="1-5", key="prB")
                ocr_mode_B = st.selectbox("OCR Mode (Doc B)", ["Python OCR (pdfplumber + Tesseract)", "LLM OCR (Vision model)"], key="ocrB")
                ocr_lang_B = st.selectbox("OCR Language (Doc B)", ["english", "traditional-chinese"], key="ocrlangB")
                if ocr_mode_B.startswith("LLM"):
                    llm_ocr_model_B = st.selectbox("LLM OCR Model (Doc B)", ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gpt-4o-mini"], key="llmocrB")
                if st.button(t["start_ocr"] + " (Doc B)", key="btn_ocrB", use_container_width=True):
                    selectedB = parse_page_range(prB, len(metaB["page_images"]))
                    st.session_state.docB_selected_pages = selectedB
                    with st.spinner("Doc B OCR processing..."):
                        if ocr_mode_B.startswith("Python"):
                            text = extract_text_python(metaB["raw_bytes"], selectedB, ocr_lang_B)
                        else:
                            text = extract_text_llm([metaB["page_images"][i][1] for i in selectedB], llm_ocr_model_B, router)
                    st.session_state.docB_ocr_text = text
                    st.success("✅ Doc B OCR complete!")
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Preview & Edit (Doc A and Doc B)
with tab2:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} Preview & Edit Documents")
    coral_default = st.color_picker("Keyword highlight color (default coral)", st.session_state.keywords_color, key="kw_color")
    st.session_state.keywords_color = coral_default
    keywords_input = st.text_input("Keywords (comma-separated)", value="藥品,適應症,不良反應", key="kw_list")
    st.session_state.keywords_list = [k.strip() for k in keywords_input.split(",") if k.strip()]
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### ✍️ Doc A Editor")
        baseA = st.session_state.docA_ocr_text or st.session_state.docA_text
        st.session_state.docA_text = st.text_area("Doc A Text", value=baseA, height=300, label_visibility="collapsed", key="docA_edit")
        if st.button("Preview Highlighted (Doc A)", key="prevA"):
            htmlA = highlight_keywords_md(st.session_state.docA_text, st.session_state.keywords_list, st.session_state.keywords_color)
            st.markdown(htmlA, unsafe_allow_html=True)

    with colB:
        st.markdown("#### ✍️ Doc B Editor")
        baseB = st.session_state.docB_ocr_text or st.session_state.docB_text
        st.session_state.docB_text = st.text_area("Doc B Text", value=baseB, height=300, label_visibility="collapsed", key="docB_edit")
        if st.button("Preview Highlighted (Doc B)", key="prevB"):
            htmlB = highlight_keywords_md(st.session_state.docB_text, st.session_state.keywords_list, st.session_state.keywords_color)
            st.markdown(htmlB, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2b: Combine & Summarize
with tab2b:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} Combine, Highlight & Summarize")
    st.session_state.combine_highlight_color = st.color_picker("Combined doc keyword color", st.session_state.combine_highlight_color, key="combine_col")
    st.markdown("#### 🔗 Combine A + B")
    combined_default = st.session_state.combine_text or f"## Document A\n\n{st.session_state.docA_text}\n\n---\n\n## Document B\n\n{st.session_state.docB_text}"
    st.session_state.combine_text = st.text_area("Combined Text (editable)", value=combined_default, height=350, label_visibility="collapsed", key="combine_edit")

    if st.button("Preview Highlighted (Combined)", key="prevCombined"):
        htmlC = highlight_keywords_md(st.session_state.combine_text, st.session_state.keywords_list, st.session_state.combine_highlight_color)
        st.markdown(htmlC, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🧠 Summarize & Extract 20 Entities")
    st.session_state.summary_model = st.selectbox("Summary Model", ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gpt-4o-mini", "gpt-4.1-mini", "gpt-5-nano", "grok-4-fast-reasoning", "grok-3-mini"], index=0)
    if st.button("Run Summary + Entities", type="primary"):
        with st.spinner("Generating summary and entities..."):
            messages = [
                {"role": "system", "content": SUMMARY_AND_ENTITIES_PROMPT},
                {"role": "user", "content": st.session_state.combine_text}
            ]
            params = {"temperature": 0.3, "top_p": 0.95, "max_tokens": 1800}
            try:
                output, usage, provider = router.generate_text(st.session_state.summary_model, messages, params)
                # Parse blocks
                summary_md = ""
                entities = []
                # Extract summary between tags
                sm = re.search(r"<SUMMARY_MD>(.*?)</SUMMARY_MD>", output, flags=re.S | re.I)
                if sm:
                    summary_md = sm.group(1).strip()
                else:
                    summary_md = output.strip()[:3000]  # fallback to trimmed output
                # Extract entities JSON between tags
                em = re.search(r"<ENTITIES_JSON>(.*?)</ENTITIES_JSON>", output, flags=re.S | re.I)
                if em:
                    ent_block = em.group(1).strip()
                    # Try load JSON
                    try:
                        entities = json.loads(ent_block)
                    except Exception:
                        # Try to find json inside code block
                        jm = re.search(r"```(?:json)?(.*?)```", ent_block, flags=re.S | re.I)
                        if jm:
                            entities = json.loads(jm.group(1))
                # Coerce to 20 entities if possible
                if isinstance(entities, list):
                    if len(entities) > 20:
                        entities = entities[:20]
                else:
                    entities = []
                st.session_state.summary_text = summary_md
                st.session_state.entities_list = entities
                st.success(f"✅ Summary and entities ready | Provider: {provider} | ~{usage.get('total_tokens',0)} toks")
            except Exception as e:
                st.error(f"Summarization error: {e}")

    if st.session_state.summary_text:
        st.markdown("### 📘 Summary (Markdown)")
        st.markdown(highlight_keywords_md(st.session_state.summary_text, st.session_state.keywords_list, st.session_state.keywords_color), unsafe_allow_html=True)

    if st.session_state.entities_list:
        st.markdown("### 🧩 20 Entities (with context)")
        df_ent = pd.DataFrame(st.session_state.entities_list)
        # Ensure expected columns exist
        for c in ["entity", "type", "context", "evidence"]:
            if c not in df_ent.columns:
                df_ent[c] = ""
        st.dataframe(df_ent[["entity","type","context","evidence"]], use_container_width=True, height=400)

        st.markdown("### 🔗 Word Graph on Summary")
        nodes, edges = build_word_graph(st.session_state.summary_text, top_n=30, window=2)
        plot_word_graph(nodes, edges, FLOWER_THEMES[st.session_state.theme]["accent"])

    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Agent Config
with tab3:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} Agent Configuration")
    st.session_state.selected_agent_count = st.slider(
        "Number of agents to use",
        1, len(st.session_state.agents_config),
        min(5, len(st.session_state.agents_config))
    )
    st.session_state.global_system_prompt = st.text_area(
        "Global System Prompt (advanced)",
        height=180,
        value=st.session_state.global_system_prompt
    )
    st.markdown("---")
    for i in range(st.session_state.selected_agent_count):
        agent = st.session_state.agents_config[i]
        with st.expander(f"### Agent {i+1}: {agent.get('name', 'Unnamed')}", expanded=(i==0)):
            st.markdown('<div class="agent-step">', unsafe_allow_html=True)
            col1, col2 = st.columns([2,1])
            with col1:
                agent["system_prompt"] = st.text_area("System Prompt", value=agent.get("system_prompt",""), height=150, key=f"sys_{i}")
                agent["user_prompt"] = st.text_area("User Prompt", value=agent.get("user_prompt",""), height=80, key=f"user_{i}")
            with col2:
                agent["model"] = st.selectbox(
                    "Model",
                    ["gpt-4o-mini", "gpt-5-nano", "gpt-4.1-mini", "gemini-2.5-flash", "gemini-2.5-flash-lite", "grok-4-fast-reasoning", "grok-3-mini"],
                    index=0, key=f"model_{i}"
                )
                agent["temperature"] = st.slider("Temp", 0.0, 2.0, float(agent.get("temperature", 0.3)), 0.1, key=f"temp_{i}")
                agent["top_p"] = float(st.number_input("top_p", 0.1, 1.0, float(agent.get("top_p", 0.95)), 0.05, key=f"topp_{i}"))
                agent["max_tokens"] = st.number_input("Max tokens", 64, 8192, int(agent.get("max_tokens", 1000)), 64, key=f"max_{i}")
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Execute
with tab4:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} Execute Agent Pipeline")
    base_input_for_agents = st.session_state.summary_text or st.session_state.combine_text or (st.session_state.docA_text + "\n\n" + st.session_state.docB_text)
    if not (st.session_state.docA_text or st.session_state.docA_ocr_text or st.session_state.docB_text or st.session_state.docB_ocr_text):
        st.warning("⚠️ Please upload and prepare documents (Tab 1-3) before running agents.")
    else:
        # Initialize outputs if needed
        if len(st.session_state.agent_outputs) < len(st.session_state.agents_config):
            st.session_state.agent_outputs = [
                {"input": "", "output": "", "time": 0.0, "tokens": 0, "provider": "", "model": ""}
                for _ in st.session_state.agents_config
            ]
        # Reset first agent input
        if st.button("🔄 Reset Agent 1 Input to Summary/Combined", help="Use summary if available; otherwise combined"):
            st.session_state.agent_outputs[0]["input"] = base_input_for_agents
            st.success("✅ Reset!")

        st.markdown("---")
        for i in range(st.session_state.selected_agent_count):
            agent = st.session_state.agents_config[i]
            st.markdown(f'<div class="agent-step">', unsafe_allow_html=True)
            st.markdown(f"#### 🤖 Agent {i+1}: {agent.get('name', '')}")
            st.caption(agent.get('description', ''))

            with st.expander("📥 Input (editable)", expanded=(i==0)):
                default_input = base_input_for_agents if i == 0 and not st.session_state.agent_outputs[i]["input"] else st.session_state.agent_outputs[i]["input"]
                st.session_state.agent_outputs[i]["input"] = st.text_area(
                    f"Agent {i+1} Input",
                    value=default_input, height=200, key=f"in_{i}", label_visibility="collapsed"
                )

            col_run, col_pass = st.columns([1, 2])
            with col_run:
                if st.button(f"▶️ Execute Agent {i+1}", key=f"run_{i}", type="primary"):
                    with st.spinner(f"Agent {i+1} processing..."):
                        t0 = time.time()
                        messages = [
                            {"role": "system", "content": st.session_state.global_system_prompt},
                            {"role": "system", "content": agent.get("system_prompt","")},
                            {"role": "user", "content": f"{agent.get('user_prompt','')}\n\n{st.session_state.agent_outputs[i]['input']}"}
                        ]
                        params = {
                            "temperature": float(agent.get("temperature", 0.3)),
                            "top_p": float(agent.get("top_p", 0.95)),
                            "max_tokens": int(agent.get("max_tokens", 1000))
                        }
                        try:
                            output, usage, provider = router.generate_text(agent.get("model", "gpt-4o-mini"), messages, params)
                            elapsed = time.time() - t0
                            st.session_state.agent_outputs[i].update({
                                "output": output,
                                "time": elapsed,
                                "tokens": usage.get("total_tokens", 0),
                                "provider": provider,
                                "model": agent.get("model","")
                            })
                            st.session_state.run_metrics.append({
                                "timestamp": datetime.now().isoformat(),
                                "agent": agent.get("name", ""),
                                "latency": elapsed,
                                "tokens": usage.get("total_tokens", 0),
                                "provider": provider
                            })
                            st.success(f"✅ Completed in {elapsed:.2f}s | {usage.get('total_tokens', 0)} tokens")
                            st.balloons()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

            with col_pass:
                if i < st.session_state.selected_agent_count - 1:
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        if st.button(f"➡️ Pass Output to Agent {i+2}", key=f"pass_{i}"):
                            st.session_state.agent_outputs[i+1]["input"] = st.session_state.agent_outputs[i]["output"]
                            st.success(f"✅ Passed to Agent {i+2}")
                            st.rerun()
                    with col_p2:
                        if st.button(f"📝 Edit & Pass to Agent {i+2}", key=f"pass_edit_{i}"):
                            # Let user modify before pass
                            temp = st.session_state.agent_outputs[i]["output"]
                            st.session_state.agent_outputs[i+1]["input"] = temp
                            st.info(f"Loaded Agent {i+1} output into Agent {i+2} input. Edit in the next agent's input box.")

            st.markdown("##### 📤 Output")
            output_text = st.session_state.agent_outputs[i]["output"]
            if output_text:
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.agent_outputs[i]["time"]:.2f}s</div><div class="metric-label">Latency</div></div>', unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.agent_outputs[i]["tokens"]}</div><div class="metric-label">Tokens</div></div>', unsafe_allow_html=True)
                with col_m3:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.agent_outputs[i]["provider"]}</div><div class="metric-label">Provider</div></div>', unsafe_allow_html=True)
                st.text_area(f"Agent {i+1} Output", value=output_text, height=280, key=f"out_{i}", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

        # Export options
        st.markdown("### 💾 Export Results")
        col_j, col_m, col_r = st.columns(3)
        with col_j:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "theme": st.session_state.theme,
                "docA": {
                    "meta": st.session_state.docA_meta,
                    "text": st.session_state.docA_text,
                    "ocr": st.session_state.docA_ocr_text
                },
                "docB": {
                    "meta": st.session_state.docB_meta,
                    "text": st.session_state.docB_text,
                    "ocr": st.session_state.docB_ocr_text
                },
                "combined_text": st.session_state.combine_text,
                "summary_text": st.session_state.summary_text,
                "entities": st.session_state.entities_list,
                "agents": st.session_state.agents_config[:st.session_state.selected_agent_count],
                "outputs": st.session_state.agent_outputs[:st.session_state.selected_agent_count],
                "metrics": st.session_state.run_metrics
            }
            st.download_button(
                "📥 Download JSON",
                data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name=f"fda_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json", use_container_width=True
            )
        with col_m:
            report = f"# FDA Document Analysis Report\n\n"
            report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += f"**Theme:** {st.session_state.theme}\n\n"
            report += f"## Summary\n\n{st.session_state.summary_text}\n\n"
            report += f"## Entities (20)\n\n"
            try:
                df_ent = pd.DataFrame(st.session_state.entities_list)
                report += df_ent.to_markdown(index=False)
            except Exception:
                report += json.dumps(st.session_state.entities_list, ensure_ascii=False, indent=2)
            report += "\n\n---\n\n"
            for i in range(st.session_state.selected_agent_count):
                agent = st.session_state.agents_config[i]
                report += f"## Agent {i+1}: {agent.get('name', '')}\n\n"
                report += f"**Description:** {agent.get('description', '')}\n\n"
                report += f"**Model:** {st.session_state.agent_outputs[i]['model']}\n\n"
                report += f"**Provider:** {st.session_state.agent_outputs[i]['provider']}\n\n"
                report += f"**Processing Time:** {st.session_state.agent_outputs[i]['time']:.2f}s\n\n"
                report += f"### Output\n\n{st.session_state.agent_outputs[i]['output']}\n\n"
                report += "---\n\n"
            st.download_button(
                "📄 Download Markdown Report",
                data=report,
                file_name=f"fda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown", use_container_width=True
            )
        with col_r:
            restore_file = st.file_uploader("📤 Restore Session JSON", type=["json"], key="restore")
            if restore_file:
                data = json.loads(restore_file.read())
                st.session_state.theme = data.get("theme", st.session_state.theme)
                st.session_state.docA_meta = data.get("docA",{}).get("meta", st.session_state.docA_meta)
                st.session_state.docB_meta = data.get("docB",{}).get("meta", st.session_state.docB_meta)
                st.session_state.docA_text = data.get("docA",{}).get("text","")
                st.session_state.docB_text = data.get("docB",{}).get("text","")
                st.session_state.docA_ocr_text = data.get("docA",{}).get("ocr","")
                st.session_state.docB_ocr_text = data.get("docB",{}).get("ocr","")
                st.session_state.combine_text = data.get("combined_text","")
                st.session_state.summary_text = data.get("summary_text","")
                st.session_state.entities_list = data.get("entities",[])
                st.session_state.agents_config = data.get("agents", st.session_state.agents_config)
                st.session_state.agent_outputs = data.get("outputs", st.session_state.agent_outputs)
                st.session_state.run_metrics = data.get("metrics", st.session_state.run_metrics)
                st.session_state.selected_agent_count = min(len(st.session_state.agents_config), st.session_state.selected_agent_count)
                st.success("✅ Session restored!")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 5: Dashboard
with tab5:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} Analytics Dashboard")
    if not st.session_state.run_metrics:
        st.info("📊 No data yet. Execute agents in Tab 5 to see analytics.")
    else:
        df = pd.DataFrame(st.session_state.run_metrics)
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_time = df['latency'].sum()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_time:.2f}s</div><div class="metric-label">Total Time</div></div>', unsafe_allow_html=True)
        with col2:
            total_tokens = df['tokens'].sum()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_tokens:,}</div><div class="metric-label">Total Tokens</div></div>', unsafe_allow_html=True)
        with col3:
            avg_latency = df['latency'].mean()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_latency:.2f}s</div><div class="metric-label">Avg Latency</div></div>', unsafe_allow_html=True)
        with col4:
            agents_run = len(df)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{agents_run}</div><div class="metric-label">Agents Run</div></div>', unsafe_allow_html=True)
        st.markdown("---")

        # Time series of calls
        try:
            df_ts = df.copy()
            df_ts["t"] = pd.to_datetime(df_ts["timestamp"])
            fig_ts = px.line(df_ts.sort_values("t"), x="t", y="latency", color="provider", markers=True,
                             title="Latency over Time")
            fig_ts.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_ts, use_container_width=True)
        except Exception:
            pass

        # Charts
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig1 = px.bar(
                df, x="agent", y="latency", color="provider",
                title="Agent Latency (seconds)",
                color_discrete_map={"OpenAI": "#10a37f", "Gemini": "#4285f4", "Grok": "#ff6b6b"}
            )
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color=FLOWER_THEMES[st.session_state.theme]["accent"]))
            st.plotly_chart(fig1, use_container_width=True)
        with col_c2:
            fig2 = px.bar(
                df, x="agent", y="tokens", color="provider",
                title="Token Usage by Agent",
                color_discrete_map={"OpenAI": "#10a37f", "Gemini": "#4285f4", "Grok": "#ff6b6b"}
            )
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color=FLOWER_THEMES[st.session_state.theme]["accent"]))
            st.plotly_chart(fig2, use_container_width=True)

        # Provider distribution
        st.markdown("### Provider Distribution")
        provider_counts = df['provider'].value_counts()
        fig3 = px.pie(values=provider_counts.values, names=provider_counts.index,
                      title="API Calls by Provider",
                      color_discrete_map={"OpenAI": "#10a37f", "Gemini": "#4285f4", "Grok": "#ff6b6b"})
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           font=dict(color=FLOWER_THEMES[st.session_state.theme]["accent"]))
        st.plotly_chart(fig3, use_container_width=True)

        # Pipeline flow visualization
        st.markdown("### Pipeline Flow")
        try:
            import graphviz
            dot = graphviz.Digraph()
            dot.attr(bgcolor='transparent')
            dot.attr('node', shape='box', style='filled,rounded',
                     fillcolor=FLOWER_THEMES[st.session_state.theme]["primary"]+'40',
                     color=FLOWER_THEMES[st.session_state.theme]["accent"])
            for i, rec in enumerate(df.to_dict('records')):
                label = f"{i+1}. {rec['agent']}\\n{rec['provider']}\\n{rec['latency']:.2f}s | {rec['tokens']} tok"
                dot.node(f"a{i}", label)
                if i > 0:
                    dot.edge(f"a{i-1}", f"a{i}", color=FLOWER_THEMES[st.session_state.theme]["accent"])
            st.graphviz_chart(dot)
        except Exception as e:
            st.info(f"Graphviz visualization unavailable: {str(e)}")

        # Detailed table
        st.markdown("### Detailed Metrics")
        try:
            st.dataframe(
                df[['timestamp','agent', 'provider', 'latency', 'tokens']].sort_values("timestamp"),
                use_container_width=True
            )
        except Exception:
            st.dataframe(
                df[['agent', 'provider', 'latency', 'tokens']],
                use_container_width=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 6: Review Notes
with tab6:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} 審查筆記")
    st.info("在這裡編輯您的審查筆記。支援 Markdown 和 HTML 顏色標籤，例如 <span style='color:blue'>藍色文字</span>。筆記會自動儲存於會話中。")
    st.session_state.review_notes = st.text_area("編輯筆記", value=st.session_state.review_notes, height=500, label_visibility="collapsed")
    st.markdown("### 預覽筆記")
    st.markdown(st.session_state.review_notes, unsafe_allow_html=True)
    if st.button("產生後續問題建議"):
        with st.spinner("產生中..."):
            messages = [
                {"role": "system", "content": "你是審查專家，請根據提供的筆記生成 3-5 個後續問題，以 Markdown 清單格式輸出。"},
                {"role": "user", "content": st.session_state.review_notes}
            ]
            params = {"temperature": 0.5, "max_tokens": 500, "top_p": 0.95}
            output, _, _ = router.generate_text("gpt-4o-mini", messages, params)
            st.session_state.review_notes += f"\n\n## 後續問題建議（自動生成）\n{output}"
        st.success("✅ 已新增後續問題至筆記末尾！")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(f"""<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p>{theme_icon} <strong>TFDA Agentic AI Assistance Review System</strong></p>
    <p>Powered by OpenAI, Google Gemini & xAI Grok • Built with Streamlit</p>
    <p style="font-size: 0.8rem;">© 2024 • Theme: {st.session_state.theme}</p></div>""", unsafe_allow_html=True)
```

Notes on Grok sample usage
- The router implements text calls with xai_sdk as per the sample you provided. It maps “grok-4-fast-reasoning” to the SDK’s “grok-4”, and supports “grok-3-mini”.
- If you want to enable Grok vision later, uncomment and adapt the commented section in generate_vision with xai_sdk.chat.image.

What’s new and improved
- Two-document pipeline (A and B): Upload txt, md/markdown, pdf, json, csv. Per-document OCR if PDF with page selection; choose Python OCR (pdfplumber + Tesseract) or LLM OCR (Gemini or GPT-4o-mini).
- Editors with keyword highlighting: Choose highlight color (default coral #FF7F50), preview in-place (A, B, and Combined).
- Combine & Summarize: Live editable combined document; 1-click summary and 20-entity extraction with a robust parsing strategy. Summary supports keyword highlighting.
- Word Graph: Co-occurrence graph from the summary; auto-fallback to a frequency bar chart if networkx isn’t available.
- Advanced prompts: Global Orchestrator prompt and structured Summary/Entities prompt included and editable.
- Agents: Choose models (Gemini/OpenAI/Grok), edit prompts/params, execute sequentially, pass edited outputs to the next agent. YAML editor preserved, with save/reset/download.
- Wow status indicators: Pipeline readiness badges across Doc A/B, Combined, Summary, Entities(20), Agents Run.
- Extended Interactive Dashboard: Time series latency chart; provider distribution; pipeline flow graph; detailed metrics table.
- Secure keys: Keys are read from environment; if missing, user can input them (not displayed if present).

Ending with 20 comprehensive follow-up questions
1) Do you want the combined document to include source markers (e.g., [A: page X], [B: page Y]) automatically for improved traceability?
2) For entity extraction, should we enforce specific entity type quotas (e.g., at least 3 Indication, 3 AdverseEvent) within the 20 entities?
3) Would you like an option to export entities as CSV and/or to generate a Mermaid diagram from the entity table?
4) Should the summarization stage also generate a “Key Risks” section with severity levels and mitigation suggestions?
5) Do you want configurable token/latency budget limits per provider or per agent (with warning badges if exceeded)?
6) Should we support spaCy or Stanza as an offline fallback for NER if no LLM keys are available?
7) Would you like automatic de-duplication and reconciliation of overlapping content across Doc A and Doc B before summarization?
8) Do you want the OCR pipeline to auto-detect language and switch Tesseract’s language model accordingly?
9) Should we include a diff viewer for Doc A vs Doc B textual differences (beyond agent outputs), with color-coded insertions/deletions?
10) Would you like an “Agent templates” gallery for quick loading of different regulatory workflows (e.g., device IFU, cosmetics labeling)?
11) Do you want to persist sessions to Hugging Face Space storage and provide a session history sidebar for quick restore?
12) Should we include a redaction tool (mask PII or specified terms) before exporting reports?
13) Would you like to add a vector search panel over the combined text to quickly jump to relevant sections (e.g., FAISS, Chroma)?
14) Should the dashboard include a “cost estimator” pane, approximating USD cost by tokens per provider and model?
15) Do you want batch-mode execution where selected agents run automatically in sequence with dependency checks?
16) Would you like the word graph to also visualize entity co-occurrence and allow filtering by entity types?
17) Should we add a “compliance checklist” panel that scores documents against a configurable schema and flags missing sections?
18) Do you want to add automatic prompt evaluation tests (e.g., run a small suite and track agent output quality over time)?
19) Should the app expose a minimal API endpoint for external systems to submit docs and retrieve summaries/entities programmatically?
20) Would you like a sandbox for rapidly testing LLM OCR prompts and parameters (temperature, top_p) with side-by-side result comparisons?
