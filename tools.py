# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 12:58:45 2025

@author: johnp
"""
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import CharacterTextSplitter
from typing import TypedDict, List, Dict, Any, Optional, Annotated, Union, Sequence
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import re
from pathlib import Path
import pandas as pd


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def wiki_search(query: str) -> Dict[str, str]:
    """Search Wikipedia for a query and return maximum 2 results.
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"wiki_results": formatted_search_docs}


@tool
def wiki_get_section(title: str, section_title: str, match_mode: str = "iexact") -> dict:
    """
    Fetch a specific section from English Wikipedia using the MediaWiki API.
    - title: Page title (e.g., "Mercedes Sosa")
    - section_title: Section header text (e.g., "Studio albums")
    - match_mode: "iexact" (default, case-insensitive exact match) or "icontains" (case-insensitive substring)
    Returns:
      {
        "url": "...",             # canonical page URL
        "section_index": "5",     # MediaWiki section index (string)
        "section_html": "<...>",  # raw HTML of that section
        "section_text": "...",    # plain text extracted from HTML
        "title": "Mercedes Sosa"
      }
    or { "error": "..." } if not found.
    """

    base = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        # 1) Resolve the page (ensures correct case/redirects) & get sections
        params_sections = {
            "action": "parse",
            "format": "json",
            "prop": "sections",
            "page": title,
            "redirects": 1
        }
        rs = requests.get(base, params=params_sections, headers=headers, timeout=20)
        rs.raise_for_status()
        js = rs.json()

        if "error" in js:
            return {"error": f"Wikipedia API error: {js['error'].get('info', 'unknown')}"}

        parsed = js.get("parse", {})
        canonical_title = parsed.get("title", title)
        sections = parsed.get("sections", [])

        # 2) Find matching section index
        tgt = section_title.strip().lower()
        idx = None
        for sec in sections:
            name = sec.get("line", "").strip()
            low = name.lower()
            if (match_mode == "iexact" and low == tgt) or (match_mode == "icontains" and tgt in low):
                idx = sec.get("index")
                break

        if idx is None:
            # Build a small hint of section names for debugging
            available = [s.get("line", "") for s in sections]
            return {
                "error": f"Section '{section_title}' not found on '{canonical_title}'.",
                "available_sections": available[:30],  # cap for brevity
                "title": canonical_title
            }

        # 3) Fetch only that section's HTML
        params_text = {
            "action": "parse",
            "format": "json",
            "prop": "text",
            "page": canonical_title,
            "section": idx,
            "redirects": 1
        }
        rt = requests.get(base, params=params_text, headers=headers, timeout=20)
        rt.raise_for_status()
        jt = rt.json()

        if "error" in jt:
            return {"error": f"Wikipedia API error: {jt['error'].get('info', 'unknown')}"}

        html = jt["parse"]["text"]["*"]
        text = BeautifulSoup(html, "html.parser").get_text("\n").strip()

        # 4) Build canonical URL to the page (not a specific oldid)
        page_url = f"https://en.wikipedia.org/wiki/{canonical_title.replace(' ', '_')}"

        return {
            "url": page_url,
            "title": canonical_title,
            "section_index": idx,
            "section_html": html,
            "section_text": text
        }

    except requests.exceptions.Timeout:
        return {"error": "Wikipedia request timed out."}
    except requests.RequestException as e:
        return {"error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@tool
def arxiv_search(query: str) -> Dict[str, str]:
    """Search Arxiv for a query and return maximum 3 result.
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arxiv_results": formatted_search_docs}

@tool
def web_search(query: str) -> Dict[str, str]:
    """Search Tavily for a query and return maximum 3 results.
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(
        max_results=5, 
        include_raw_content=True, 
        exclude_domains=['huggingface.co']
    ).invoke(query)
    
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.get("url", "")}" title="{doc.get("title", "")}"/>\n{doc.get("content", "")}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"web_results": formatted_search_docs}

@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given url and reads its content as a markdown string. Use this tool to visit webpages and retrieve their content.
    Args:
        url: The url of the page to visit."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en-US,en;q=0.7"
            }
        # Send a GET request to the URL with a 20-second timeout
        response = requests.get(url, timeout=20,headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        html = response.text[:2_000_000]
        # Convert the HTML content to Markdown
        markdown_content = markdownify(html).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return {"url": url, "content": CharacterTextSplitter(chunk_size=12000, chunk_overlap=0).split_text(markdown_content)[0]}

    except requests.exceptions.Timeout:
        return "The request timed out. Please try again later or check the URL."
    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    
def _split_md_row(line: str) -> list[str]:
    # Trim outer pipes and split into cells
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]

@tool("markdown_table_to_dataframe")
def markdown_table_to_dataframe(
    md_table: str,
    set_index: bool = True,
    index_header_name: str = "*",
) -> pd.DataFrame:
    """
    Convert a Markdown or similar table string to a pandas DataFrame. Use this when you see similar input (table, markdown)

    Args:
        md_table: Markdown table text. Accepts real newlines or literal '\\n'.
        set_index: If True and `index_header_name` exists, set it as the DataFrame index.
        index_header_name: Header name to use as index (default '*').

    Returns:
        pandas.DataFrame
    """
    # Normalize newlines if the string contains literal "\n"
    if "\\n" in md_table and "\n" not in md_table:
        md_table = md_table.replace("\\n", "\n")

    # Clean and split lines, skip blanks
    lines = [ln.strip() for ln in md_table.strip().splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()

    # Header
    header = _split_md_row(lines[0])

    # Optional alignment/separator row detection (|---|:---:|---|)
    align_re = re.compile(r"^\s*:?-{3,}:?\s*$")
    data_start = 1
    if len(lines) > 1:
        maybe_sep = _split_md_row(lines[1])
        if len(maybe_sep) == len(header) and all(align_re.match(c or "") for c in maybe_sep):
            data_start = 2

    # Data rows
    rows = []
    for ln in lines[data_start:]:
        cells = _split_md_row(ln)
        # Normalize length
        if len(cells) < len(header):
            cells += [None] * (len(header) - len(cells))
        elif len(cells) > len(header):
            cells = cells[:len(header)]
        rows.append(cells)

    df = pd.DataFrame(rows, columns=header)

    if set_index and index_header_name in df.columns:
        df = df.set_index(index_header_name)

    return df




def _bytes_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0

@tool("fetch_task_file")
def fetch_task_file(
    task_id: str,
    file_name: str,
    base_url: str = "https://agents-course-unit4-scoring.hf.space/files/",
) -> dict:
    """
    Use this tool when the question asks you to download a file. Ensure a task file is present locally. If missing, download it.

    Args:
        task_id: The task identifier used to build the download URL.
        file_name: Target file name (suffix must be .py, .mp3, or .xlsx).
        base_url: Base URL 

    Returns:
        dict with keys:
          - status: "exists", "downloaded", or "error"
          - path: absolute path of the file (if available)
          - bytes: file size in bytes (if available)
          - url: the download URL used (only when downloaded)
          - message: extra info on errors or warnings
    """
    
    target = Path.cwd() / file_name  # ensure we save/read in the current directory
    

    if target.exists():
        return {
            "status": "exists",
            "path": str(target.resolve()),
            "bytes": _bytes_of(target),
            "message": "File already present in current directory.",
        }

    # Build URL and download
    url = f"{base_url.rstrip('/')}/{task_id}"
    try:
        with requests.get(url, stream=True, timeout=20) as r:
            r.raise_for_status()
            # Stream to disk
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # filter keep-alive chunks
                        f.write(chunk)
    except requests.RequestException as e:
        return {
            "status": "error",
            "path": str(target.resolve()) if target.exists() else "",
            "bytes": _bytes_of(target) if target.exists() else 0,
            "message": f"Download failed: {e}",
        }

    return {
        "status": "downloaded",
        "path": str(target.resolve()),
        "bytes": _bytes_of(target),
        "url": url,
        "message": "Downloaded successfully.",
    }


@tool("read_text_file")
def read_text_file(
    file_name: str = "",
    path: str = "",
    max_bytes: int = 200_000,
) -> dict:
    """
    Use this tool to read a local text (.py, .txt or .md) file and return its content (truncated if large).
    Provide either file_name (look in CWD) or an absolute/relative path.
    """
    if not file_name and not path:
        return {"status": "error", "message": "Provide file_name or path."}

    p = Path(path) if path else (Path.cwd() / file_name)
    if not p.exists():
        return {"status": "error", "message": f"File not found: {p}"}

    data = p.read_bytes()
    truncated = False
    if len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = True

    return {
        "status": "ok",
        "path": str(p.resolve()),
        "truncated": truncated,
        "content": data.decode("utf-8", errors="replace"),
    }


@tool("transcribe_mp3_openai")
def transcribe_mp3_openai(
    file_path: str,
    model: str = "whisper-1",  # or "gpt-4o-mini-transcribe", "gpt-4o-transcribe"
    language: Optional[str] = None,  # e.g., "en", "el"
    prompt: Optional[str] = None,    # optional biasing prompt
) -> str:
    """
    Use this tool to transcribe an audio file (e.g., MP3/M4A/WAV) to text using OpenAI Audio Transcriptions.
    Returns plain text.
    Requires OPENAI_API_KEY in the environment.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
        with open(file_path, "rb") as f:
            # default response format is JSON; we return the `.text` field for plain text
            tr = client.audio.transcriptions.create(
                model=model,
                file=f,
                language=language,
                prompt=prompt,
            )
        # New SDK returns an object with `.text`; be defensive just in case
        return getattr(tr, "text", str(tr))
    except FileNotFoundError:
        return f"[transcribe_mp3_openai] File not found: {file_path}"
    except Exception as e:
        return f"[transcribe_mp3_openai] Error: {e}"
    
    
@tool("load_excel_df")
def load_excel_df(file_name: str) -> pd.DataFrame:
    """
    Use this tool when you want to load a single-sheet Excel file from the current working directory as a pandas DataFrame.

    Args:
        file_name: e.g., 'data.xlsx' (must be in the current working directory)

    Returns:
        dict

    Raises:
        FileNotFoundError: If file doesn't exist in CWD.
        ValueError: If extension isn't allowed.
        PermissionError: If path escapes CWD.
    """
    cwd = Path.cwd().resolve()
    path = (cwd / file_name).resolve()

    # prevent path traversal outside CWD
    try:
        path.relative_to(cwd)
    except ValueError:
        raise PermissionError("Access outside the current working directory is not allowed.")

    if not path.exists():
        raise FileNotFoundError(f"File not found in current directory: {file_name}")


    # Assumes a single sheet; pandas returns a DataFrame for a single sheet by default
    df = pd.read_excel(path)  # requires openpyxl for .xlsx, xlrd for .xls
    return df

@tool("sum_numbers")
def sum_numbers(values: List[Union[int, float, str]]) -> float:
    """
    Use this tool when you want to compute the sum of a list of numbers.

    Args:
        values: A list/sequence of numbers (ints/floats). Numeric strings like "3.5" are accepted.

    Returns:
        float: The total sum (0.0 for an empty list).

    Raises:
        ValueError: If any item can't be interpreted as a number.
    """
    total = 0.0
    for v in values:
        try:
            total += float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Non-numeric value encountered: {v!r}")
    return total
