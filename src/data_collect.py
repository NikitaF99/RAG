
# This module builds the cybersecurity knowledge base used
# in the RAG pipeline.
# 1. Web scrapes MITRE ATT&CK Enterprise techniques
# 2. Loads additional cybersecurity QA data from the pAILabs/infosec-security-qa
# 3. Cleans and structures all collected data into
# 4. Combines all documents into a single unified
# 5. Saves the processed knowledge base as JSON

import requests
from bs4 import BeautifulSoup
import time
import json
from config import *
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datasets import load_dataset

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
session.mount("https://", HTTPAdapter(max_retries=retries))


def clean_text(text):
    if not text:
        return ""

    return " ".join(text.split())


def safe_get(url):
    try:
        r = session.get(url, headers=HEADERS, timeout=20)

        if r.status_code == 200:
            return r.text

        print(f"[ERROR] Status {r.status_code}: {url}")
        return None

    except Exception as e:
        print(f"[ERROR] Request failed: {url}")
        print(e)
        return None

def get_all_technique_urls():
    """Get links to all technique pages from the main techniques listing."""
    url = f"{BASE_URL}/techniques/enterprise/"
    html = safe_get(f"{BASE_URL}/techniques/enterprise/")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    
    urls = set() 
    for a in soup.select("td a[href^='/techniques/T']"):
        href = a["href"]
        if href:
            urls.add(href)
    return sorted(list(urls))  # returns 600 technique + sub-technique URLs

def scrape_technique(path):
    """Scrape one technique page and return a text document."""
    url = BASE_URL + path
    html = safe_get(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Basic info
    title = soup.select_one("h1")
    technique_name = clean_text(title.get_text()) if title else "Unknown"
    technique_id = path.strip("/").split("/")[-1]

    # Description
    desc_div = soup.select_one(".description-body")
    description = ""
    if desc_div:
        description = clean_text(desc_div.get_text(separator=" ", strip=True))

    # Card metadata
    metadata = {} 
    for row in soup.select(".card-data"):
        label = row.select_one(".card-title")
        value = row.select_one(".card-value")  
        if label and value:
            key = clean_text(label.get_text())
            val = clean_text(value.get_text())

            metadata[key] = val

    # Mitigations section
    mitigations = []
    mitigation_table = soup.select_one("#mitigations")
    if mitigation_table:
        rows = mitigation_table.select("tr")
        for row in rows:
            cols = [
                clean_text(td.get_text(separator=" "))
                for td in row.find_all("td")
            ]

            if len(cols) >= 3:

                mitigations.append({
                    "mitigation_id": cols[0],
                    "mitigation_name": cols[1],
                    "description": cols[2]
                })

    # Detection section
    detection_header = soup.find(["h2", "h3"],
                                 string=lambda t: t and "Detection" in t
                                )

    if detection_header:
        detection_paragraphs = []
        current = detection_header.find_next_sibling()
        while current:
            if current.name in ["h2", "h3"]:
                break
            text = clean_text(
                current.get_text(separator=" ")
            )
            if text:
                detection_paragraphs.append(text)
            current = current.find_next_sibling()
        detection = "\n".join(detection_paragraphs)

    docs = []

    common_metadata = {
        "technique_id": technique_id,
        "technique_name": technique_name,
        "source": url
    }

    # description doc
    if description:
        docs.append({
            **common_metadata,
            "doc_type": "description",
            "text": description
        })


    # Detection doc
    if detection:
        docs.append({
            **common_metadata,
            "doc_type": "detection",
            "text": detection
        })

    # Platforms, permissions doc
    metadata_text = []

    important_fields = [
        "Platforms",
        "Permissions Required",
        "Data Sources",
        "Defense Bypassed"
    ]

    for field in important_fields:
        if field in metadata:
            metadata_text.append(
                f"{field}: {metadata[field]}"
            )

    if metadata_text:
        docs.append({
            **common_metadata,
            "doc_type": "metadata",
            "text": "\n".join(metadata_text)
        })

    # Mitigation docs
    for mitigation in mitigations:
        docs.append({
            **common_metadata,
            "doc_type": "mitigation",
            "text": mitigation["description"]
        })

    return docs


def webscrape_mitre_techniques():
    print("Fetching technique URLs...")
    urls = get_all_technique_urls()
    print(f"Found {len(urls)} techniques")

    docs = []
    for i, path in enumerate(urls):
        try:
            doc = scrape_technique(path)
            docs.extend(doc)
            print(
                    f"[{i+1}/{len(urls)}] "
                    f"Collected {len(doc)} docs from {path}"
                )
            time.sleep(0.4) 
        except Exception as e:
            print(f"Error occurred while scraping {path}: {e}")
        

    print(f"\nCollected {len(docs)} technique documents")
    return docs

def load_hf_infosec_qa():
    
    dataset = load_dataset("pAILabs/infosec-security-qa",split="train")

    qa_docs = []

    for row in dataset:
        question = clean_text(row["question"])
        answer = clean_text(row["answer"])

        if not answer:
            continue

        qa_docs.append({
            "doc_type": "qa",
            "question": question,
            "text": answer,
            "source": "pAILabs/infosec-security-qa"
        })

    print(f"Loaded {len(qa_docs)} QA docs")

    return qa_docs

def build_knowledge_base():
    # Scrape MITRE techniques
    mitre_docs = webscrape_mitre_techniques()

    # Load Hugging Face InfoSec QA dataset
    hf_docs = load_hf_infosec_qa()

    # Combine all documents into one knowledge base
    all_docs = mitre_docs + hf_docs
    print(f"Total documents in knowledge base: {len(all_docs)}")

    with open(SCRAP, "w") as f:
        json.dump(all_docs, f, indent=2)
    return all_docs

def load_knowledge_base():
    with open(SCRAP, "r") as f:
        all_docs = json.load(f)
    print(f"Loaded {len(all_docs)} documents from {SCRAP}")
    return all_docs
 
if __name__ == "__main__":
    docs = build_knowledge_base()
    
    # print(f"Total: {len(all_docs)} documents")