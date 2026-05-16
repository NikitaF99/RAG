import requests
from bs4 import BeautifulSoup
import time
import json
from config import *
from datasets import load_dataset

def get_all_technique_urls():
    """Get links to all technique pages from the main techniques listing."""
    r = requests.get(f"{BASE_URL}/techniques/enterprise/", 
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    
    urls = []
    for a in soup.select("td a[href^='/techniques/T']"):
        href = a["href"]
        if href not in urls:
            urls.append(href)
    return urls  # returns ~600 technique + sub-technique URLs

def scrape_technique(path):
    """Scrape one technique page and return a text document."""
    url = BASE_URL + path
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    if r.status_code != 200:
        return None
    
    soup = BeautifulSoup(r.text, "html.parser")
    
    # Name and ID
    name = soup.select_one("h1").get_text(strip=True) if soup.select_one("h1") else ""
    tid  = path.strip("/").split("/")[-1].replace("/", ".")

    # Description
    desc_div = soup.select_one(".description-body")
    description = desc_div.get_text(separator=" ", strip=True) if desc_div else ""

    # Card fields (Platforms, Permissions, Data Sources etc.)
    card_text = ""
    for row in soup.select(".card-data"):
        label = row.select_one(".card-title")
        value = row.select_one(".card-value")  
        if label and value:
            card_text += f"{label.get_text(strip=True)}: {value.get_text(strip=True)}\n"

    # Mitigations section
    mitigations = []
    for row in soup.select("#mitigations ~ table tr"):
        cells = [td.get_text(separator=" ", strip=True) for td in row.select("td")]
        if len(cells) >= 3:
            mitigations.append(f"{cells[0]} - {cells[1]}: {cells[2]}")

    # Detection section
    detection_section = soup.find("h2", string=lambda t: t and "Detection" in t)
    detection = ""
    if detection_section:
        next_p = detection_section.find_next("p")
        if next_p:
            detection = next_p.get_text(separator=" ", strip=True)

    # Compose final document
    text = f"""TECHNIQUE: {name} (ATT&CK ID: {tid})

            DESCRIPTION:
            {description}

            {card_text}
            DETECTION:
            {detection}

            MITIGATIONS:
            {chr(10).join(mitigations) if mitigations else "See ATT&CK page for mitigations."}
            """
    return {"technique_id": tid, "name": name, "text": text.strip(), "source": url}


def webscrape_mitre_techniques():
    print("Fetching technique URLs...")
    urls = get_all_technique_urls()
    print(f"Found {len(urls)} techniques")

    docs = []
    for i, path in enumerate(urls):
        doc = scrape_technique(path)
        if doc:
            docs.append(doc)
        if i % 20 == 0:
            print(f"  {i}/{len(urls)} done...")
        time.sleep(0.4)  # be polite

    print(f"\nDone! Collected {len(docs)} technique documents")

    # Save

    
    return docs

def load_hf_infosec_qa():
    hf_ds = load_dataset("pAILabs/infosec-security-qa", split="train")
    hf_docs = [
        {"text": f"Q: {row['question']}\nA: {row['answer']}", "source": "pAILabs/infosec-qa"}
        for row in hf_ds
    ]

    return hf_docs

def build_knowledge_base():
    # Scrape MITRE techniques
    mitre_docs = webscrape_mitre_techniques()

    # Load Hugging Face InfoSec QA dataset
    hf_docs = load_hf_infosec_qa()

    # Combine all documents into one knowledge base
    all_docs = mitre_docs + hf_docs
    print(f"Total documents in knowledge base: {len(all_docs)}")

    with open(SCRAP, "a") as f:
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