# lib/backend/prev_cases.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Iterable, Optional, Dict
import random, uuid, json, os
from datetime import datetime, timedelta
import re

# ---------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "case_data")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "cases_200000.jsonl")  # default file

# ---------------------------------------------------
# EXTENDED CASE TYPES FOR INDIA
# ---------------------------------------------------
CASE_TYPES: List[str] = [
    "Cheque Bounce – NI Act Sec 138",
    "Civil – Recovery of Money",
    "Civil – Property Dispute",
    "Civil – Motor Accident Claim",
    "Consumer Complaint",
    "Family – Divorce",
    "Family – Domestic Violence",
    "Criminal – Theft IPC 379",
    "Criminal – Cyber Crime IT Act 66C/66D",
    "Criminal – Fraud IPC 420",
    "Criminal – Criminal Breach of Trust IPC 406",
    "Labour – Illegal Termination",
    "Company – Insolvency / IBC",
]

# ---------------------------------------------------
# CITY + NAME DATA
# ---------------------------------------------------
INDIAN_CITIES: List[Tuple[str, str]] = [
    ("Mumbai", "Maharashtra"),
    ("Delhi", "NCT of Delhi"),
    ("Bengaluru", "Karnataka"),
    ("Chennai", "Tamil Nadu"),
    ("Kolkata", "West Bengal"),
    ("Hyderabad", "Telangana"),
    ("Ahmedabad", "Gujarat"),
    ("Pune", "Maharashtra"),
]

FIRST_NAMES = ["Arjun","Priya","Rohan","Neha","Rahul","Nandini","Ravi","Sneha"]
LAST_NAMES = ["Sharma","Verma","Patel","Reddy","Singh","Gupta","Menon","Das"]

COURT_LEVELS = [
    "Metropolitan Magistrate Court",
    "District & Sessions Court",
    "Family Court",
    "Consumer Redressal Forum",
]

OUTCOMES = [
    "Decreed in favour of Plaintiff",
    "Dismissed",
    "Partly Allowed",
    "Compromised / Settled",
    "Withdrawn",
    "Conviction",
    "Acquittal",
]

# ---------------------------------------------------
# DATA MODEL
# ---------------------------------------------------
@dataclass
class PrevCase:
    case_id: str
    case_type: str
    court_name: str
    city: str
    state: str
    filing_date: str
    decision_date: str
    plaintiff: str
    defendant: str
    amount_involved: Optional[int]
    short_summary: str
    outcome: str
    possible_outcome_explained: str

    def to_dict(self) -> Dict:
        return asdict(self)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def _rnd_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def _rnd_dates() -> tuple[str, str]:
    start = datetime(1990, 1, 1)
    filing = start + timedelta(days=random.randint(3000, 12000))
    decision = filing + timedelta(days=random.randint(180, 1200))
    return filing.strftime("%d-%m-%Y"), decision.strftime("%d-%m-%Y")

def _rnd_amount(case_type: str) -> Optional[int]:
    if "Money" in case_type or "Cheque" in case_type:
        return random.randint(20000, 2500000)
    if "Company" in case_type:
        return random.randint(500000, 50000000)
    return None

# ---------------------------------------------------
# GENERATOR
# ---------------------------------------------------
def generate_case() -> PrevCase:
    case_type = random.choice(CASE_TYPES)
    city, state = random.choice(INDIAN_CITIES)
    filing, decision = _rnd_dates()
    plaintiff = _rnd_name()
    defendant = _rnd_name()
    amount = _rnd_amount(case_type)
    outcome = random.choice(OUTCOMES)

    return PrevCase(
        case_id=str(uuid.uuid4()),
        case_type=case_type,
        court_name=f"{random.choice(COURT_LEVELS)}, {city}",
        city=city,
        state=state,
        filing_date=filing,
        decision_date=decision,
        plaintiff=plaintiff,
        defendant=defendant,
        amount_involved=amount,
        short_summary=f"{case_type} involving {plaintiff} vs {defendant}.",
        outcome=outcome,
        possible_outcome_explained=f"{outcome} (amount: ₹{amount})" if amount else outcome,
    )

def generate_and_save(num: int, path: str = DATA_FILE):
    print(f"📦 Generating {num:,} legal cases…")
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(num):
            f.write(json.dumps(generate_case().to_dict(), ensure_ascii=False) + "\n")
    print(f"✔ Saved to: {path}")

# ---------------------------------------------------
# SEARCH + MATCH ENGINE
# ---------------------------------------------------
_CACHE: List[PrevCase] = []

def load_cache(limit: int = 2000, path: str = DATA_FILE):
    global _CACHE
    _CACHE.clear()

    if not os.path.exists(path):
        generate_and_save(limit, path)  # auto-generate if missing

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            _CACHE.append(PrevCase(**json.loads(line)))
    print(f"🔍 Loaded {len(_CACHE)} cases into memory!")

def match_case(user_text: str) -> Dict:
    if not _CACHE:
        load_cache()

    text = user_text.lower()
    best_case = None
    best_score = 0

    for c in _CACHE:
        score = sum([
            c.city.lower() in text,
            c.case_type.lower().split()[0] in text,
        ]) * 25

        if score > best_score:
            best_score = score
            best_case = c

    if not best_case:
        return {}

    return {
        "match_rate": f"{best_score}%",
        "similar_case": best_case.short_summary,
        "predicted_outcome": best_case.possible_outcome_explained
    }

# ---------------------------------------------------
# EXECUTE DIRECTLY TO GENERATE DATASET
# ---------------------------------------------------
if __name__ == "__main__":
    generate_and_save(200000)
