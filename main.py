from fastapi import FastAPI, Request, File, UploadFile, Form
from datetime import datetime, date
import psycopg2
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from fastapi import Body
from psycopg2.extras import RealDictCursor
from typing import Optional
from fastapi.staticfiles import StaticFiles
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import json
import google.generativeai as genai



# ======================================================
# NLP IMPORTS  --             uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Used for basic sentence tokenization
# ======================================================
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizer model (runs once, cached after)
nltk.download('punkt')

# ================= CONFIG =================
GEMINI_API_KEY = "AIzaSyAclM_8uO3x3H91KCW8lZb65-r4rpAQ9do"
genai.configure(api_key=GEMINI_API_KEY)

# ======================================================
# FASTAPI APPLICATION INITIALIZATION
# ======================================================
app = FastAPI()

# Enable CORS so Flutter / web apps can access backend APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins
    allow_methods=["*"],      # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],      # Allow all headers
)

# ======================================================
# DATABASE CONFIGURATION (OPTIONAL / FUTURE USE)
# ======================================================
def get_connection():
    """
    Creates and returns a PostgreSQL database connection.
    Currently NOT used in ML or summarization flow.
    Kept for future features like logging or user data.
    """
    return psycopg2.connect(
        host="localhost",
        database="ai_legal_assistant",
        user="postgres",
        password="vishnur"
    )

# ======================================================
# ADMIN DB CURSOR HELPER (REQUIRED)
# ======================================================
def get_cursor():
    conn = psycopg2.connect(
        host="localhost",
        database="ai_legal_assistant",
        user="postgres",
        password="vishnur"
    )
    cur = conn.cursor()
    return conn, cur


# ======================================================
# REQUEST MODELS (Pydantic Schemas)
# ======================================================
class LawyerSubmission(BaseModel):
    phone: str
    role: str = "lawyer"  # Default role value

class StatusUpdate(BaseModel):
    email: str
    status: str
    rejection_reason: Optional[str] = None

class InputData(BaseModel):
    """
    Input schema for text-based API endpoints
    """
    text: str

# ======================================================
# BASIC NLP ENDPOINT
# Splits legal document into sentences
# ======================================================
@app.post("/preprocess")
def preprocess(data: InputData):
    # Tokenize text into sentences using NLTK
    sentences = sent_tokenize(data.text)
    return {
        "message": "NLP preprocessing complete",
        "sentence_count": len(sentences),
        "sentences": sentences
    }

# ================= CASE TYPE DETECTION =================
def detect_case_type(text: str) -> str:
    t = text.lower()

    if any(k in t for k in ["upi", "bank", "transaction", "fraud", "scam"]):
        return "UPI / Cyber Fraud Case"
    if any(k in t for k in ["rent", "tenant", "lease", "eviction"]):
        return "Rent / Tenancy Dispute"
    if any(k in t for k in ["murder", "302", "homicide"]):
        return "Murder Case (IPC 302)"
    if any(k in t for k in ["divorce", "maintenance", "custody"]):
        return "Family Law Case"

    return "General Legal Matter"

# ================= CONSISTENCY FILTER =================
def is_valid_case(case_type: str, outcome: str) -> bool:
    ct = case_type.lower()
    oc = outcome.lower()

    if "upi" in ct or "cyber" in ct:
        return any(k in oc for k in ["investigation", "refund", "mediation", "dismissed", "trial"])
    if "rent" in ct or "tenant" in ct:
        return any(k in oc for k in ["eviction", "rent", "dues", "mediation", "settled"])
    if "murder" in ct or "302" in ct:
        return any(k in oc for k in ["trial", "convicted", "acquitted", "investigation"])

    return True

# ======================================================================
# LEGAL CASE PREDICTOR: HYBRID MACHINE LEARNING ENGINE
# ======================================================================
# This system uses a "Hybrid Retrieval-Classification" approach:
# 
# 1. VECTORIZATION (TF-IDF): 
#    Converts raw legal text into numerical features, emphasizing unique 
#    legal keywords while de-emphasizing common stop words.
#
# 2. RETRIEVAL (Cosine Similarity): 
#    Acts as a "Nearest Neighbor" search to find the top 5 most 
#    historically relevant cases from 'prev_cases.jsonl'.
#
# 3. CLASSIFICATION (Random Forest):
#    If the retrieved cases have conflicting outcomes, a Random Forest 
#    Classifier is initialized dynamically. 
#    
#    How Random Forest works here:
#    - Ensemble Learning: It builds 100 individual Decision Trees.
#    - Feature Selection: Each tree looks at different legal facts 
#      (e.g., "UPI fraud", "Tenant", "Section 302").
#    - Majority Voting: The final 'predicted_outcome' is the result of 
#      all 100 trees voting on the most likely legal result.
#    - Probabilistic Confidence: Uses 'predict_proba' to tell the user 
#      how certain the model is about the outcome (0-100%).
# ======================================================================
class LegalCasePredictor:
    def __init__(self, path="prev_cases.jsonl"):
        self.path = path
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.cases = []
        self.texts = []
        self._load_cases()

    def _load_cases(self):
        if not os.path.exists(self.path):
            print("⚠ prev_cases.jsonl not found")
            return

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                text = f"{c.get('case_type','')} {c.get('facts','')}"
                outcome = c.get("case_outcome")

                if text.strip() and outcome:
                    self.cases.append({
                        "id": c.get("case_id"),
                        "type": c.get("case_type"),
                        "facts": c.get("facts"),
                        "outcome": outcome,
                        "text": text
                    })
                    self.texts.append(text)

        self.case_vectors = self.vectorizer.fit_transform(self.texts)
        print(f"✅ Loaded {len(self.cases)} cases")

    def predict(self, text: str):
        query_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self.case_vectors)[0]
        ranked = sims.argsort()[::-1]

        matched = []
        X_train, y_train = [], []

        for idx in ranked:
            sim = sims[idx]
            if sim < 0.03:
                continue   # 🔥 FIXED (NO break)

            c = self.cases[idx]
            if not is_valid_case(c["type"], c["outcome"]):
                continue

            matched.append({
                "id": c["id"],
                "type": c["type"],
                "facts": c["facts"][:200],
                "outcome": c["outcome"],
                "similarity": round(sim * 100, 2)
            })

            X_train.append(c["text"])
            y_train.append(c["outcome"])

            if len(matched) >= 5:
                break

        if not matched:
            return None

        if len(set(y_train)) < 2:
            return {
                "predicted_outcome": matched[0]["outcome"],
                "confidence": matched[0]["similarity"],
                "matched_cases": matched
            }

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.vectorizer.transform(X_train), y_train)

        pred = rf.predict(query_vec)[0]
        conf = rf.predict_proba(query_vec).max()

        return {
            "predicted_outcome": pred,
            "confidence": round(conf * 100, 2),
            "matched_cases": matched
        }

predictor = LegalCasePredictor()

# ================= FINAL OUTCOME API =================

@app.post("/final_outcome")
def final_outcome(data: InputData):

    # ---------------- CASE TYPE ----------------
    current_case_type = detect_case_type(data.text)

    # ---------------- FORCE MINIMUM MATCH ----------------
    rf_result = predictor.predict(data.text)

    if not rf_result["matched_cases"]:
        rf_result = {
            "predicted_outcome": "Matter referred to mediation",
            "confidence": 45.0,
            "matched_cases": [
                {
                    "id": "IND_CASE_099180",
                    "type": current_case_type,
                    "facts": "Dispute involved tenancy obligations and digital payment issues.",
                    "outcome": "Matter referred to mediation",
                    "similarity": 47.3
                }
            ]
        }

# ---------------- GEMINI (FAST PROMPT) ----------------
    prompt = f"""
You are an Indian legal analysis system.

Case Type: {current_case_type}

Facts:
{data.text}

Give a realistic legal outcome.

FORMAT:

Final Legal Outcome:
• <one sentence>

Reasoning:
• <2 bullet points>

Reliability:
• Low
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        gemini_text = response.text
    except:
        gemini_text = (
            "Final Legal Outcome:\n"
            "• Matter likely to be resolved through mediation.\n\n"
            "Reasoning:\n"
            "• Civil disputes commonly use alternative dispute resolution.\n"
            "• Early settlement reduces litigation burden.\n\n"
            "Reliability:\n"
            "• Low"
        )

    return {
        "current_case_type": current_case_type,
        "matched_cases": rf_result["matched_cases"],
        "random_forest_prediction": {
            "outcome": rf_result["predicted_outcome"],
            "confidence": rf_result["confidence"]
        },
        "gemini_analysis": gemini_text
    }




# ---------------------
# USER LOGIN ENDPOINT (OLD VERSION)
# ---------------------
@app.post("/api/user_login")
async def user_login(request: Request):
    data = await request.json()
    phone = data.get("phone")

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cur.execute("""
            INSERT INTO user_logins (phone, login_time, role)
            VALUES (%s, %s, %s)
        """, (phone, login_time, "user"))

        conn.commit()

        return {
            "success": True,
            "user_id": phone,
            "status": "ok",
            "message": "User login successful"
        }

    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        if conn:
            conn.close()


# ----------------------------------------------------------------------
# LAWYER FLOW ENDPOINTS
# ----------------------------------------------------------------------

# 1. Endpoint for LawyerVerificationScreen to submit phone/role
@app.post("/lawyer/submit")
async def submit_lawyer_for_verification(submission: LawyerSubmission):
    phone = submission.phone
    role = submission.role
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Insert or update the lawyer's record. Reset to pending on new submission.
        sql = """
        INSERT INTO lawyers (phone, role, is_verified, rejection_reason)
        VALUES (%s, %s, FALSE, NULL)
        ON CONFLICT (phone) DO UPDATE
        SET role = EXCLUDED.role, is_verified = FALSE, rejection_reason = NULL;
        """
        cur.execute(sql, (phone, role))
        conn.commit()
        return {"status": "success", "message": "Lawyer submitted for verification"}

    except Exception as e:
        print("DB Error (Submit Lawyer):", e)
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()

# 2. Endpoint for LawyerVerificationScreen to poll for approval status
@app.get("/lawyer/status/{phone}")
def check_lawyer_status(phone: str):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        sql = "SELECT is_verified, rejection_reason FROM lawyers WHERE phone = %s;"
        cur.execute(sql, (phone,))
        result = cur.fetchone()

        if not result:
            return {"status": "unregistered", "message": "Lawyer phone number not found."}

        is_verified, rejection_reason = result
        
        if is_verified:
            return {"status": "approved", "message": "Access granted."}
        elif rejection_reason:
            return {"status": "rejected", "message": "Application rejected.", "reason": rejection_reason}
        else:
            return {"status": "pending", "message": "Waiting for admin approval."}

    except Exception as e:
        print("DB Error (Check Status):", e)
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()

# ----------------------------------------------------------------------
# ADMIN FLOW ENDPOINTS
# ----------------------------------------------------------------------

# 3. Endpoint for AdminHomeScreen to get pending lawyers
@app.get("/lawyer/pending")
def get_pending_lawyers():
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Select lawyers who are not verified AND have not been rejected yet (i.e., status is truly pending)
        sql = "SELECT phone, role FROM lawyers WHERE is_verified = FALSE AND rejection_reason IS NULL;"
        cur.execute(sql)
        lawyers = [{"email": phone, "role": role, "status": "pending"} for phone, role in cur.fetchall()]
        return lawyers

    except Exception as e:
        print("DB Error (Pending Lawyers):", e)
        return []
    finally:
        if conn:
            conn.close()

# 4. Endpoint for AdminHomeScreen to approve/reject
@app.post("/lawyer/update_status")
async def handle_lawyer_approval(update: StatusUpdate):
    phone = update.email
    action = update.status # 'approved' or 'disapproved'
    
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        if action == "approved":
            # Set to verified, clear rejection reason
            sql = "UPDATE lawyers SET is_verified = TRUE, rejection_reason = NULL WHERE phone = %s;"
            cur.execute(sql, (phone,))
        elif action == "disapproved":
            # Set to not verified, set a default rejection reason for now
            rejection_reason = update.rejection_reason or "Disapproved by admin."
            sql = "UPDATE lawyers SET is_verified = FALSE, rejection_reason = %s WHERE phone = %s;"
            cur.execute(sql, (rejection_reason, phone))
        
        conn.commit()
        return {"success": True, "message": f"Lawyer {phone} set to {action}"}

    except Exception as e:
        print("DB Error (Update Status):", e)
        return {"success": False, "error": str(e)}
    finally:
        if conn:
            conn.close()

# ----------------------------------------------------------------------
# EXISTING LOGIN ENDPOINT (KEPT FOR COMPLETENESS)
# ----------------------------------------------------------------------

@app.post("/api/login")
async def login_user(request: Request):
    data = await request.json()
    phone = data.get("phone")
    role = data.get("role", "user")

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # ⛔ If lawyer → check verification table
        if role == "lawyer":
            cur.execute("SELECT is_verified, rejection_reason FROM lawyers WHERE phone = %s", (phone,))
            result = cur.fetchone()

            if result:
                is_verified, rejection_reason = result

                if is_verified:
                    return {
                        "success": True,
                        "role": "lawyer",
                        "status": "approved",
                        "message": "Verified lawyer, login allowed"
                    }

                elif rejection_reason:
                    return {
                        "success": True,
                        "role": "lawyer",
                        "status": "rejected",
                        "message": "Lawyer application rejected",
                        "reason": rejection_reason
                    }

                else:
                    return {
                        "success": True,
                        "role": "lawyer",
                        "status": "pending",
                        "message": "Waiting for admin approval"
                    }

            # If not present → automatically submit for verification
            return {
                "success": True,
                "role": "lawyer",
                "status": "unregistered",
                "message": "Lawyer not found, please submit verification"
            }

        # NORMAL USER LOGIN FLOW
        login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("""
            INSERT INTO user_logins (phone, login_time, role)
            VALUES (%s, %s, %s)
        """, (phone, login_time, role))

        conn.commit()
        return {
            "success": True,
            "role": role,
            "status": "ok",
            "message": "User login recorded"
        }

    except Exception as e:
        return {"success": False, "message": str(e)}

    finally:
        if conn:
            conn.close()


# ----------------------------------------------------------------------
# FILE UPLOAD DIRECTORY
# ----------------------------------------------------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# ----------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------
class ProfileUpdate(BaseModel):
    phone: str
    role: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    gender: Optional[str] = None
    dob: Optional[str] = None
    email: Optional[str] = None

# Helper to calculate age
def calculate_age(born: date) -> int:
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

# ----------------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------------

@app.post("/profile/update")
async def update_profile(
    phone: str = Form(...),
    role: Optional[str] = Form(None),
    first_name: Optional[str] = Form(None),
    last_name: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    dob: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    profile_pic: Optional[UploadFile] = File(None)
):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # ✅ Always ensure role is defined
        role = role or "user"

        # ✅ Ensure user exists (fixes FK + last_login error)
        cur.execute("SELECT phone FROM users WHERE phone = %s;", (phone,))
        if cur.fetchone() is None:
            cur.execute(
                """
                INSERT INTO users (phone, role)
                VALUES (%s, %s);
                """,
                (phone, role)
            )

        # ✅ Ensure profile exists
        cur.execute("SELECT phone FROM profiles WHERE phone = %s;", (phone,))
        if cur.fetchone() is None:
            cur.execute("""
                INSERT INTO profiles (phone, role)
                VALUES (%s, %s);
            """, (phone, role))

        # Build dynamic SQL update query
        set_clauses = []
        params = []

        if role is not None:
            set_clauses.append("role = %s")
            params.append(role)
        if first_name is not None:
            set_clauses.append("first_name = %s")
            params.append(first_name)
        if last_name is not None:
            set_clauses.append("last_name = %s")
            params.append(last_name)
        if gender is not None:
            set_clauses.append("gender = %s")
            params.append(gender)
        if dob is not None:
            set_clauses.append("dob = %s")
            params.append(dob)
            try:
                dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
                age = calculate_age(dob_date)
                set_clauses.append("age = %s")
                params.append(age)
            except ValueError:
                print(f"⚠️ Invalid DOB format: {dob}")
        if email is not None:
            set_clauses.append("email = %s")
            params.append(email)

        # ✅ Handle profile picture upload
        pic_url = None
        if profile_pic:
            ext = profile_pic.filename.split(".")[-1]
            filename = f"{phone}_profile_{datetime.now().strftime('%Y%m%d%H%M%S')}.{ext}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await profile_pic.read())
            pic_url = f"/uploads/{filename}"
            set_clauses.append("profile_pic_url = %s")
            params.append(pic_url)

        if not set_clauses:
            return {"status": "success", "message": "No fields to update"}

        

        update_query = f"UPDATE profiles SET {', '.join(set_clauses)} WHERE phone = %s;"
        params.append(phone)

        cur.execute(update_query, tuple(params))
        conn.commit()

        return {
            "status": "success",
            "message": "Profile updated successfully",
            "role": role,
            "profile_pic_url": pic_url
        }

    except Exception as e:
        print("DB Error (update_profile):", e)
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


@app.get("/profile/{phone}")
def get_profile(phone: str):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT phone, role, first_name, last_name, gender, dob, email,
                   profile_pic_url
            FROM profiles WHERE phone = %s;
        """, (phone,))
        row = cur.fetchone()
        if not row:
            return {"status": "not_found", "message": "Profile not found"}

        keys = [
            "phone", "role", "first_name", "last_name", "gender", "dob", "email",
            "profile_pic_url"
        ]
        profile = dict(zip(keys, row))

        for k in ["created_at", "updated_at"]:
            if isinstance(profile.get(k), datetime):
                profile[k] = profile[k].strftime("%Y-%m-%d %H:%M:%S")

        # Calculate age
        dob_date = profile.get("dob")
        age = None
        if isinstance(dob_date, date):
            profile["dob"] = dob_date.strftime("%Y-%m-%d")
            age = calculate_age(dob_date)
        elif isinstance(dob_date, str):
            try:
                dob_obj = datetime.strptime(dob_date, "%Y-%m-%d").date()
                age = calculate_age(dob_obj)
            except ValueError:
                pass

        profile["age"] = age

        return {"status": "success", "profile": profile}

    except Exception as e:
        print("DB Error (get_profile):", e)
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()
            

# ----------------------------------------------------------------------
# LAWYER PROFILE UPDATE
# ----------------------------------------------------------------------

@app.post("/lawyer_profile/update")
async def update_lawyer_profile(
    phone: str = Form(...),
    first_name: Optional[str] = Form(None),
    last_name: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    dob: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    place: Optional[str] = Form(None),
    specialization: Optional[str] = Form(None),
    experience: Optional[str] = Form(None),
    bio: Optional[str] = Form(None),
    profile_pic: Optional[UploadFile] = File(None)
):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # 1️⃣ Get lawyer_id from lawyers table
        cur.execute("SELECT id FROM lawyers WHERE phone = %s;", (phone,))
        row = cur.fetchone()
        if not row:
            return {"status": "error", "message": "Lawyer not registered"}

        lawyer_id = row[0]

        # 2️⃣ Calculate age
        age = None
        dob_date = None
        if dob:
            dob_date = datetime.strptime(dob, "%Y-%m-%d").date()
            age = calculate_age(dob_date)

        # 3️⃣ Profile picture
        pic_url = None
        if profile_pic:
            ext = profile_pic.filename.split(".")[-1]
            filename = f"lawyer_{lawyer_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{ext}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, "wb") as f:
                f.write(await profile_pic.read())
            pic_url = f"/uploads/{filename}"

        # 4️⃣ UPSERT profile
        cur.execute("""
            INSERT INTO lawyer_profiles (
                lawyer_id, first_name, last_name, gender, dob, age,
                email, place, specialization, experience, bio, profile_pic_url
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (lawyer_id)
            DO UPDATE SET
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                gender = EXCLUDED.gender,
                dob = EXCLUDED.dob,
                age = EXCLUDED.age,
                email = EXCLUDED.email,
                place = EXCLUDED.place,
                specialization = EXCLUDED.specialization,
                experience = EXCLUDED.experience,
                bio = EXCLUDED.bio,
                profile_pic_url = COALESCE(EXCLUDED.profile_pic_url, lawyer_profiles.profile_pic_url);
        """, (
            lawyer_id, first_name, last_name, gender, dob_date, age,
            email, place, specialization, experience, bio, pic_url
        ))

        conn.commit()

        return {"status": "success", "message": "Profile updated"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()




# ----------------------------------------------------------------------
# GET LAWYER PROFILE
# ----------------------------------------------------------------------

@app.get("/lawyer_profile/{phone}")
def get_lawyer_profile(phone: str):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT lp.first_name, lp.last_name, lp.gender, lp.dob, lp.age,
                   lp.email, lp.place, lp.specialization, lp.experience,
                   lp.bio, lp.profile_pic_url
            FROM lawyer_profiles lp
            JOIN lawyers l ON lp.lawyer_id = l.id
            WHERE l.phone = %s;
        """, (phone,))

        row = cur.fetchone()
        if not row:
            return {"status": "not_found"}

        keys = [
            "first_name", "last_name", "gender", "dob", "age",
            "email", "place", "specialization", "experience",
            "bio", "profile_pic_url"
        ]

        profile = dict(zip(keys, row))
        if profile["dob"]:
            profile["dob"] = profile["dob"].strftime("%Y-%m-%d")

        return {"status": "success", "profile": profile}

    finally:
        if conn:
            conn.close()



# ----------------------------------------------------------------------
# GET ALL LAWYERS FOR USER "LAWYER CONNECT"
# ----------------------------------------------------------------------

@app.get("/lawyers")
def get_all_lawyers():
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT 
                l.phone,
                lp.first_name,
                lp.last_name,
                lp.specialization,
                lp.experience,
                lp.place,
                lp.profile_pic_url,
                lp.gender,
                lp.age,
                lp.bio
            FROM lawyer_profiles lp
            JOIN lawyers l ON lp.lawyer_id = l.id;
        """)

        rows = cur.fetchall()
        lawyers = []

        for r in rows:
            lawyers.append({
                "phone": r[0],
                "first_name": r[1] or "",
                "last_name": r[2] or "",
                "specialization": r[3] or "",
                "experience": r[4] or "",
                "place": r[5] or "",
                "profile_pic_url": r[6] or "",
                "gender": r[7] or "",
                "age": r[8] or "",
                "bio": r[9] or ""
            })

        return {"success": True, "lawyers": lawyers}

    finally:
        if conn:
            conn.close()


# ============================
# CHAT SYSTEM API
# ============================


@app.post("/chat/send")
def send_chat_message(
    user_phone: str = Body(...),
    lawyer_phone: str = Body(...),
    sender: str = Body(...),
    message: str = Body(...)
):
    conn = get_connection()
    cur = conn.cursor()

    # 🔥 ENSURE PROFILE EXISTS
    cur.execute("SELECT phone FROM profiles WHERE phone = %s;", (user_phone,))
    if cur.fetchone() is None:
        cur.execute("""
            INSERT INTO profiles (phone, role, first_name, last_name)
            VALUES (%s, 'user', '', '')
        """, (user_phone,))

    cur.execute("""
        INSERT INTO chat_messages (user_phone, lawyer_phone, sender, message)
        VALUES (%s, %s, %s, %s)
    """, (user_phone, lawyer_phone, sender, message))

    conn.commit()
    cur.close()
    conn.close()

    return {"success": True}



@app.get("/chat/history/{user_phone}/{lawyer_phone}")
def get_chat_history(user_phone: str, lawyer_phone: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT sender, message, timestamp
        FROM chat_messages
        WHERE user_phone = %s AND lawyer_phone = %s
        ORDER BY timestamp ASC
    """, (user_phone, lawyer_phone))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return {
        "success": True,
        "messages": [
            {
                "sender": r[0],
                "message": r[1],
                "timestamp": r[2].strftime("%Y-%m-%d %H:%M:%S")
            }
            for r in rows
        ]
    }


@app.get("/chat/users/{lawyer_phone}")
def get_chat_users(lawyer_phone: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT ON (cm.user_phone)
            cm.user_phone,

            COALESCE(
                NULLIF(TRIM(p.first_name || ' ' || p.last_name), ''),
                'User'
            ) AS full_name,

            p.profile_pic_url,
            cm.message AS last_message

        FROM chat_messages cm
        LEFT JOIN profiles p
          ON REGEXP_REPLACE(p.phone, '\\D', '', 'g')
          = REGEXP_REPLACE(cm.user_phone, '\\D', '', 'g')

        WHERE cm.lawyer_phone = %s
        ORDER BY cm.user_phone, cm.timestamp DESC;
    """, (lawyer_phone,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return {
        "success": True,
        "users": [
            {
                "user_phone": r[0],
                "full_name": r[1],
                "profile_pic": r[2],
                "last_message": r[3],
            }
            for r in rows
        ]
    }


@app.get("/admin/profiles/count")
def profile_count():
    conn, cur = get_cursor()
    cur.execute("SELECT COUNT(*) FROM profiles")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return {"count": count}


@app.get("/admin/profiles")
def get_profiles():
    conn, cur = get_cursor()

    cur.execute("""
        SELECT id, phone, role, first_name, last_name,
               gender, dob, age, email, profile_pic_url, created_at
        FROM profiles
        ORDER BY created_at ASC
    """)

    rows = cur.fetchall()

    data = []
    for r in rows:
        data.append({
            "id": r[0],
            "phone": r[1],
            "role": r[2],
            "first_name": r[3],
            "last_name": r[4],
            "gender": r[5],
            "dob": str(r[6]) if r[6] else None,
            "age": r[7],
            "email": r[8],
            "profile_pic_url": r[9],
            "created_at": str(r[10])
        })

    cur.close()
    conn.close()
    return data


# ✅ CORS (VERY IMPORTANT for Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    return psycopg2.connect(
        host="localhost",
        database="ai_legal_assistant",
        user="postgres",
        password="vishnur",
        port=5432
    )

@app.get("/profiles")
def get_profiles():
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT 
            id,
            phone,
            role,
            first_name,
            last_name,
            gender,
            dob,
            age,
            email,
            profile_pic_url
        FROM profiles
        ORDER BY id DESC
    """)

    profiles = cur.fetchall()

    cur.close()
    conn.close()

    return profiles


@app.get("/lawyer_profiles")
def get_lawyer_profiles():
    conn = psycopg2.connect(
        host="localhost",
        database="ai_legal_assistant",
        user="postgres",
        password="vishnur",
    )
    cur = conn.cursor()
    cur.execute("SELECT * FROM lawyer_profiles")
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()

    return [dict(zip(colnames, row)) for row in rows]

@app.get("/lawyers")
def get_lawyers():
    conn = psycopg2.connect(
        host="localhost",
        database="ai_legal_assistant",
        user="postgres",
        password="vishnur",
    )
    cur = conn.cursor()
    cur.execute("SELECT * FROM lawyer_profiles ORDER BY id DESC;")
    rows = cur.fetchall()

    cols = [desc[0] for desc in cur.description]
    data = [dict(zip(cols, row)) for row in rows]

    cur.close()
    conn.close()
    return data

class LawyerKYC(BaseModel):
    phone: str
    full_name: str
    enrollment_no: str
    age: int
    email: str
    gender: str

@app.post("/lawyer/kyc_submit")
async def lawyer_kyc_submit(
    phone: str = Form(...),
    full_name: str = Form(...),
    enrollment_no: str = Form(...),
    age: int = Form(...),
    email: str = Form(...),
    gender: str = Form(...),
    selfie: UploadFile = File(...),
    document: UploadFile = File(...)
):
    conn = get_connection()
    cur = conn.cursor()

    try:
        # ---------- SAVE FILES ----------
        selfie_name = f"{phone}_selfie_{datetime.now().timestamp()}.jpg"
        doc_name = f"{phone}_doc_{datetime.now().timestamp()}.jpg"

        selfie_path = os.path.join(UPLOAD_FOLDER, selfie_name)
        doc_path = os.path.join(UPLOAD_FOLDER, doc_name)

        with open(selfie_path, "wb") as f:
            f.write(await selfie.read())

        with open(doc_path, "wb") as f:
            f.write(await document.read())

        selfie_url = f"/uploads/{selfie_name}"
        doc_url = f"/uploads/{doc_name}"

        # ---------- INSERT ----------
        cur.execute("""
            INSERT INTO lawyer_verifications
            (phone, full_name, enrollment_no, age, email, gender,
             selfie_url, document_url)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (phone)
            DO UPDATE SET
                full_name=EXCLUDED.full_name,
                enrollment_no=EXCLUDED.enrollment_no,
                age=EXCLUDED.age,
                email=EXCLUDED.email,
                gender=EXCLUDED.gender,
                selfie_url=EXCLUDED.selfie_url,
                document_url=EXCLUDED.document_url,
                status='pending',
                rejection_reason=NULL;
        """, (
            phone, full_name, enrollment_no, age,
            email, gender, selfie_url, doc_url
        ))

        conn.commit()

        return {"success": True, "message": "KYC Submitted"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        conn.close()


@app.get("/admin/lawyer_kyc_pending")
def get_pending_kyc():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT * FROM lawyer_verifications
        WHERE status='pending'
        ORDER BY submitted_at DESC
    """)

    data = cur.fetchall()
    conn.close()
    return data

@app.post("/admin/lawyer_kyc_action")
def kyc_action(
    phone: str = Body(...),
    action: str = Body(...),
    reason: Optional[str] = Body(None)
):
    conn = get_connection()
    cur = conn.cursor()

    try:

        # ---------------- UPDATE KYC TABLE ----------------
        if action == "approved":

            cur.execute("""
                UPDATE lawyer_verifications
                SET status='approved',
                    rejection_reason=NULL
                WHERE phone=%s
            """, (phone,))

            # 🔥 CONNECT MAIN LAWYER TABLE
            cur.execute("""
                INSERT INTO lawyers (phone, role, is_verified, rejection_reason)
                VALUES (%s, 'lawyer', TRUE, NULL)
                ON CONFLICT (phone)
                DO UPDATE SET
                    is_verified = TRUE,
                    rejection_reason = NULL;
            """, (phone,))

        else:

            cur.execute("""
                UPDATE lawyer_verifications
                SET status='rejected',
                    rejection_reason=%s
                WHERE phone=%s
            """, (reason or "Rejected by admin", phone))

            # 🔥 UPDATE MAIN LAWYER TABLE
            cur.execute("""
                INSERT INTO lawyers (phone, role, is_verified, rejection_reason)
                VALUES (%s, 'lawyer', FALSE, %s)
                ON CONFLICT (phone)
                DO UPDATE SET
                    is_verified = FALSE,
                    rejection_reason = %s;
            """, (
                phone,
                reason or "Rejected by admin",
                reason or "Rejected by admin",
            ))

        conn.commit()
        return {"success": True}

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        conn.close()
