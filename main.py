from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
import os
import time
import psycopg2
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from psycopg2.extras import RealDictCursor

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_API_TOKEN = os.getenv("APP_API_TOKEN", "")
DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Recall Me AI Backend")


SYSTEM_PROMPT = """
You are an assistant that summarizes personal notes.

RULES:
- Always respond in the SAME LANGUAGE as the input.
- Never explain or define general concepts.
- Focus only on what the user's note says.
- Do not invent details.
- If the note is short, keep it very short.
- If unclear, simplify without adding meaning.
- Highlight actions if present.
- Avoid overly formal or robotic phrasing.
- Preserve key actors (who wants what).
- If there is a conflict or negotiation, clearly state each side’s position.
- Aggressively compress to core meaning.
- Remove minor details (exact numbers, dates, repetitions) unless critical.
- Avoid first-person narration unless necessary.

OUTPUT RULES:
- Prefer 1 sentence. Use 2 only if necessary.
- Keep it concise and natural.
- Use clear and unambiguous phrasing.
- Keep within ~5–25 words in most cases.
- Max ~35 words if needed.
- Ensure correct spelling and grammar.
- Do not repeat the same idea.
- No bullet points, no titles, no quotes.
"""


class NoteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class UserRequest(BaseModel):
    user_id: str = Field(..., min_length=8, max_length=120)

class BillingVerifyRequest(BaseModel):
    user_id: str = Field(..., min_length=8, max_length=120)
    package_name: str = Field(..., min_length=3, max_length=150)
    product_id: str = Field(..., min_length=3, max_length=150)
    purchase_token: str = Field(..., min_length=10)


def verify_app_token(request: Request):
    expected_token = APP_API_TOKEN.strip()
    if expected_token:
        received_token = request.headers.get("X-App-Token", "")
        if received_token != expected_token:
            raise HTTPException(status_code=401, detail="Unauthorized")


def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def get_or_create_user(user_id: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into users (user_id)
                values (%s)
                on conflict (user_id) do nothing
                """,
                (user_id,)
            )

            cur.execute(
                """
                select user_id, is_pro, free_records_left, subscription_state
                from users
                where user_id = %s
                """,
                (user_id,)
            )

            user = cur.fetchone()
            conn.commit()
            return user


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "recall-me-ai-backend"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/user/status")
def user_status(user_id: str, request: Request):
    verify_app_token(request)

    if not user_id or len(user_id.strip()) < 8:
        raise HTTPException(status_code=400, detail="Invalid user_id")

    user = get_or_create_user(user_id.strip())

    return {
        "user_id": user["user_id"],
        "is_pro": user["is_pro"],
        "free_records_left": user["free_records_left"],
        "subscription_state": user["subscription_state"]
    }


@app.post("/user/consume-record")
def consume_record(req: UserRequest, request: Request):
    verify_app_token(request)

    user_id = req.user_id.strip()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into users (user_id)
                values (%s)
                on conflict (user_id) do nothing
                """,
                (user_id,)
            )

            cur.execute(
                """
                select is_pro, free_records_left
                from users
                where user_id = %s
                for update
                """,
                (user_id,)
            )

            user = cur.fetchone()

            if user["is_pro"]:
                conn.commit()
                return {
                    "allowed": True,
                    "is_pro": True,
                    "free_records_left": user["free_records_left"]
                }

            if user["free_records_left"] <= 0:
                conn.commit()
                return {
                    "allowed": False,
                    "is_pro": False,
                    "free_records_left": 0
                }

            new_left = user["free_records_left"] - 1

            cur.execute(
                """
                update users
                set free_records_left = %s,
                    updated_at = now()
                where user_id = %s
                """,
                (new_left, user_id)
            )

            conn.commit()

            return {
                "allowed": True,
                "is_pro": False,
                "free_records_left": new_left
            }

def get_android_publisher_service():
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise HTTPException(status_code=500, detail="Google service account missing")

    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)

    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/androidpublisher"]
    )

    return build("androidpublisher", "v3", credentials=credentials)


@app.post("/billing/verify")
def verify_billing(req: BillingVerifyRequest, request: Request):
    verify_app_token(request)

    user_id = req.user_id.strip()
    purchase_token = req.purchase_token.strip()

    try:
        service = get_android_publisher_service()

        result = service.purchases().subscriptionsv2().get(
            packageName=req.package_name,
            token=purchase_token
        ).execute()

        subscription_state = result.get("subscriptionState", "")

        is_active = subscription_state in [
            "SUBSCRIPTION_STATE_ACTIVE",
            "SUBSCRIPTION_STATE_IN_GRACE_PERIOD"
        ]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into users (user_id)
                    values (%s)
                    on conflict (user_id) do nothing
                    """,
                    (user_id,)
                )

                cur.execute(
                    """
                    update users
                    set is_pro = %s,
                        purchase_token = %s,
                        subscription_state = %s,
                        updated_at = now()
                    where user_id = %s
                    """,
                    (
                        is_active,
                        purchase_token,
                        subscription_state if subscription_state else "unknown",
                        user_id
                    )
                )

                conn.commit()

        return {
            "is_pro": is_active,
            "subscription_state": subscription_state,
            "free_records_left": 10
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Billing verification failed")

@app.post("/summarize")
def summarize_note(req: NoteRequest, request: Request):
    verify_app_token(request)

    note_text = req.text.strip()

    if not note_text:
        raise HTTPException(status_code=400, detail="Text is empty")

    try:
        start = time.time()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=90,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Note:\n{note_text}\n\nSummary:"
                }
            ],
            timeout=20
        )

        summary = response.choices[0].message.content.strip()

        if not summary:
            raise HTTPException(status_code=502, detail="Empty summary")

        return {
            "summary": summary,
            "elapsed_ms": int((time.time() - start) * 1000)
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Summary generation failed")
