"""
WheelGuard API v2
FastAPI + Supabase + Stripe + Resend
"""
import os, json, tempfile, shutil, secrets, httpx
from datetime import datetime, timezone
import stripe
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client, Client
from analyzer import analyze_model

# ── CONFIG ────────────────────────────────────────────
stripe.api_key        = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
SUPABASE_URL          = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_KEY", "")
RESEND_API_KEY        = os.environ.get("RESEND_API_KEY", "")
FRONTEND_URL          = os.environ.get("FRONTEND_URL", "https://wheelguard-frontend.onrender.com")
PRICE_PRO             = "price_1T3ww0AHfGepCjte6wnWaCnK"
PRICE_TEAM            = "price_1T3wwUAHfGepCjteORV5RU5N"

try:
    print(f"URL: '{SUPABASE_URL}'")
    print(f"KEY length: {len(SUPABASE_SERVICE_KEY)}")
    print(f"KEY start: '{SUPABASE_SERVICE_KEY[:20]}'")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("Supabase OK")
except Exception as e:
    print(f"Supabase init error: {e}")
    supabase = None

app = FastAPI(
    title="WheelGuard API",
    description="Algebraic NaN protection for PyTorch models via Wheel arithmetic",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HELPERS ───────────────────────────────────────────

def get_user_by_email(email: str):
    res = supabase.table("users").select("*").eq("email", email).execute()
    return res.data[0] if res.data else None

def get_user_by_id(user_id: str):
    res = supabase.table("users").select("*").eq("id", user_id).execute()
    return res.data[0] if res.data else None

def create_user(email: str):
    now = datetime.now(timezone.utc)
    res = supabase.table("users").insert({
        "email": email,
        "plan": "free",
        "analyses_count": 0,
        "analyses_reset_month": now.month,
    }).execute()
    return res.data[0]

def get_or_create_user(email: str):
    user = get_user_by_email(email)
    if not user:
        user = create_user(email)
    return user

def reset_if_new_month(user: dict):
    now = datetime.now(timezone.utc)
    if user["analyses_reset_month"] != now.month:
        supabase.table("users").update({
            "analyses_count": 0,
            "analyses_reset_month": now.month
        }).eq("id", user["id"]).execute()
        user["analyses_count"] = 0
        user["analyses_reset_month"] = now.month
    return user

def can_analyze(user: dict) -> bool:
    if user["plan"] in ("pro", "team"):
        return True
    return user["analyses_count"] < 3

def increment_analysis(user: dict):
    supabase.table("users").update({
        "analyses_count": user["analyses_count"] + 1
    }).eq("id", user["id"]).execute()

def save_analysis(user_id: str, filename: str, result: dict):
    supabase.table("analyses").insert({
        "user_id": user_id,
        "filename": filename,
        "result": result,
    }).execute()

async def send_magic_link(email: str, token: str):
    link = f"{FRONTEND_URL}?token={token}&email={email}"
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": "WheelGuard <noreply@resend.dev>",
                "to": email,
                "subject": "🛡 Your WheelGuard login link",
                "html": f"""
                <div style="font-family:monospace;background:#040810;color:#c8d8f0;padding:40px;max-width:500px">
                    <h2 style="color:#00e5ff;letter-spacing:3px">WHEELGUARD</h2>
                    <p>Click the link below to sign in. Valid for 15 minutes.</p>
                    <a href="{link}" style="display:inline-block;margin:24px 0;padding:14px 28px;background:#00e5ff;color:#040810;font-weight:bold;text-decoration:none;letter-spacing:2px">
                        SIGN IN TO WHEELGUARD
                    </a>
                    <p style="color:#4a6a90;font-size:12px">If you didn't request this, ignore this email.</p>
                </div>
                """
            }
        )

def verify_supabase_token(authorization: str) -> dict:
    """Verify JWT from Supabase magic link."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.replace("Bearer ", "")
    try:
        # Use Supabase to verify the token
        res = supabase.auth.get_user(token)
        if not res.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return res.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ── ROUTES ────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "online", "service": "WheelGuard API", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/auth/magic-link")
async def magic_link(request: Request):
    """Send a magic link to the user's email via Supabase Auth."""
    body = await request.json()
    email = body.get("email", "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")

    try:
        # Use Supabase built-in magic link
        supabase.auth.sign_in_with_otp({
            "email": email,
            "options": {
                "email_redirect_to": f"{FRONTEND_URL}/dashboard"
            }
        })
        # Ensure user exists in our users table
        get_or_create_user(email)
        return {"status": "sent", "message": "Magic link sent to your email"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send magic link: {str(e)}")

@app.get("/auth/me")
def get_me(authorization: str = Header(None)):
    """Get current user info and usage."""
    supabase_user = verify_supabase_token(authorization)
    email = supabase_user.email
    user = get_or_create_user(email)
    user = reset_if_new_month(user)
    limit = None if user["plan"] in ("pro", "team") else 3
    return {
        "id": user["id"],
        "email": user["email"],
        "plan": user["plan"],
        "analyses_used": user["analyses_count"],
        "analyses_limit": limit,
        "analyses_remaining": None if limit is None else max(0, limit - user["analyses_count"]),
    }

@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    authorization: str = Header(None)
):
    """Analyze a PyTorch model. Requires authentication."""
    # Auth
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")
    supabase_user = verify_supabase_token(authorization)
    email = supabase_user.email
    user = get_or_create_user(email)
    user = reset_if_new_month(user)

    # Check plan limits
    if not can_analyze(user):
        raise HTTPException(
            status_code=402,
            detail={
                "error": "free_limit_reached",
                "message": "You've used your 3 free analyses this month.",
                "upgrade_url": f"{FRONTEND_URL}/#pricing"
            }
        )

    # File validation
    if not file.filename.endswith(('.pt', '.pth')):
        raise HTTPException(status_code=400, detail="Only .pt and .pth files are supported.")

    # Max 100MB
    contents = await file.read()
    if len(contents) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 100MB.")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)

        report = analyze_model(tmp_path)

        if report.get("status") == "analyzed":
            increment_analysis(user)
            save_analysis(user["id"], file.filename, report)

        user = get_user_by_id(user["id"])
        limit = None if user["plan"] in ("pro", "team") else 3
        report["usage"] = {
            "plan": user["plan"],
            "analyses_used": user["analyses_count"],
            "analyses_remaining": None if limit is None else max(0, limit - user["analyses_count"]),
        }

        return JSONResponse(content=report)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.get("/analyses/history")
def get_history(authorization: str = Header(None)):
    """Get user's analysis history."""
    supabase_user = verify_supabase_token(authorization)
    email = supabase_user.email
    user = get_or_create_user(email)
    res = supabase.table("analyses")\
        .select("id, filename, created_at, result->neutralized, result->total_layers")\
        .eq("user_id", user["id"])\
        .order("created_at", desc=True)\
        .limit(20)\
        .execute()
    return {"analyses": res.data}

@app.post("/create-checkout-session")
async def create_checkout_session(request: Request, authorization: str = Header(None)):
    """Create Stripe Checkout session."""
    supabase_user = verify_supabase_token(authorization)
    email = supabase_user.email
    user = get_or_create_user(email)

    body = await request.json()
    plan = body.get("plan", "pro")
    price_id = PRICE_TEAM if plan == "team" else PRICE_PRO

    # Get or create Stripe customer
    customer_id = user.get("stripe_customer_id")
    if not customer_id:
        customer = stripe.Customer.create(email=email)
        customer_id = customer.id
        supabase.table("users").update({
            "stripe_customer_id": customer_id
        }).eq("id", user["id"]).execute()

    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url=f"{FRONTEND_URL}/dashboard?success=true",
        cancel_url=f"{FRONTEND_URL}/#pricing",
        metadata={"user_id": user["id"], "plan": plan}
    )

    return {"checkout_url": session.url}

@app.post("/cancel-subscription")
async def cancel_subscription(authorization: str = Header(None)):
    """Cancel user's Stripe subscription."""
    supabase_user = verify_supabase_token(authorization)
    email = supabase_user.email
    user = get_or_create_user(email)

    sub_id = user.get("stripe_subscription_id")
    if not sub_id:
        raise HTTPException(status_code=400, detail="No active subscription")

    stripe.Subscription.modify(sub_id, cancel_at_period_end=True)
    return {"status": "cancelled", "message": "Subscription will end at period end"}

@app.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["metadata"].get("user_id")
        plan = session["metadata"].get("plan", "pro")
        sub_id = session.get("subscription")
        if user_id:
            supabase.table("users").update({
                "plan": plan,
                "stripe_subscription_id": sub_id
            }).eq("id", user_id).execute()

    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        customer_id = sub["customer"]
        supabase.table("users").update({
            "plan": "free",
            "stripe_subscription_id": None
        }).eq("stripe_customer_id", customer_id).execute()

    return {"status": "ok"}
