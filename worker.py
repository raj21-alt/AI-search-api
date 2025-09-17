import time
import os
import google.generativeai as genai
from main import Review, ReviewStatus, AuditLog, SessionLocal

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyDuabsAYHTAh_48zmeSZEuNubsaTIbeMvc"))
model = genai.GenerativeModel("gemini-2.0-flash")

def classify_review(text: str):
    """Ask Gemini to classify the review text into moderation categories"""
    prompt = f"""
    You are a moderation system. Classify this text:
    "{text}"

    Return only one label: SAFE, TOXIC, HATE, PROFANITY, or SEXUAL.
    """
    resp = model.generate_content(prompt)
    return resp.text.strip().upper()

def process_review(db, review):
    """Run Gemini moderation and update the review status"""
    try:
        label = classify_review(review.content)
        review.ai_scores = {"label": label}

        if label in ["TOXIC", "HATE", "PROFANITY", "SEXUAL"]:
            review.status = ReviewStatus.QUARANTINED
            review.decision_reason = f"Auto-quarantined: Gemini flagged as {label}"
            db.add(AuditLog(
                review_id=review.id,
                action="AUTO_QUARANTINE",
                actor="system",
                note=label
            ))
        else:
            review.status = ReviewStatus.PUBLISHED
            review.decision_reason = "Auto-published: Gemini marked SAFE"
            db.add(AuditLog(
                review_id=review.id,
                action="AUTO_PUBLISH",
                actor="system",
                note="SAFE"
            ))

        db.commit()
        print(f"✅ Processed review {review.id}: {review.status} ({label})")

    except Exception as e:
        print(f"❌ Error processing review {review.id}: {e}")
        db.rollback()

def run_worker():
    print("🚀 Worker started, polling for reviews...")
    while True:
        db = SessionLocal()
        pending = db.query(Review).filter(Review.status == ReviewStatus.PENDING).all()

        for review in pending:
            process_review(db, review)

        db.close()
        time.sleep(5)  # poll every 5 seconds

if __name__ == "__main__":
    run_worker()
