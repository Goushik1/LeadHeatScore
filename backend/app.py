from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, brier_score_loss
from lead_classification_model import (
    best_model,
    best_model_name,
    feature_cols,
    num_features,
    cat_features,
    scaler,
    classify_lead,
    X_test,
    y_test
)
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder

app = Flask(__name__)
CORS(app)
load_dotenv() 

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K_SNIPPETS = 5

# MongoDB setup (vector store)
mongo_client = MongoClient(os.environ.get("MONGO_URI"))
mongo_collection = mongo_client["leads_db"]["leads_embeddings"]

# Elasticsearch setup
es = Elasticsearch(
    os.environ["ELASTICSEARCH_HOST"],
    basic_auth=(os.environ["ELASTICSEARCH_USER"], os.environ["ELASTICSEARCH_PASSWORD"])
)
es_index = "leads_keyword_index"

try:
    mongo_client.admin.command("ping")
    print("MongoDB Atlas connected")
except Exception as e:
    print("MongoDB connection failed:", e)

try:
    if es.ping():
        print("Elasticsearch connected")
    else:
        print("Elasticsearch ping failed")
except Exception as e:
    print("Elasticsearch connection error:", e)


# predictions from trained model
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# reliability plot
def generate_reliability_plot(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
    plt.figure(figsize=(4,4))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.title("Reliability Plot")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed probability")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Metrics endpoint
@app.route("/metrics")
def metrics():
    if X_test is None or y_test is None or len(X_test) == 0 or len(y_test) == 0:
     return jsonify({"error": "No metrics available. Test data not found."}), 404

    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred).tolist()
    roc_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    reliability_plot = generate_reliability_plot(y_test, y_prob)
    
    return jsonify({
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "reliability_plot": reliability_plot
    })


# Load policy doc
with open("policy.txt", "r", encoding="utf-8") as f:
    POLICY_DOC = f.read()


# Example persona snippet lookup
def get_persona_snippet(lead_json):
    role = lead_json.get("role", "").lower()
    if "student" in role:
        return "This lead is a student exploring career opportunities in tech."
    elif "engineer" in role:
        return "This lead is a working professional in software engineering."
    else:
        return "This lead is exploring options for professional upskilling."

# Groq LLM Setup
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7,
    api_key=os.environ.get("API_KEY")

)

json_schema = {
    "type": "object",
    "properties": {
        "channel": {"type": "string"},
        "message": {"type": "string"},
        "rationale": {"type": "string"},
        "reference": {"type": "string"},
    },
    "required": ["channel", "message", "rationale", "reference"],
}

parser = JsonOutputParser(pydantic_object=json_schema)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an outreach assistant. Follow the policy document strictly."),
    ("user", """
Generate the first outreach message ONLY.
Return output as JSON as per schema.

Policy Document: {policy_doc}

Lead Info: {lead_info}

Persona Snippet: {persona_snippet}
     
Lead Class: {lead_class}  # Hot, Warm, or Cold
Important: ONLY use CTA options allowed for this lead class.

Engagement Strategy:

- Hot leads:
  • Use exclusivity (VIP vouchers, early access, loyalty perks).
  • Add surprise rewards (hidden deals, bonus points).
  • Strong CTA: Choose any of these options:
      - Schedule a VIP call within 24–48 hours
      - Join a live workshop this week
      - Redeem your exclusive voucher today
  • Channel: WhatsApp.
  • Tone: Rewarding, appreciative, professional, and friendly.

- Warm leads:
  • Share limited-time coupons or expiring offers.
  • Provide personalized product/service recommendations based on prior_course_interest, page_views, email_opens.
  • Moderate CTA: Invite to webinar, resource, or subscription (no strict 48-hour push).
  • Channel: Email.
  • Tone: Encouraging, resourceful, friendly.

- Cold leads:
  • Provide welcome voucher or first-purchase discount.
  • Highlight best-selling/trending picks.
  • Add social proof (reviews, ratings).
  • Gentle CTA: Newsletter signup or free resource.
  • Channel: Email with minimal frequency.
  • Tone: Simple, welcoming, trust-building.

Message Requirements:
- Keep between 2–4 sentences.
- Personalize using role, region, prior_course_interest, page_views, last_touch_days, recency_days, email_opens, webinar_attended, prior_purchases, and persona snippet.
- Do NOT include sensitive attributes.
- Each JSON output must include:
  {{
    "message": "<the outreach message>",
    "channel": "<whatsapp/email>",
    "cta": "<the main CTA>",
    "engagement_level": "<hot/warm/cold>",
    "rationale": "<why this tone/channel/cta was chosen based on lead attributes>"
  }}
"""),
])


def hybrid_retrieve(query_text, top_k=TOP_K_SNIPPETS):
    # Generate query embedding
    try:
        query_emb = embedding_model.encode([query_text])[0]
    except Exception as e:
        print("Embedding error:", e)
        return []

    # Fetch Mongo candidates
    try:
        vector_candidates = list(mongo_collection.find({}, {"lead_id": 1, "embedding": 1, "_id": 0}))
    except Exception as e:
        print("Mongo fetch error:", e)
        vector_candidates = []

    # Compute similarity, skip invalid embeddings
    q_vec = np.array(query_emb, dtype=float)
    for doc in vector_candidates:
        emb = doc.get("embedding")
        if not emb or len(emb) != 384:  # Defensive check
            doc["score"] = -1.0
            continue
        doc_emb = np.array(emb, dtype=float)
        doc["score"] = float(np.dot(q_vec, doc_emb) / (np.linalg.norm(q_vec) * np.linalg.norm(doc_emb) + 1e-10))

    vector_candidates = sorted([d for d in vector_candidates if d.get("score", -1) > -0.5], key=lambda x: x["score"], reverse=True)[:10]

    # Elasticsearch part (unchanged)
    try:
        es_query = {
            "query": {
                "multi_match": {"query": query_text, "fields": ["role", "prior_course_interest", "region"]}
            },
            "size": 10
        }
        es_results = es.search(index=es_index, body=es_query)
        keyword_candidates = [{"lead_id": hit["_source"].get("lead_id"), "source_doc": hit["_source"]} for hit in es_results["hits"]["hits"]]
    except Exception as e:
        print("Elasticsearch error:", e)
        keyword_candidates = []

    # Merge and re-rank using cross-encoder
    merged = []
    for vc in vector_candidates:
        merged.append({"lead_id": vc.get("lead_id"), "metadata": vc})
    for kc in keyword_candidates:
        merged.append({"lead_id": kc.get("lead_id"), "metadata": kc.get("source_doc", kc)})

    if not merged:
        return []

    cross_inputs = []
    for doc in merged:
        meta = doc["metadata"]
        role = meta.get("role") or meta.get("metadata", {}).get("role") or ""
        pci = meta.get("prior_course_interest") or meta.get("metadata", {}).get("prior_course_interest") or ""
        text_for_rank = f"{role} {pci}"
        cross_inputs.append((query_text, text_for_rank))

    try:
        cross_scores = cross_encoder.predict(cross_inputs)
    except Exception as e:
        print("Cross-encoder error:", e)
        cross_scores = [doc.get("metadata", {}).get("score", 0.0) for doc in merged]

    top = sorted(zip(merged, cross_scores), key=lambda x: x[1], reverse=True)[:top_k]
    top_metadata = []
    for doc, sc in top:
        meta = doc["metadata"]
        if "metadata" in meta and isinstance(meta["metadata"], dict):
            top_metadata.append(meta["metadata"])
        else:
            top_metadata.append(meta)
    return top_metadata


# API-style scoring function
def score_lead_api(lead_json):
    lead_df = pd.DataFrame([lead_json])
    lead_df_encoded = pd.get_dummies(lead_df, columns=cat_features, drop_first=True)
    lead_df_encoded = lead_df_encoded.reindex(columns=feature_cols, fill_value=0)
    lead_df_encoded[num_features] = scaler.transform(lead_df_encoded[num_features])

    prob = float(best_model.predict_proba(lead_df_encoded)[:, 1][0])
    label = classify_lead(prob)

    if best_model_name == "Logistic Regression":
        coef = best_model.base_estimator_.coef_[0]
        top_feats = sorted(zip(feature_cols, coef), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_feats = [(f, float(c)) for f, c in top_feats]
    else:  # XGBoost
        coef = best_model.feature_importances_
        top_feats = sorted(zip(feature_cols, coef), key=lambda x: x[1], reverse=True)[:5]
        top_feats = [(f, float(c)) for f, c in top_feats]

    return {"class": label, "prob": prob, "top_features": top_feats}


@app.route("/score", methods=["POST"])
def score_endpoint():
    lead_json = request.json.get("lead_json")
    
    # Score the lead
    scored = score_lead_api(lead_json)
    
    # Prepare text for embedding
    text_repr = " ".join([f"{k}:{v}" for k, v in lead_json.items()])
    
    # Generate 384-dim embedding using MiniLM
    embedding = embedding_model.encode(text_repr).tolist()
    
    # Insert into MongoDB
    mongo_collection.insert_one({
        "lead_id": lead_json.get("lead_id"),
        "metadata": {**lead_json, **scored},
        "embedding": embedding  # this is the vector for RAG retrieval
    })
    
    # Return scored result
    return jsonify({**lead_json, **scored})


@app.route("/score_csv", methods=["POST"])
def score_csv_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    scored_leads = []
    for _, row in df.iterrows():
        lead_json = row.to_dict()
        scored = score_lead_api(lead_json)

        # Generate embedding and inserting into Mongo
        text_repr = " ".join([f"{k}:{v}" for k,v in lead_json.items()])
        embedding = embedding_model.encode(text_repr).tolist()
        mongo_collection.insert_one({
            "lead_id": lead_json.get("lead_id"),
            "metadata": {**lead_json, **scored},
            "embedding": embedding
        })

        scored_leads.append({**lead_json, **scored})

    top_df = pd.DataFrame(scored_leads)
    top3 = pd.concat([top_df[top_df['class'] == cls].sort_values('prob', ascending=False).head(3) for cls in ["Hot","Warm","Cold"]])

    def serialize_lead(l):
        l["prob"] = float(l["prob"])
        l["top_features"] = [(f, float(c)) for f, c in l.get("top_features", [])]
        return l

    return jsonify({
        "scored_leads": [serialize_lead(l) for l in scored_leads],
        "top_3_per_class": [serialize_lead(l) for l in top3.to_dict(orient="records")]
    })

@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    lead_json = request.json.get("lead_json")
    scored = score_lead_api(lead_json)
    lead_info = {**lead_json, **scored}
    persona_snippet = get_persona_snippet(lead_json)
    query_text = f"{lead_info.get('role','')} interested in {lead_info.get('prior_course_interest','')}"
    context_snippets = hybrid_retrieve(query_text)

    try:
        chain = prompt_template | llm | parser
        response = chain.invoke({
            "policy_doc": POLICY_DOC,
            "lead_info": str(lead_info),
            "persona_snippet": persona_snippet,
            "lead_class": scored["class"],
            "context_snippets": context_snippets
        })
        return jsonify({**response, "lead_id": lead_json.get("lead_id"), "context_snippets": context_snippets, "source": "groq"})
    except Exception:
        lead_emb = embedding_model.encode(" ".join([f"{k}:{v}" for k,v in lead_json.items()]), convert_to_tensor=True)
        past_messages = [
            {"text": "Hi {name}, redeem your VIP voucher today!", "class":"Hot"},
            {"text": "Join our webinar to grab your offer.", "class":"Warm"},
            {"text": "Enjoy a first-purchase discount.", "class":"Cold"}
        ]
        past_embs = embedding_model.encode([p["text"] for p in past_messages], convert_to_tensor=True)
        sims = util.pytorch_cos_sim(lead_emb, past_embs)[0]
        best_idx = sims.argmax().item()
        return jsonify({
            "lead_id": lead_json.get("lead_id"),
            "class": scored["class"],
            "prob": scored["prob"],
            "message": past_messages[best_idx]["text"],
            "similarity_score": float(sims[best_idx]),
            "source": "minilm"
        })

    
# Calibration plot endpoint
@app.route("/calibration", methods=["POST"])
def calibration_endpoint():
    data = request.json
    y_true = data.get("y_true")  
    y_prob = data.get("y_prob")  
    leads = data.get("leads", []) 

    if not y_true or not y_prob or len(y_true) != len(y_prob):
        return jsonify({"error": "y_true and y_prob must be same-length lists"}), 400

    brier = brier_score_loss(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Plot')
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    top3_by_class = []
    if leads:
        scored_leads = []
        for lead_json, prob in zip(leads, y_prob):
            scored = score_lead_api(lead_json)
            scored["prob"] = prob
            scored_leads.append({**lead_json, **scored})
        df = pd.DataFrame(scored_leads)
        top3_by_class = []
        for cls in ["Hot", "Warm", "Cold"]:
            cls_df = df[df["class"]==cls].sort_values("prob", ascending=False).head(3)
            top3_by_class.extend(cls_df.to_dict(orient="records"))

    return jsonify({
        "brier_score": brier,
        "reliability_plot": plot_base64,
        "top_3_per_class": top3_by_class
    })


# A/B test endpoint with RAG insights
manual_scores = []

@app.route("/abtest_score", methods=["POST"])
def abtest_score_endpoint():
    data = request.json
    msg_type = data.get("message_type")  # "template" or "rag"
    score = data.get("score")  # integer 1-5

    if msg_type not in ("template", "rag") or not (1 <= score <= 5):
        return jsonify({"error": "Invalid input"}), 400

    manual_scores.append({"message_type": msg_type, "score": score})
    return jsonify({"status": "success"})


@app.route("/abtest_summary", methods=["GET"])
def abtest_summary_endpoint():
    if not manual_scores:
        return jsonify({"error": "No scores found"}), 404

    df_scores = pd.DataFrame(manual_scores)
    avg_scores = df_scores.groupby("message_type")["score"].mean().to_dict()
    
    try:
        scored_leads_df = pd.DataFrame(manual_scores)  
        top3_by_class = []
        for cls in ["Hot", "Warm", "Cold"]:
            cls_df = scored_leads_df[scored_leads_df.get("class","") == cls].sort_values("score", ascending=False).head(3)
            top3_by_class.extend(cls_df.to_dict(orient="records"))
        avg_scores["top_3_per_class"] = top3_by_class
    except Exception:
        pass

    return jsonify(avg_scores)

@app.route("/data_status", methods=["GET"])
def data_status():
    return jsonify({
        "metrics_available": bool(len(y_test) > 0),
        "abtest_available": bool(len(manual_scores) > 0)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
