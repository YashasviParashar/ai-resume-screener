# ════════════════════════════════════════════════════════════════════════════
# PASTE THIS INTO main.py — replace your existing /match endpoint
# ════════════════════════════════════════════════════════════════════════════

@app.post("/match")
async def match_resume_endpoint(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    resume_text  = extract_text_from_pdf(file)
    result       = match_resume(resume_text, job_description)

    return {
        # ── Original fields (unchanged) ───────────────────────────────────
        "match_score":      result["match_score"],
        "matched_skills":   result["matched_skills"],
        "missing_skills":   result["missing_skills"],

        # ── New SBERT fields ──────────────────────────────────────────────
        "tfidf_score":      result["tfidf_score"],
        "sbert_score":      result["sbert_score"],
        "partial_skills":   result["partial_skills"],
        "section_scores":   result["section_scores"],
        "matching_method":  result["matching_method"],
    }


# ════════════════════════════════════════════════════════════════════════════
# ALSO update /explain endpoint — add these fields to its return dict:
# ════════════════════════════════════════════════════════════════════════════

# Inside your existing explain_endpoint, after match_result = match_resume(...):
# Add these to the return dict:
#
#   "tfidf_score":     match_result["tfidf_score"],
#   "sbert_score":     match_result["sbert_score"],
#   "partial_skills":  match_result["partial_skills"],
#   "section_scores":  match_result["section_scores"],
#   "matching_method": match_result["matching_method"],
