"""
matcher.py  —  FIXED VERSION
==============================
ROOT CAUSE OF LOW SCORES:
  TF-IDF cosine similarity on full resume text always gives 5-30% because
  resumes contain hundreds of unique personal words (names, company names,
  dates, addresses) that never appear in a JD — this destroys cosine similarity.

FIX:
  Use a WEIGHTED HYBRID scoring system:
    1. Skill Match Score      — 50% weight  (most important)
    2. Section Coverage Score — 25% weight  (does resume have exp/edu/projects?)
    3. Experience Score       — 15% weight  (years mentioned vs years required)
    4. TF-IDF on KEYWORDS     — 10% weight  (not on full text)

  This gives realistic 60-85% scores for good resumes.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ── Expanded skill bank ──────────────────────────────────────────────────────
SKILL_PATTERNS = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r ",
    "scala", "go ", "rust", "kotlin", "swift", "php", "ruby",
    # Web / Backend
    "react", "angular", "vue", "node", "fastapi", "flask", "django",
    "spring", "html", "css", "rest api", "graphql", "express",
    # Data / ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data analysis", "data science", "tensorflow",
    "keras", "pytorch", "sklearn", "scikit-learn", "pandas", "numpy",
    "scipy", "xgboost", "lightgbm", "huggingface", "transformers",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "sqlite", "bigquery", "cassandra", "dynamodb",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "jenkins", "ci/cd", "linux", "git", "github",
    "databricks", "spark", "hadoop", "kafka", "airflow", "kubeflow",
    # Visualization / BI
    "tableau", "power bi", "matplotlib", "seaborn", "plotly", "excel",
    # Other
    "agile", "scrum", "figma", "photoshop", "communication",
    "leadership", "management", "sagemaker", "mlflow",
]

# Synonyms — if resume has LEFT, it counts as RIGHT in JD
SYNONYMS = {
    "scikit-learn": "sklearn",
    "sklearn":      "scikit-learn",
    "gcp":          "google cloud",
    "google cloud": "gcp",
    "nlp":          "natural language processing",
    "natural language processing": "nlp",
    "pytorch":      "torch",
    "torch":        "pytorch",
    "k8s":          "kubernetes",
    "js":           "javascript",
}

# Section keywords — shows resume is well-structured
SECTION_KEYWORDS = {
    "experience":    ["experience", "work", "employment", "career", "worked at", "company"],
    "education":     ["education", "university", "college", "degree", "bachelor", "master",
                      "btech", "b.tech", "be ", "b.e", "mtech", "phd"],
    "skills":        ["skills", "technologies", "tools", "proficient", "expertise"],
    "projects":      ["project", "built", "developed", "created", "implemented"],
    "achievements":  ["award", "certified", "certification", "achievement", "published",
                      "scholarship", "hackathon", "winner"],
}


def extract_keywords(text: str) -> set:
    """Extract skill keywords from text (with synonym expansion)."""
    text_lower = text.lower()
    found = set()
    for skill in SKILL_PATTERNS:
        if skill in text_lower:
            found.add(skill)
            # Add synonym too so matching works both ways
            if skill in SYNONYMS:
                found.add(SYNONYMS[skill])
    return found


def _skill_match_score(resume_skills: set, jd_skills: set) -> float:
    """
    Score based on what % of JD skills the resume covers.
    If JD has no detectable skills, fall back to resume skill richness.
    """
    if not jd_skills:
        # JD has no detectable skills — score based on resume richness
        return min(len(resume_skills) * 6, 85)

    matched = resume_skills & jd_skills
    # Partial credit: each matched skill = (100 / total_jd_skills)
    score = (len(matched) / len(jd_skills)) * 100

    # Bonus: if resume has MORE skills than JD requires → up to +10 bonus
    bonus = min(max(len(resume_skills) - len(jd_skills), 0) * 1.5, 10)
    return min(score + bonus, 100)


def _section_coverage_score(resume_text: str) -> float:
    """Score how many important sections the resume covers (0-100)."""
    text_lower = resume_text.lower()
    found = 0
    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found += 1
    return (found / len(SECTION_KEYWORDS)) * 100


def _experience_score(resume_text: str, jd_text: str) -> float:
    """
    Compare years of experience in resume vs required in JD.
    Returns 0-100.
    """
    def extract_years(text):
        patterns = re.findall(
            r'(\d+\.?\d*)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s*(?:experience|exp)?',
            text.lower()
        )
        return max([float(y) for y in patterns], default=0)

    resume_yrs = extract_years(resume_text)
    jd_yrs     = extract_years(jd_text)

    if jd_yrs == 0:
        # JD doesn't specify — give full score if candidate has any experience
        return 85 if resume_yrs > 0 else 50

    if resume_yrs >= jd_yrs:
        return 100
    elif resume_yrs >= jd_yrs * 0.5:
        return 70
    elif resume_yrs > 0:
        return 45
    else:
        return 20


def _tfidf_keyword_score(resume_text: str, jd_text: str) -> float:
    """
    TF-IDF on extracted KEYWORDS only (not full text).
    This avoids the noise from personal info, dates, addresses.
    """
    def text_to_keyword_string(text):
        text_lower = text.lower()
        found = [s for s in SKILL_PATTERNS if s in text_lower]
        return " ".join(found) if found else text_lower[:500]

    resume_kw = text_to_keyword_string(resume_text)
    jd_kw     = text_to_keyword_string(jd_text)

    try:
        vectorizer   = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_kw, jd_kw])
        similarity   = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0]) * 100
    except Exception:
        return 50.0


def match_resume(resume_text: str, job_description: str) -> dict:
    """
    FIXED: Match resume against job description using weighted hybrid scoring.

    Weights:
        Skill Match Score      : 50%
        Section Coverage Score : 25%
        Experience Score       : 15%
        TF-IDF Keyword Score   : 10%
    """
    resume_skills = extract_keywords(resume_text)
    jd_skills     = extract_keywords(job_description)

    # ── Component scores ────────────────────────────────────────────────────
    skill_score    = _skill_match_score(resume_skills, jd_skills)
    section_score  = _section_coverage_score(resume_text)
    exp_score      = _experience_score(resume_text, job_description)
    tfidf_score    = _tfidf_keyword_score(resume_text, job_description)

    # ── Weighted final score ─────────────────────────────────────────────────
    final_score = (
        skill_score   * 0.50 +
        section_score * 0.25 +
        exp_score     * 0.15 +
        tfidf_score   * 0.10
    )
    final_score = round(min(final_score, 98), 2)   # cap at 98 (100 = perfect)

    # ── Skill gap analysis ───────────────────────────────────────────────────
    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    if not matched_skills:
        matched_skills = ["No common skills detected"]
    if not missing_skills:
        missing_skills = ["No missing skills — great match!"]

    return {
        "match_score":    final_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        # Breakdown for transparency (shown in /explain endpoint)
        "score_breakdown": {
            "skill_match_score":    round(skill_score,   1),
            "section_coverage":     round(section_score, 1),
            "experience_score":     round(exp_score,     1),
            "tfidf_keyword_score":  round(tfidf_score,   1),
        }
    }