# ⚡ AI Resume Screening System
> B.Tech CSE Project — 2nd Year 

![Work In Progress](https://img.shields.io/badge/Status-Work%20In%20Progress-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-purple)

## 🎯 Problem Statement
Companies receive hundreds of resumes per job posting.
Manual screening is slow, inconsistent, and often biased.
This system automates and improves the process using AI.

## ✅ What's Done So Far (25%)

| Module | File | Status |
|--------|------|--------|
| PDF Parser | `resume_parser.py` | ✅ Complete |
| Feature Extractor | `feature_extractor.py` | ✅ Complete |
| Resume Matcher | `matcher.py` | ✅ Complete |
| Bias Detector | `bias_detector.py` | ✅ Complete |
| Explainable AI | `explainability.py` | ✅ Complete |
| ML Training | `train_model.py` | ✅ Complete |
| REST API | `main.py` | ✅ Complete |
| Frontend UI | `index.html` | ✅ Complete |

## 🔮 What's Coming Next (75%)
- [ ] Multi-resume bulk upload and ranking
- [ ] Resume scoring leaderboard for HR
- [ ] Interview question generator per candidate
- [ ] Email notification system for shortlisted candidates
- [ ] Admin dashboard with analytics
- [ ] Database integration (PostgreSQL)
- [ ] Authentication system for HR login
- [ ] Resume improvement chatbot

## 🏗️ Architecture
```
PDF Resume → Parser → Feature Extractor → Matcher
                                              ↓
                              XAI + Bias Detector + Attrition
                                              ↓
                              FastAPI Backend → HTML Frontend
```

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, Uvicorn
- **ML:** Scikit-learn, XGBoost, SHAP
- **Data:** IBM HR Attrition Dataset (1,470 records)
- **Frontend:** HTML, CSS, JavaScript
- **PDF:** PyPDF2

## 🚀 How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (run once)
python -m app.train_model

# Start the server
uvicorn app.main:app --reload

# Open index.html in browser
```

## 📊 Model Results (So Far)
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 87.41% |
| SVM | 86.39% |
| Random Forest | 85.71% |
| XGBoost | 84.35% |

## 👥 Team
- MANJEET KAUR - https://www.linkedin.com/in/manjeet-kaur-410685326?utm_source=share_via&utm_content=profile&utm_medium=member_android
- YASHASVI - https://www.linkedin.com/in/yashasvi-parashar-662817322?utm_source=share_via&utm_content=profile&utm_medium=member_android
- MAHAK - https://www.linkedin.com/in/mahak-gilhotra-2680223aa?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app



