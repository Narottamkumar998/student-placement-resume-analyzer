
import re

try:
    import pymupdf as fitz
except ImportError:
    import fitz


# Skills that the analyzer can recognize in resumes.
skills_list = [
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "html", "css", "bootstrap", "react", "angular", "node.js", "express",
    "flask", "django", "fastapi", "streamlit",
    "sql", "mysql", "postgresql", "sqlite", "mongodb",
    "pandas", "numpy", "matplotlib", "seaborn", "power bi", "tableau",
    "excel", "power query", "dax",
    "machine learning", "deep learning", "artificial intelligence",
    "scikit-learn", "tensorflow", "pytorch", "nlp", "computer vision",
    "data analysis", "data visualization", "statistics",
    "git", "github", "docker", "aws", "azure", "linux",
    "rest api", "api", "oop", "data structures", "algorithms",
    "communication", "problem solving", "teamwork"
]


# Role-specific skills used for skill-gap analysis.
job_roles = {
    "Data Analyst": [
        "python", "sql", "excel", "power bi", "tableau",
        "pandas", "numpy", "matplotlib", "statistics", "data visualization"
    ],
    "Python Developer": [
        "python", "oop", "data structures", "algorithms",
        "sql", "git", "github", "rest api", "flask", "django"
    ],
    "Frontend Developer": [
        "html", "css", "javascript", "typescript",
        "react", "angular", "bootstrap", "git", "github", "rest api"
    ],
    "Backend Developer": [
        "python", "java", "sql", "mysql", "postgresql",
        "rest api", "api", "flask", "django", "fastapi", "git", "docker"
    ],
    "Java Developer": [
        "java", "oop", "data structures", "algorithms",
        "sql", "mysql", "rest api", "git", "github"
    ],
    "AI/ML Developer": [
        "python", "numpy", "pandas", "statistics", "machine learning",
        "scikit-learn", "deep learning", "tensorflow", "pytorch",
        "nlp", "computer vision", "matplotlib"
    ],
}


def extract_resume_text(pdf_path):
    """Extract selectable text from a PDF resume."""
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts).strip()


def _contains_skill(text, skill):
    """Match a skill as a phrase, avoiding partial-word matches where possible."""
    normalized_text = re.sub(r"\s+", " ", text.lower()).strip()
    normalized_skill = skill.lower().strip()

    # Skills containing punctuation/spaces are matched as literal phrases.
    if normalized_skill in {"c", "c++", "c#", "r"}:
        pattern = r"(?<![a-z0-9+#])" + re.escape(normalized_skill) + r"(?![a-z0-9+#])"
        return re.search(pattern, normalized_text) is not None

    pattern = r"(?<![a-z0-9])" + re.escape(normalized_skill) + r"(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def detect_skills(resume_text, skills):
    """Return recognized skills as a list, using the provided skill list."""
    if not resume_text:
        return []
    found = []
    for skill in skills:
        if _contains_skill(resume_text, skill):
            found.append(skill)
    return sorted(set(found))


def categorize_skills(resume_text):
    """Group detected skills into broad categories."""
    categories = {
        "Programming Languages": [
            "python", "java", "c", "c++", "c#", "javascript", "typescript"
        ],
        "Web Development": [
            "html", "css", "bootstrap", "react", "angular", "node.js",
            "express", "flask", "django", "fastapi", "rest api", "api"
        ],
        "Database": ["sql", "mysql", "postgresql", "sqlite", "mongodb"],
        "Data Analysis & Visualization": [
            "pandas", "numpy", "matplotlib", "seaborn", "power bi",
            "tableau", "excel", "power query", "dax", "data analysis",
            "data visualization", "statistics"
        ],
        "AI & Machine Learning": [
            "machine learning", "deep learning", "artificial intelligence",
            "scikit-learn", "tensorflow", "pytorch", "nlp", "computer vision"
        ],
        "Tools & Platforms": ["git", "github", "docker", "aws", "azure", "linux"],
        "Core Computer Science": ["oop", "data structures", "algorithms"],
        "Soft Skills": ["communication", "problem solving", "teamwork"],
    }

    detected = detect_skills(resume_text, skills_list)
    detected_set = set(detected)
    result = {}

    for category, category_skills in categories.items():
        result[category] = [
            skill for skill in category_skills if skill in detected_set
        ]

    return result


def analyze_skill_gap(found_skills, selected_role):
    """Compare detected resume skills with the selected role's required skills."""
    if selected_role not in job_roles:
        raise ValueError(
            f"Unknown job role: {selected_role}. Available roles: {', '.join(job_roles)}"
        )

    found_set = {str(skill).lower().strip() for skill in found_skills}
    required_skills = list(job_roles[selected_role])
    matched_skills = [skill for skill in required_skills if skill in found_set]
    missing_skills = [skill for skill in required_skills if skill not in found_set]

    total_required = len(required_skills)
    total_matched = len(matched_skills)
    total_missing = len(missing_skills)

    match_percentage = round((total_matched / total_required) * 100, 2) if total_required else 0
    gap_percentage = round((total_missing / total_required) * 100, 2) if total_required else 0

    return {
        "selected_role": selected_role,
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "total_required": total_required,
        "total_matched": total_matched,
        "total_missing": total_missing,
        "match_percentage": match_percentage,
        "gap_percentage": gap_percentage,
    }


def generate_suggestions(missing_skills):
    """Return a practical improvement suggestion for each missing skill."""
    suggestions = {}
    for skill in missing_skills:
        suggestions[skill] = (
            f"Learn {skill} fundamentals, practise with a small hands-on project, "
            "and add the project or evidence of this skill to your resume."
        )
    return suggestions


def display_report(analysis):
    """Print a concise report in the terminal."""
    print("\n===== RESUME SKILL GAP REPORT =====")
    print(f"Target role: {analysis['selected_role']}")
    print(f"Required skills: {analysis['total_required']}")
    print(f"Matched skills: {analysis['total_matched']}")
    print(f"Missing skills: {analysis['total_missing']}")
    print(f"Skill match: {analysis['match_percentage']}%")
    print(f"Skill gap: {analysis['gap_percentage']}%")
    print("Matched:", ", ".join(analysis["matched_skills"]) or "None")
    print("Missing:", ", ".join(analysis["missing_skills"]) or "None")


def export_report(analysis, output_path="resume_skill_gap_report.csv"):
    """Export a skill-gap report to CSV."""
    import pandas as pd

    row = {
        "Job Role": analysis["selected_role"],
        "Total Required Skills": analysis["total_required"],
        "Total Matched Skills": analysis["total_matched"],
        "Total Missing Skills": analysis["total_missing"],
        "Matched Skills": ", ".join(analysis["matched_skills"]),
        "Missing Skills": ", ".join(analysis["missing_skills"]),
        "Skill Match Percentage": analysis["match_percentage"],
        "Skill Gap Percentage": analysis["gap_percentage"],
    }
    pd.DataFrame([row]).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def run_resume_analyzer(pdf_path, output_path="resume_skill_gap_report.csv"):
    """Run the analyzer from a terminal, without using Streamlit."""
    resume_text = extract_resume_text(pdf_path)
    if not resume_text:
        raise ValueError(
            "No readable text was found in the PDF. Scanned resumes may need OCR."
        )

    print("Available job roles:")
    roles = list(job_roles.keys())
    for index, role in enumerate(roles, start=1):
        print(f"{index}. {role}")

    while True:
        choice = input("Select a job role number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(roles):
            selected_role = roles[int(choice) - 1]
            break
        print("Invalid selection. Please enter a listed number.")

    found_skills = detect_skills(resume_text, skills_list)
    analysis = analyze_skill_gap(found_skills, selected_role)
    display_report(analysis)
    export_report(analysis, output_path)
    print(f"\nReport saved to: {output_path}")
    return analysis
