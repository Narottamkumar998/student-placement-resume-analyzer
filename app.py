import os
import hmac
from contextlib import contextmanager
import mysql.connector
from mysql.connector import IntegrityError, Error as MySQLError
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

# Import existing Resume Analyzer functions
from resume_analyzer import (
    skills_list,
    job_roles,
    extract_resume_text,
    detect_skills,
    categorize_skills,
    analyze_skill_gap,
    generate_suggestions,
)


# =====================================================
# 1. APPLICATION CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Student Career & Placement System",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
PLACEMENT_PIPELINE_PATH = BASE_DIR / "models" / "placement_pipeline.pkl"


@st.cache_resource
def load_placement_pipeline():
    """Load the trained preprocessing + Logistic Regression pipeline."""
    if not PLACEMENT_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Placement pipeline not found: {PLACEMENT_PIPELINE_PATH}"
        )
    return joblib.load(PLACEMENT_PIPELINE_PATH)


# =====================================================
# 2. MYSQL DATABASE CONNECTION
# =====================================================

def get_setting(name, default=""):
    """Read configuration from Streamlit secrets first, then environment."""
    try:
        value = st.secrets.get(name, None)
        if value is not None:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)


@contextmanager
def get_connection():
    """Open a MySQL connection and safely commit/rollback/close it."""
    connection = mysql.connector.connect(
        host=get_setting("MYSQL_HOST"),
        port=int(get_setting("MYSQL_PORT", "3306")),
        user=get_setting("MYSQL_USER"),
        password=get_setting("MYSQL_PASSWORD"),
        database=get_setting("MYSQL_DATABASE"),
        autocommit=False,
        connection_timeout=20,
        ssl_disabled=False,
    )
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    required = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
    missing = [key for key in required if not get_setting(key)]
    if missing:
        raise RuntimeError(
            "MySQL is not configured. Add these values to the hosting provider's "
            "environment variables/secrets: " + ", ".join(missing)
        )

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                user_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                full_name VARCHAR(150) NOT NULL,
                email VARCHAR(254) NOT NULL UNIQUE,
                phone VARCHAR(30),
                college VARCHAR(200) NOT NULL,
                degree VARCHAR(50) NOT NULL,
                branch VARCHAR(100) NOT NULL,
                graduation_year SMALLINT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.close()


try:
    initialize_database()
except Exception as db_error:
    st.error("Database connection/setup failed. Check your MySQL credentials and network access.")
    st.exception(db_error)
    st.stop()


# =====================================================
# 3. SESSION STATE
# =====================================================

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "delete_message" not in st.session_state:
    st.session_state.delete_message = ""


# =====================================================
# 4. STUDENT DATABASE FUNCTIONS
# =====================================================

def register_student(
    full_name,
    email,
    phone,
    college,
    degree,
    branch,
    graduation_year,
):
    try:
        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO students
                (full_name, email, phone, college, degree, branch, graduation_year)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    full_name.strip(),
                    email.strip().lower(),
                    phone.strip(),
                    college.strip(),
                    degree.strip(),
                    branch.strip(),
                    graduation_year,
                ),
            )
            new_id = cursor.lastrowid
            cursor.close()
            return new_id, None

    except IntegrityError:
        return None, "This email is already registered."
    except MySQLError as error:
        return None, f"Database error: {error}"


def get_student_by_id(user_id):
    with get_connection() as connection:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students WHERE user_id = %s", (user_id,))
        student = cursor.fetchone()
        cursor.close()
    return student


def get_all_students():
    query = """
        SELECT user_id, full_name, email, phone, college, degree, branch,
               graduation_year, created_at
        FROM students
        ORDER BY user_id DESC
    """
    with get_connection() as connection:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
    return pd.DataFrame(rows, columns=[
        "user_id", "full_name", "email", "phone", "college", "degree",
        "branch", "graduation_year", "created_at"
    ])


def delete_student(user_id):
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM students WHERE user_id = %s", (user_id,))
        deleted = cursor.rowcount > 0
        cursor.close()
    return deleted


# =====================================================
# 5. ADMIN AUTHENTICATION
# =====================================================

def verify_admin_password(password):
    saved_password = get_setting("ADMIN_PASSWORD", "")

    if not saved_password:
        return False

    return hmac.compare_digest(password, saved_password)


# =====================================================
# 6. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🎓 Student Career System")

if st.session_state.admin_logged_in:

    page = st.sidebar.radio(
        "Navigation",
        [
            "Admin Dashboard",
            "Admin Logout",
        ],
    )

elif st.session_state.user_id is not None:

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Placement Prediction",
            "Resume Analyzer",
            "Student Logout",
        ],
    )

else:

    page = st.sidebar.radio(
        "Navigation",
        [
            "Registration",
            "Admin Login",
        ],
    )


# =====================================================
# 7. STUDENT REGISTRATION
# =====================================================

if page == "Registration":

    st.title("🎓 Student Registration")

    st.write(
        "Register your basic details to access "
        "the Student Career & Placement System."
    )

    with st.form("registration_form"):

        full_name = st.text_input("Full Name")

        email = st.text_input("Email Address")

        phone = st.text_input("Phone Number (Optional)")

        college = st.text_input("College Name")

        degree = st.selectbox(
            "Degree",
            [
                "B.Tech",
                "B.E.",
                "BCA",
                "MCA",
                "M.Tech",
                "B.Sc.",
                "Other",
            ],
        )

        branch = st.selectbox(
            "Branch",
            [
                "Computer Science",
                "Information Technology",
                "Artificial Intelligence",
                "Data Science",
                "Electronics",
                "Mechanical",
                "Civil",
                "Other",
            ],
        )

        graduation_year = st.selectbox(
            "Graduation Year",
            list(range(2025, 2036)),
        )

        submitted = st.form_submit_button(
            "Register",
            use_container_width=True,
        )

    if submitted:

        if not all(
            [
                full_name.strip(),
                email.strip(),
                college.strip(),
                degree,
                branch,
            ]
        ):

            st.error("Please fill in all required fields.")

        elif "@" not in email or "." not in email.rsplit("@", 1)[-1]:

            st.error("Please enter a valid email address.")

        else:

            user_id, error = register_student(
                full_name,
                email,
                phone,
                college,
                degree,
                branch,
                graduation_year,
            )

            if error:
                st.error(error)

            else:
                st.session_state.user_id = user_id

                # Redirect without showing a registration popup.
                st.rerun()


# =====================================================
# 8. STUDENT HOME PAGE
# =====================================================

elif page == "Home":

    if st.session_state.user_id is None:

        st.warning("Please register first.")

    else:

        student = get_student_by_id(
            st.session_state.user_id
        )

        if student is None:

            st.session_state.user_id = None

            st.warning("Your student record was not found.")

            st.rerun()

        else:

            st.title("🎓 Student Career Dashboard")

            st.subheader(
                f"Welcome, {student['full_name']}!"
            )

            st.write(
                "Use the navigation menu to access "
                "your career tools."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.info(
                    """
                    ### 📊 Placement Prediction

                    Use your academic and skill-related
                    details to estimate your placement status.
                    """
                )

            with col2:

                st.info(
                    """
                    ### 📄 Resume Analyzer

                    Upload your resume, select a target job
                    role, and identify your skill gaps.
                    """
                )

            st.subheader("Your Registered Details")

            details = {
                "Full Name": student["full_name"],
                "Email": student["email"],
                "College": student["college"],
                "Degree": student["degree"],
                "Branch": student["branch"],
                "Graduation Year": student["graduation_year"],
            }

            # Convert values to strings so PyArrow does not receive
            # mixed Python types (for example, text + integer year)
            # in the same DataFrame column.
            details_df = pd.DataFrame(
                [(field, str(value)) for field, value in details.items()],
                columns=["Field", "Value"],
            )

            st.dataframe(
                details_df,
                hide_index=True,
                use_container_width=True,
            )


# =====================================================
# 9. PLACEMENT PREDICTION PAGE
# =====================================================

elif page == "Placement Prediction":

    if st.session_state.user_id is None:

        st.warning("Please register first.")

    else:

        st.title("📊 Placement Prediction")

        st.write(
            "Enter your academic details and placement-related information. "
            "The trained pipeline will preprocess the inputs and predict the result."
        )

        st.info(
            "Prediction is an estimate based on the model's training data; "
            "it does not guarantee actual placement."
        )

        try:
            placement_pipeline = load_placement_pipeline()
        except Exception as error:
            st.error(f"Could not load the placement pipeline: {error}")
            st.caption(
                "Make sure models/placement_pipeline.pkl exists and run this app "
                "using the project's .venv Python environment."
            )
            st.stop()

        with st.form("placement_prediction_form"):
            st.subheader("Academic Details")

            col1, col2 = st.columns(2)

            with col1:
                cgpa = st.number_input(
                    "CGPA (0–10)", min_value=0.0, max_value=10.0,
                    value=7.5, step=0.1, format="%.2f"
                )
                internships = st.number_input(
                    "Number of Internships", min_value=0, max_value=20,
                    value=1, step=1
                )
                projects = st.number_input(
                    "Number of Projects", min_value=0, max_value=50,
                    value=2, step=1
                )
                workshops = st.number_input(
                    "Workshops / Certifications", min_value=0, max_value=50,
                    value=2, step=1
                )
                aptitude_score = st.number_input(
                    "Aptitude Test Score", min_value=0.0, max_value=100.0,
                    value=70.0, step=1.0
                )

            with col2:
                soft_skills_rating = st.number_input(
                    "Soft Skills Rating (1–5)", min_value=1.0, max_value=5.0,
                    value=3.5, step=0.5
                )
                ssc_marks = st.number_input(
                    "SSC Marks (%)", min_value=0.0, max_value=100.0,
                    value=80.0, step=1.0
                )
                hsc_marks = st.number_input(
                    "HSC Marks (%)", min_value=0.0, max_value=100.0,
                    value=75.0, step=1.0
                )
                extracurricular = st.selectbox(
                    "Extracurricular Activities",
                    options=["Yes", "No"]
                )
                placement_training = st.selectbox(
                    "Placement Training",
                    options=["Yes", "No"]
                )

            predict_button = st.form_submit_button(
                "Predict Placement",
                type="primary",
                use_container_width=True
            )

        if predict_button:
            input_df = pd.DataFrame([{
                "CGPA": float(cgpa),
                "Internships": int(internships),
                "Projects": int(projects),
                "Workshops/Certifications": int(workshops),
                "AptitudeTestScore": float(aptitude_score),
                "SoftSkillsRating": float(soft_skills_rating),
                "ExtracurricularActivities": extracurricular,
                "PlacementTraining": placement_training,
                "SSC_Marks": float(ssc_marks),
                "HSC_Marks": float(hsc_marks),
            }])

            try:
                prediction = placement_pipeline.predict(input_df)[0]
                classes = list(getattr(placement_pipeline, "classes_", []))

                # Present a readable label while preserving unknown class labels.
                normalized_prediction = str(prediction).strip().lower()
                if normalized_prediction in {"1", "yes", "placed", "true"}:
                    result_label = "Likely to be Placed"
                elif normalized_prediction in {"0", "no", "not placed", "not_placed", "false"}:
                    result_label = "Not Likely to be Placed"
                else:
                    result_label = f"Predicted Class: {prediction}"

                st.subheader("Prediction Result")
                if normalized_prediction in {"1", "yes", "placed", "true"}:
                    st.success(f"🎉 {result_label}")
                elif normalized_prediction in {"0", "no", "not placed", "not_placed", "false"}:
                    st.warning(f"📌 {result_label}")
                else:
                    st.info(result_label)

                if hasattr(placement_pipeline, "predict_proba"):
                    probabilities = placement_pipeline.predict_proba(input_df)[0]
                    probability_df = pd.DataFrame({
                        "Class": [str(item) for item in classes],
                        "Probability (%)": [
                            round(float(value) * 100, 2)
                            for value in probabilities
                        ],
                    })
                    st.subheader("Prediction Probabilities")
                    st.dataframe(
                        probability_df,
                        hide_index=True,
                        use_container_width=True
                    )
                    st.caption(
                        "Probabilities reflect the model output and should not be "
                        "interpreted as a guarantee of an individual outcome."
                    )

                with st.expander("View submitted input"):
                    st.dataframe(input_df, hide_index=True, use_container_width=True)

            except Exception as error:
                st.error(f"Prediction failed: {error}")
                st.caption(
                    "Check the pipeline file and confirm that the feature names "
                    "match the model's expected inputs."
                )


# =====================================================
# 10. RESUME ANALYZER PAGE
# =====================================================

elif page == "Resume Analyzer":

    if st.session_state.user_id is None:

        st.warning("Please register first.")

    else:

        st.title("📄 Resume Analyzer")

        st.write(
            """
            Upload your resume in PDF format and select
            your target job role.

            The system will identify skills, compare them
            with the selected role, and suggest areas
            for improvement.
            """
        )

        st.warning(
            "Skill matching is based on the keywords "
            "present in the resume. It does not guarantee "
            "job selection."
        )

        uploaded_file = st.file_uploader(
            "Upload your resume (PDF)",
            type=["pdf"],
            key="resume_pdf",
        )

        selected_role = st.selectbox(
            "Select your target job role",
            list(job_roles.keys()),
            key="target_job_role",
        )

        analyze_button = st.button(
            "Analyze Resume",
            type="primary",
            use_container_width=True,
        )

        if analyze_button:

            if uploaded_file is None:

                st.error("Please upload a PDF resume first.")

            else:

                temporary_pdf_path = None

                try:

                    # Save the uploaded PDF temporarily.
                    # The PDF is not permanently stored.
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=False,
                    ) as temporary_file:

                        temporary_file.write(
                            uploaded_file.getvalue()
                        )

                        temporary_pdf_path = temporary_file.name

                    # Extract resume text.
                    resume_text = extract_resume_text(
                        temporary_pdf_path
                    )

                    if not resume_text.strip():

                        st.error(
                            "No readable text was found in "
                            "the PDF. If it is a scanned image, "
                            "OCR may be required."
                        )

                    else:

                        # Detect skills.
                        found_skills = detect_skills(
                            resume_text,
                            skills_list,
                        )

                        # Categorize skills.
                        categorized_skills = categorize_skills(
                            resume_text
                        )

                        # Analyze skills against the selected role.
                        analysis = analyze_skill_gap(
                            found_skills,
                            selected_role,
                        )

                        # Generate recommendations.
                        suggestions = generate_suggestions(
                            analysis["missing_skills"]
                        )

                        st.success(
                            "Resume analysis completed."
                        )

                        # ---------------------------------
                        # Summary metrics
                        # ---------------------------------

                        st.subheader("📊 Skill Gap Summary")

                        col1, col2, col3, col4 = st.columns(4)

                        col1.metric(
                            "Required Skills",
                            analysis["total_required"],
                        )

                        col2.metric(
                            "Matched Skills",
                            analysis["total_matched"],
                        )

                        col3.metric(
                            "Missing Skills",
                            analysis["total_missing"],
                        )

                        col4.metric(
                            "Skill Match",
                            f'{analysis["match_percentage"]}%',
                        )

                        st.progress(
                            min(
                                max(
                                    analysis["match_percentage"] / 100,
                                    0.0,
                                ),
                                1.0,
                            )
                        )

                        st.caption(
                            f'Skill Gap: {analysis["gap_percentage"]}%'
                        )

                        # ---------------------------------
                        # Matched skills
                        # ---------------------------------

                        st.subheader("✅ Matched Skills")

                        matched_skills = sorted(
                            analysis["matched_skills"]
                        )

                        if matched_skills:

                            st.write(
                                ", ".join(
                                    skill.title()
                                    for skill in matched_skills
                                )
                            )

                        else:

                            st.info(
                                "No required skills matched "
                                "the detected resume keywords."
                            )

                        # ---------------------------------
                        # Missing skills
                        # ---------------------------------

                        st.subheader("📌 Missing Skills")

                        missing_skills = sorted(
                            analysis["missing_skills"]
                        )

                        if missing_skills:

                            st.write(
                                ", ".join(
                                    skill.title()
                                    for skill in missing_skills
                                )
                            )

                        else:

                            st.success(
                                "All listed required skills "
                                "were matched."
                            )

                        # ---------------------------------
                        # All detected skills
                        # ---------------------------------

                        st.subheader("🧠 Skills Detected in Resume")

                        if found_skills:

                            st.write(
                                ", ".join(found_skills)
                            )

                        else:

                            st.info(
                                "No skills from the current "
                                "skills database were detected."
                            )

                        # ---------------------------------
                        # Skills by category
                        # ---------------------------------

                        st.subheader("📂 Skills by Category")

                        for category, category_skills in (
                            categorized_skills.items()
                        ):

                            with st.expander(category):

                                if category_skills:

                                    st.write(
                                        ", ".join(category_skills)
                                    )

                                else:

                                    st.write(
                                        "No skills detected "
                                        "in this category."
                                    )

                        # ---------------------------------
                        # Improvement suggestions
                        # ---------------------------------

                        st.subheader(
                            "🚀 Skill Improvement Suggestions"
                        )

                        if suggestions:

                            suggestion_rows = []

                            for skill, suggestion in (
                                suggestions.items()
                            ):

                                suggestion_rows.append(
                                    {
                                        "Missing Skill": skill.title(),
                                        "Recommendation": suggestion,
                                    }
                                )

                            suggestion_df = pd.DataFrame(
                                suggestion_rows
                            )

                            st.dataframe(
                                suggestion_df,
                                hide_index=True,
                                use_container_width=True,
                            )

                        else:

                            st.success(
                                "No improvement suggestions are "
                                "needed for the listed skills."
                            )

                        # ---------------------------------
                        # Downloadable CSV report
                        # ---------------------------------

                        st.subheader("📥 Download Report")

                        report_data = {
                            "Job Role": [
                                analysis["selected_role"]
                            ],
                            "Total Required Skills": [
                                analysis["total_required"]
                            ],
                            "Total Matched Skills": [
                                analysis["total_matched"]
                            ],
                            "Total Missing Skills": [
                                analysis["total_missing"]
                            ],
                            "Matched Skills": [
                                ", ".join(matched_skills)
                            ],
                            "Missing Skills": [
                                ", ".join(missing_skills)
                            ],
                            "Skill Match Percentage": [
                                analysis["match_percentage"]
                            ],
                            "Skill Gap Percentage": [
                                analysis["gap_percentage"]
                            ],
                            "Improvement Suggestions": [
                                " | ".join(
                                    f"{skill.title()}: {suggestion}"
                                    for skill, suggestion
                                    in suggestions.items()
                                )
                            ],
                        }

                        report_df = pd.DataFrame(report_data)

                        st.download_button(
                            label="Download Skill Gap Report (CSV)",
                            data=report_df.to_csv(
                                index=False
                            ).encode("utf-8-sig"),
                            file_name="resume_skill_gap_report.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                except Exception as error:

                    st.error(
                        f"Resume analysis failed: {error}"
                    )

                    st.caption(
                        "Check that resume_analyzer.py and its "
                        "required libraries are available."
                    )

                finally:

                    # Remove the temporary PDF after processing.
                    if (
                        temporary_pdf_path
                        and os.path.exists(temporary_pdf_path)
                    ):

                        os.remove(temporary_pdf_path)


# =====================================================
# 11. STUDENT LOGOUT
# =====================================================

elif page == "Student Logout":

    st.session_state.user_id = None

    st.rerun()


# =====================================================
# 12. ADMIN LOGIN
# =====================================================

elif page == "Admin Login":

    st.title("🔐 Admin Login")

    st.write(
        "Only the administrator can access student records."
    )

    with st.form("admin_login_form"):

        admin_password = st.text_input(
            "Admin Password",
            type="password",
        )

        login_button = st.form_submit_button(
            "Login",
            use_container_width=True,
        )

    if login_button:

        if verify_admin_password(admin_password):

            st.session_state.admin_logged_in = True

            st.rerun()

        else:

            if not get_setting("ADMIN_PASSWORD"):

                st.error(
                    "Admin password is not configured. "
                    "Set the ADMIN_PASSWORD environment variable "
                    "before starting the application."
                )

            else:

                st.error("Incorrect admin password.")


# =====================================================
# 13. ADMIN DASHBOARD
# =====================================================

elif page == "Admin Dashboard":

    if not st.session_state.admin_logged_in:

        st.error("Please log in as admin.")

        st.stop()

    st.title("🔐 Admin Dashboard")

    st.write(
        "View and manage student registration records."
    )

    if st.session_state.delete_message:

        st.success(st.session_state.delete_message)

        st.session_state.delete_message = ""

    students_df = get_all_students()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Registered Students",
            len(students_df),
        )

    with col2:

        st.metric(
            "Graduation Years",
            (
                students_df["graduation_year"].nunique()
                if not students_df.empty
                else 0
            ),
        )

    st.subheader("👨‍🎓 Student Records")

    if students_df.empty:

        st.info("No student registrations found.")

    else:

        st.dataframe(
            students_df,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            label="Download Student Records (CSV)",
            data=students_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="student_records.csv",
            mime="text/csv",
        )

        st.divider()

        st.subheader("🗑️ Delete Student Record")

        student_options = {
            (
                f"{row['full_name']} | "
                f"{row['email']} | "
                f"ID: {row['user_id']}"
            ): int(row["user_id"])
            for _, row in students_df.iterrows()
        }

        selected_student_label = st.selectbox(
            "Select student",
            list(student_options.keys()),
        )

        confirm_delete = st.checkbox(
            "I confirm that I want to delete this record."
        )

        if st.button(
            "Delete Selected Student",
            type="primary",
        ):

            if not confirm_delete:

                st.warning(
                    "Please confirm before deleting the record."
                )

            else:

                selected_id = student_options[
                    selected_student_label
                ]

                deleted = delete_student(selected_id)

                if deleted:

                    if (
                        st.session_state.user_id
                        == selected_id
                    ):

                        st.session_state.user_id = None

                    st.session_state.delete_message = (
                        "Student record deleted successfully."
                    )

                    st.rerun()

                else:

                    st.error(
                        "The selected record could not be deleted."
                    )


# =====================================================
# 14. ADMIN LOGOUT
# =====================================================

elif page == "Admin Logout":

    st.session_state.admin_logged_in = False

    st.rerun()