from app.agentmesh.evals.models import EvalScenario

# 20 hiring scenarios — mix of clear-pass, clear-fail, and boundary cases
# against the deterministic Score Rubric node.
#
# Skill banks (per role, from mock_ats.ROLE_SKILL_BANK):
#   Backend Engineer:  python, distributed systems, postgres, api design, kubernetes
#   AI Engineer:       llm, pytorch, rag, prompt engineering, vector database
#   Frontend Engineer: react, typescript, css, accessibility, performance
#   Data Engineer:     sql, airflow, spark, etl, data modeling
#
# score = 60 * (matched/total_skills) + min(years_experience, 10) * 4
# threshold = 55.0 -> "advanced_to_interview" if score >= 55, else "rejected_by_score"

EVAL_DATASET: list[EvalScenario] = [
    # ── Backend Engineer — strong matches ──
    EvalScenario(
        scenario_id="hiring-001",
        input={
            "candidate_name": "Alex Rivera", "role": "Backend Engineer",
            "resume_text": "5 years building Python services, distributed systems, Postgres, API design, Kubernetes.",
            "years_experience": 5, "target_salary": 170000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    EvalScenario(
        scenario_id="hiring-002",
        input={
            "candidate_name": "Morgan Blake", "role": "Backend Engineer",
            "resume_text": "Junior engineer, some Python scripting experience.",
            "years_experience": 1, "target_salary": 110000,
        },
        expected_tool_sequence=[],
        expected_outcome={"status": "rejected_by_score"},
    ),
    EvalScenario(
        scenario_id="hiring-003",
        input={
            "candidate_name": "Sam Carter", "role": "Backend Engineer",
            "resume_text": "10 years, expert in distributed systems, Kubernetes, and API design at scale.",
            "years_experience": 10, "target_salary": 210000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    EvalScenario(
        scenario_id="hiring-004",
        input={
            "candidate_name": "Jordan Ellis", "role": "Backend Engineer",
            "resume_text": "Marketing coordinator with no engineering background.",
            "years_experience": 3, "target_salary": 90000,
        },
        expected_tool_sequence=[],
        expected_outcome={"status": "rejected_by_score"},
    ),
    # ── AI Engineer ──
    EvalScenario(
        scenario_id="hiring-005",
        input={
            "candidate_name": "Priya Nair", "role": "AI Engineer",
            "resume_text": "Built LLM pipelines with PyTorch, RAG systems, prompt engineering, vector database work.",
            "years_experience": 4, "target_salary": 190000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    EvalScenario(
        scenario_id="hiring-006",
        input={
            "candidate_name": "Chris Doyle", "role": "AI Engineer",
            "resume_text": "General software engineer, no ML or LLM experience listed.",
            "years_experience": 2, "target_salary": 130000,
        },
        expected_tool_sequence=[],
        expected_outcome={"status": "rejected_by_score"},
    ),
    EvalScenario(
        scenario_id="hiring-007",
        input={
            "candidate_name": "Taylor Kim", "role": "AI Engineer",
            "resume_text": "8 years ML research, PyTorch, vector databases, RAG, prompt engineering at scale.",
            "years_experience": 8, "target_salary": 220000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    # ── Frontend Engineer ──
    EvalScenario(
        scenario_id="hiring-008",
        input={
            "candidate_name": "Robin Chen", "role": "Frontend Engineer",
            "resume_text": "React, TypeScript, accessibility, and performance optimization for 6 years.",
            "years_experience": 6, "target_salary": 160000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    EvalScenario(
        scenario_id="hiring-009",
        input={
            "candidate_name": "Devon Ward", "role": "Frontend Engineer",
            "resume_text": "Backend-only background, no frontend framework experience.",
            "years_experience": 4, "target_salary": 140000,
        },
        expected_tool_sequence=[],
        expected_outcome={"status": "rejected_by_score"},
    ),
    EvalScenario(
        scenario_id="hiring-010",
        input={
            "candidate_name": "Casey Nguyen", "role": "Frontend Engineer",
            "resume_text": "CSS, React, TypeScript expert, focused on accessibility for 3 years.",
            "years_experience": 3, "target_salary": 145000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    # ── Data Engineer ──
    EvalScenario(
        scenario_id="hiring-011",
        input={
            "candidate_name": "Harper Diaz", "role": "Data Engineer",
            "resume_text": "SQL, Airflow, Spark pipelines, ETL and data modeling for 7 years.",
            "years_experience": 7, "target_salary": 175000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    EvalScenario(
        scenario_id="hiring-012",
        input={
            "candidate_name": "Skyler Fox", "role": "Data Engineer",
            "resume_text": "Recent bootcamp grad, basic SQL only.",
            "years_experience": 0.5, "target_salary": 95000,
        },
        expected_tool_sequence=[],
        expected_outcome={"status": "rejected_by_score"},
    ),
    # ── Boundary cases around the threshold ──
    EvalScenario(
        scenario_id="hiring-013",
        input={
            "candidate_name": "Quinn Adler", "role": "Backend Engineer",
            "resume_text": "Python, Postgres, API design, and Kubernetes experience, 4 years.",
            "years_experience": 4, "target_salary": 150000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    EvalScenario(
        scenario_id="hiring-014",
        input={
            "candidate_name": "Reese Palmer", "role": "AI Engineer",
            "resume_text": "Some exposure to prompt engineering only.",
            "years_experience": 1, "target_salary": 120000,
        },
        expected_tool_sequence=[],
        expected_outcome={"status": "rejected_by_score"},
    ),
    EvalScenario(
        scenario_id="hiring-015",
        input={
            "candidate_name": "Sage Whitfield", "role": "Data Engineer",
            "resume_text": "SQL, Airflow, and ETL pipelines, 3 years, learning Spark.",
            "years_experience": 3, "target_salary": 130000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    # ── Case-insensitive skill matching ──
    EvalScenario(
        scenario_id="hiring-016",
        input={
            "candidate_name": "Emerson Byrd", "role": "Backend Engineer",
            "resume_text": "PYTHON, POSTGRES, KUBERNETES, API DESIGN — all caps resume export.",
            "years_experience": 5, "target_salary": 165000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    EvalScenario(
        scenario_id="hiring-017",
        input={
            "candidate_name": "Marlowe Stone", "role": "Frontend Engineer",
            "resume_text": "react, typescript, css, accessibility — lowercase resume export, 5 years.",
            "years_experience": 5, "target_salary": 150000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    # ── Unknown role falls back to the default skill bank ──
    EvalScenario(
        scenario_id="hiring-018",
        input={
            "candidate_name": "Bellamy Cruz", "role": "Product Manager",
            "resume_text": "Strong communication, ownership, and collaboration across 6 years of PM roles.",
            "years_experience": 6, "target_salary": 175000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
    EvalScenario(
        scenario_id="hiring-019",
        input={
            "candidate_name": "Indigo Vance", "role": "Product Manager",
            "resume_text": "No relevant soft skills or experience described.",
            "years_experience": 0, "target_salary": 100000,
        },
        expected_tool_sequence=[],
        expected_outcome={"status": "rejected_by_score"},
    ),
    EvalScenario(
        scenario_id="hiring-020",
        input={
            "candidate_name": "Frankie Ross", "role": "Backend Engineer",
            "resume_text": "Full stack: Python, distributed systems, Postgres, API design, and Kubernetes, 12 years.",
            "years_experience": 12, "target_salary": 230000,
        },
        expected_tool_sequence=["schedule_interview"],
        expected_outcome={"status": "advanced_to_interview"},
    ),
]
