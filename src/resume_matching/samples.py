"""
The hand-checked reference set the engine is measured against.

Each sample states, alongside its text, the values a human read off it:
years of experience and education level, and age and CGPA where stated. The
test suite checks that the engine reproduces those numbers, and that the
two backend-relevant resumes outrank every irrelevant one against
SAMPLE_JOB_DESCRIPTION.

The first five are the core ranking set. The rest cover tricky readings: a fresher stating an
age and a DOB, dates with no stated claim, overlapping jobs, a Masters with a
CGPA, and two-digit `Jan'23` years. Their date ranges are all closed, so the
expected totals do not drift with the calendar.

Text lives here rather than in binary fixtures so the expected values sit
next to the words they come from. `resume_matching.samples.render_samples`
renders them into real PDF and DOCX files for the file-parsing path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleResume:
    key: str
    filename: str
    lines: tuple[str, ...]
    #: Hand-checked: total years of experience a human reads off this resume.
    expected_experience_years: float
    #: Hand-checked: education levels literally named in the text.
    expected_education_levels: frozenset[str]
    #: Whether this resume is relevant to SAMPLE_JOB_DESCRIPTION.
    relevant: bool
    #: Hand-checked, where the resume states one: the age and the ten-point
    #: CGPA. None means the resume states none, and nothing is scored.
    expected_age: int | None = None
    expected_cgpa: float | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def format(self) -> str:
        return self.filename.rsplit(".", 1)[-1]


SAMPLE_JOB_DESCRIPTION = """
Backend Engineer — Placement Drive 2026

We are hiring a backend engineer to build and maintain REST APIs for our
logistics platform. You will design database schemas, write Django views and
serializers, tune PostgreSQL queries, and run background jobs with Celery.

Required:
- 2+ years of professional experience writing backend services in Python
- Strong Django and Django REST Framework knowledge
- Relational database design and SQL query optimisation, ideally PostgreSQL
- Comfortable with Git, Docker and continuous integration
- A bachelor's degree in computer science, engineering or a related field

Nice to have: Celery or similar task queues, AWS deployment experience,
writing unit and integration tests.
""".strip()


SAMPLE_RESUMES: tuple[SampleResume, ...] = (
    SampleResume(
        key="backend_engineer",
        filename="aarav_mehta_backend.pdf",
        lines=(
            "Aarav Mehta",
            "aarav.mehta@college.edu | 220200901",
            "",
            "SUMMARY",
            "Backend engineer with 3 years of professional experience building REST",
            "APIs in Python and Django. Comfortable owning database design,",
            "query optimisation and background job processing end to end.",
            "",
            "EDUCATION",
            "B.Tech in Computer Engineering, 2026",
            "",
            "EXPERIENCE",
            "Backend Developer, Loadstar Logistics",
            "Built Django REST Framework APIs backing a fleet tracking product.",
            "Designed PostgreSQL schemas and tuned slow queries with EXPLAIN ANALYZE.",
            "Moved nightly reporting onto Celery workers running in Docker.",
            "",
            "SKILLS",
            "Python, Django, Django REST Framework, PostgreSQL, SQL, Celery,",
            "Docker, Git, pytest, continuous integration, AWS",
        ),
        expected_experience_years=3.0,
        expected_education_levels=frozenset({"UG"}),
        relevant=True,
    ),
    SampleResume(
        key="data_engineer",
        filename="sana_qureshi_data.docx",
        lines=(
            "Sana Qureshi",
            "sana.qureshi@college.edu | 220200904",
            "",
            "SUMMARY",
            "Data engineer with 2 years 6 months of experience writing Python",
            "services and SQL pipelines. Built and maintained ETL jobs moving data",
            "into PostgreSQL for downstream reporting.",
            "",
            "EDUCATION",
            "M.Tech in Data Science, 2026",
            "",
            "EXPERIENCE",
            "Data Engineer, Nimbus Analytics",
            "Wrote Python ETL pipelines orchestrated with Airflow.",
            "Modelled PostgreSQL warehouse tables and optimised reporting queries.",
            "Packaged jobs in Docker and shipped them through a CI pipeline.",
            "",
            "SKILLS",
            "Python, SQL, PostgreSQL, Airflow, pandas, ETL, Docker, Git, Django ORM",
        ),
        expected_experience_years=2.5,
        expected_education_levels=frozenset({"Masters"}),
        relevant=True,
    ),
    SampleResume(
        key="frontend_intern",
        filename="kabir_nair_frontend.pdf",
        lines=(
            "Kabir Nair",
            "kabir.nair@college.edu | 220200905",
            "",
            "SUMMARY",
            "Front-end focused developer with 6 months of internship experience",
            "building marketing pages and design-system components.",
            "",
            "EDUCATION",
            "Pursuing BCA, expected 2027",
            "",
            "EXPERIENCE",
            "Front-end Intern, Bright Studio",
            "Built responsive landing pages in React and Tailwind CSS.",
            "Translated Figma designs into reusable components.",
            "",
            "SKILLS",
            "React, TypeScript, HTML, CSS, Tailwind, Figma, accessibility",
        ),
        expected_experience_years=0.5,
        expected_education_levels=frozenset({"UG"}),
        relevant=False,
    ),
    SampleResume(
        key="mechanical_designer",
        filename="arjun_deshmukh_mechanical.docx",
        lines=(
            "Arjun Deshmukh",
            "arjun.deshmukh@college.edu | 220200909",
            "",
            "SUMMARY",
            "Mechanical design engineer with 4 years of experience in sheet metal",
            "and injection-moulded part design for automotive assemblies.",
            "",
            "EDUCATION",
            "B.E. in Mechanical Engineering, 2026",
            "",
            "EXPERIENCE",
            "Design Engineer, Pinnacle Auto Components",
            "Produced 3D models and manufacturing drawings in SolidWorks.",
            "Ran thermal and stress simulations on bracket assemblies.",
            "Applied GD&T to drawings and led design reviews with suppliers.",
            "",
            "SKILLS",
            "SolidWorks, AutoCAD, CATIA, GD&T, FEA, sheet metal design, ANSYS",
        ),
        expected_experience_years=4.0,
        expected_education_levels=frozenset({"UG"}),
        relevant=False,
    ),
    SampleResume(
        key="marketing_executive",
        filename="ishita_banerjee_marketing.pdf",
        lines=(
            "Ishita Banerjee",
            "ishita.banerjee@college.edu | 220200910",
            "",
            "SUMMARY",
            "Marketing executive with 1 year of experience running content and",
            "social campaigns for consumer brands.",
            "",
            "EDUCATION",
            "BBA in Marketing, 2026",
            "",
            "EXPERIENCE",
            "Marketing Executive, Hearth & Co.",
            "Planned editorial calendars and wrote long-form blog content.",
            "Ran paid social campaigns and reported on channel performance.",
            "",
            "SKILLS",
            "Content marketing, SEO, copywriting, Google Analytics, Canva,",
            "email campaigns, social media strategy",
        ),
        expected_experience_years=1.0,
        expected_education_levels=frozenset({"UG"}),
        relevant=False,
    ),
    SampleResume(
        key="civil_fresher",
        filename="rohan_patil_civil.pdf",
        lines=(
            "Rohan Patil",
            "rohan.patil@college.edu | +91 98220 11223",
            "Age: 21 | Date of Birth: 14/05/2005",
            "",
            "SUMMARY",
            "Final-year student of civil engineering. Interested in bridges, dams,",
            "canals, irrigation and the survey of farmland.",
            "",
            "EDUCATION",
            "B.Tech in Civil Engineering, 2026",
            "CGPA: 7.8/10",
            "",
            "PROJECTS",
            "Rainwater harvesting tank for a village school.",
            "Soil testing for a farmhouse foundation.",
            "",
            "SKILLS",
            "AutoCAD, STAAD Pro, surveying, levelling, concrete testing",
        ),
        expected_experience_years=0.0,
        expected_education_levels=frozenset({"UG"}),
        relevant=False,
        expected_age=21,
        expected_cgpa=7.8,
    ),
    SampleResume(
        key="electronics_dated",
        filename="meera_iyer_electronics.docx",
        lines=(
            "Meera Iyer",
            "meera.iyer@college.edu",
            "",
            "SUMMARY",
            "Electronics student who enjoys board bring-up, firmware debugging and",
            "hardware validation for consumer devices.",
            "",
            "EDUCATION",
            "B.E. in Electronics and Telecommunication, 2022 - 2026",
            "",
            "EXPERIENCE",
            "Embedded Systems Intern, Volta Devices",
            "Jun 2023 - Nov 2023",
            "Wrote STM32 firmware drivers and tested sensor boards.",
            "Hardware Test Engineer Intern, Circuitry Labs",
            "Jan 2024 - Jun 2024",
            "Built test jigs and ran EMI pre-compliance checks.",
            "",
            "SKILLS",
            "Embedded C, STM32, PCB design, KiCad, oscilloscopes, soldering",
        ),
        expected_experience_years=1.0,
        expected_education_levels=frozenset({"UG"}),
        relevant=False,
    ),
    SampleResume(
        key="operations_multi_job",
        filename="farhan_shaikh_operations.pdf",
        lines=(
            "Farhan Shaikh",
            "farhan.shaikh@college.edu",
            "",
            "SUMMARY",
            "Operations analyst who plans inventory, negotiates with transport",
            "vendors and keeps warehouse dispatch on schedule.",
            "",
            "EXPERIENCE",
            "Operations Analyst, Meridian Freight, Jan 2021 - Dec 2022",
            "Planned weekly truck allocation across four regional depots.",
            "Part-time Supply Chain Consultant, Northwind Traders, Jun 2022 - Jun 2023",
            "Redesigned the reorder policy for fast-moving retail goods.",
            "Inventory Planning Associate, Cobalt Retail, Sep 2023 - Feb 2024",
            "Ran cycle counts and cut stock-outs in two stores.",
            "",
            "EDUCATION",
            "BBA in Supply Chain Management, 2020",
            "",
            "SKILLS",
            "Inventory planning, vendor negotiation, demand forecasting, Excel",
        ),
        # Jan 2021 - Jun 2023 once (the part-time job overlaps the first),
        # then Sep 2023 - Feb 2024: 30 + 6 months.
        expected_experience_years=3.0,
        expected_education_levels=frozenset({"UG"}),
        relevant=False,
    ),
    SampleResume(
        key="biotech_masters",
        filename="nandini_rao_biotech.docx",
        lines=(
            "Nandini Rao",
            "nandini.rao@college.edu",
            "",
            "SUMMARY",
            "Biotechnology postgraduate student with 6 months of research",
            "internship experience in protein purification and cell culture.",
            "",
            "EDUCATION",
            "M.Sc in Biotechnology, 2026, CGPA 8.6/10",
            "B.Sc in Microbiology, 2024",
            "",
            "EXPERIENCE",
            "Research Intern, GenePath Labs",
            "Purified recombinant proteins and ran SDS-PAGE and western blots.",
            "",
            "SKILLS",
            "Cell culture, PCR, SDS-PAGE, chromatography, ELISA, lab safety",
        ),
        expected_experience_years=0.5,
        expected_education_levels=frozenset({"Masters", "UG"}),
        relevant=False,
        expected_cgpa=8.6,
    ),
    SampleResume(
        key="designer_short_dates",
        filename="tanvi_joshi_design.pdf",
        lines=(
            "Tanvi Joshi",
            "tanvi.joshi@college.edu",
            "",
            "SUMMARY",
            "Visual designer who makes brand identities, packaging and posters",
            "for small consumer brands.",
            "",
            "WORK EXPERIENCE",
            "Graphic Design Intern, Pixel Forge (Jan'23 - Jun'23)",
            "Visual Designer, Lumen Media (Aug'23 - Mar'24)",
            "",
            "EDUCATION",
            "Bachelor of Design in Communication Design, 2026",
            "",
            "SKILLS",
            "Illustrator, Photoshop, typography, brand identity, packaging",
        ),
        # 6 months, then 8: 14 months.
        expected_experience_years=1.2,
        expected_education_levels=frozenset({"UG"}),
        relevant=False,
    ),
)

SAMPLES_BY_KEY = {sample.key: sample for sample in SAMPLE_RESUMES}
RELEVANT_KEYS = frozenset(s.key for s in SAMPLE_RESUMES if s.relevant)


def write_pdf(path, lines) -> None:
    """Render lines into a simple PDF (needs `reportlab`)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    width, height = A4
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10.5)
    y = height - 60
    for line in lines:
        if y < 60:
            pdf.showPage()
            pdf.setFont("Helvetica", 10.5)
            y = height - 60
        pdf.drawString(56, y, line)
        y -= 15
    pdf.save()


def write_docx(path, lines) -> None:
    """Render lines into a DOCX file (needs `python-docx`)."""
    import docx

    document = docx.Document()
    for line in lines:
        document.add_paragraph(line)
    document.save(str(path))


def render_samples(out_dir) -> list:
    """Write every sample resume into `out_dir` as PDF or DOCX. Returns the paths."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for sample in SAMPLE_RESUMES:
        path = out_dir / sample.filename
        (write_pdf if sample.format == "pdf" else write_docx)(path, list(sample.lines))
        written.append(path)
    (out_dir / "job_description.txt").write_text(SAMPLE_JOB_DESCRIPTION, encoding="utf-8")
    return written
