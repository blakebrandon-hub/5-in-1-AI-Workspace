import json
import os
import re
import threading
from datetime import datetime
import zipfile
from io import BytesIO, StringIO
import uuid
import csv
import time
import sqlite3
import html as html_module
import logging

from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from openai import OpenAI
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import PyPDF2

import numpy as np
from sentence_transformers import SentenceTransformer

import requests
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Configuration & Globals
# ---------------------------------------------------

MODEL = "gpt-5.4"
MAX_ITERATIONS_CODE = 5
MAX_ITERATIONS_ESSAY = 3
MAX_ITERATIONS_RESUME = 3
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "your-secret-key-change-in-production"
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Job Stores
code_jobs = {}
essay_jobs = {}
resume_jobs = {}

# RAG Memory Store
embedding_model = None
memory_store = []
is_model_ready = False

# ---------------------------------------------------
# Core AI & RAG Utilities
# ---------------------------------------------------

def call_llm(system_prompt, user_prompt, temperature=0.3):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    return response.choices[0].message.content.strip()

class Agent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt

    def run(self, prompt, temperature=0.3):
        return call_llm(self.system_prompt, prompt, temperature)

def initialize_model():
    global embedding_model, is_model_ready
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        is_model_ready = True
        print("Model loaded successfully")

def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b)

def retrieve_memory(query_text):
    """Query the internal RAG store for relevant context to inject into agents."""
    if not is_model_ready or not memory_store:
        return ""
    try:
        query_vector = embedding_model.encode(query_text, normalize_embeddings=True)
        results = []
        for mem in memory_store:
            mem_vector = np.array(mem['vector'])
            score = float(cosine_similarity(query_vector, mem_vector))
            results.append((score, mem['text']))
        
        results.sort(key=lambda x: x[0], reverse=True)
        relevant = [text for score, text in results if score > 0.40][:3]
        if relevant:
            return "\n[Retrieved Context from Vector Store]:\n" + "\n---\n".join(relevant) + "\n"
        return ""
    except Exception as e:
        print(f"Memory retrieval error: {e}")
        return ""

# ---------------------------------------------------
# App 1: Code Generator Agents
# ---------------------------------------------------

code_planner = Agent("code_planner", """
You are a planning agent. Break the user's goal into a list of concrete tasks.
Return JSON ONLY in this format:
{
  "tasks": ["task1", "task2", "task3"]
}
""")

code_researcher = Agent("code_researcher", """
You are a research agent. Gather relevant knowledge needed to complete the tasks.
Return structured information that will help produce the final result.
Focus on facts, explanations, and useful data.
""")

code_executor = Agent("code_executor", """
You are an execution agent that generates actual code files.
When the goal involves building apps, websites, or code:
- Generate COMPLETE, WORKING code
- Use code blocks with filenames like: ```filename.ext
- Each file should be in its own code block
- Do NOT write tutorials or explanations - write actual code
""")

code_critic = Agent("code_critic", """
You are a critic agent. Evaluate whether the output satisfies the goal.
- Check if actual code files were generated (look for ```filename blocks)
- Verify the code is complete and functional
Return JSON ONLY:
{
  "pass": true or false,
  "feedback": "explanation"
}
Pass = true if the output fully satisfies the goal.
Pass = false if code is missing, incomplete, or if a tutorial was provided instead of code.
""")

def extract_and_save_files(result, job_id):
    pattern = r'```(\S+)\n(.*?)```'
    matches = re.findall(pattern, result, re.DOTALL)
    if not matches:
        return []
    output_folder = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(output_folder, exist_ok=True)
    saved_files = []
    for filename, content in matches:
        if '.' not in filename: continue
        filepath = os.path.join(output_folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        saved_files.append(filepath)
    return saved_files

def run_code_agent(goal, job_id):
    state = {
        "goal": goal, "tasks": [], "research": "", "result": "", "feedback": "",
        "files": [], "status": "running", "current_iteration": 0, "logs": []
    }
    code_jobs[job_id] = state

    try:
        for step in range(MAX_ITERATIONS_CODE):
            state["current_iteration"] = step + 1
            state["logs"].append(f"Starting iteration {step + 1}")

            memory = retrieve_memory(state["goal"])

            # Planner
            planner_prompt = f"Goal:\n{state['goal']}\nMemory:\n{memory}"
            state["logs"].append("Running planner...")
            try:
                plan_response = code_planner.run(planner_prompt, temperature=0.2)
                data = json.loads(plan_response)
                state["tasks"] = data["tasks"]
                state["logs"].append(f"Tasks identified: {len(data['tasks'])}")
            except:
                state["logs"].append("Planner JSON parse failed")
                continue

            # Researcher
            research_prompt = f"Tasks:\n{state['tasks']}\nMemory Context:\n{memory}"
            state["logs"].append("Running researcher...")
            state["research"] = code_researcher.run(research_prompt, temperature=0.2)

            # Executor
            execute_prompt = f"Goal:\n{state['goal']}\nTasks:\n{state['tasks']}\nResearch:\n{state['research']}"
            state["logs"].append("Running executor...")
            state["result"] = code_executor.run(execute_prompt, temperature=0.2)

            # Extraction
            state["logs"].append("Extracting files...")
            state["files"] = extract_and_save_files(state["result"], job_id)
            state["logs"].append(f"Saved {len(state['files'])} file(s)")

            # Critic
            critic_prompt = f"Goal:\n{state['goal']}\nResult:\n{state['result']}\nFiles generated: {len(state['files'])}"
            state["logs"].append("Running critic...")
            try:
                review = code_critic.run(critic_prompt, temperature=0.2)
                review_data = json.loads(review)
            except:
                state["logs"].append("Critic JSON parse failed")
                continue

            if review_data["pass"]:
                state["logs"].append("✓ Critic approved the result")
                state["status"] = "completed"
                state["feedback"] = review_data["feedback"]
                return
            else:
                state["logs"].append(f"✗ Critic feedback: {review_data['feedback']}")
                state["feedback"] = review_data["feedback"]
                state["goal"] += f"\n\nImprove the result using this feedback:\n{review_data['feedback']}"

        state["status"] = "completed"
        state["logs"].append("Reached iteration limit")
        
    except Exception as e:
        state["status"] = "error"
        state["logs"].append(f"Error: {str(e)}")


# ---------------------------------------------------
# App 2: Essay Generator Agents
# ---------------------------------------------------

essay_planner = Agent("essay_planner", """
You are an essay planning agent. Create a structured outline.
Return JSON ONLY:
{
  "outline": ["Introduction: ...", "Body 1: ...", "Body 2: ...", "Conclusion: ..."]
}
""")

essay_researcher = Agent("essay_researcher", """
You are a research agent for academic essays. Gather relevant information, arguments, and evidence.
Provide structured research findings.
""")

essay_writer = Agent("essay_writer", """
You are an essay writing agent. Write a complete, well-structured academic essay.
- Use proper paragraphs and transitions
- Include an introduction, body paragraphs, and conclusion
- Write in formal academic style
""")

essay_critic = Agent("essay_critic", """
You are an essay critic. Evaluate the essay quality.
Return JSON ONLY:
{
  "pass": true or false,
  "feedback": "detailed critique"
}
Pass = true if the essay is complete and well-written.
""")

def run_essay_agent(prompt, job_id):
    state = {
        "prompt": prompt, "outline": [], "research": "", "essay": "", "feedback": "",
        "status": "running", "current_iteration": 0, "logs": []
    }
    essay_jobs[job_id] = state

    try:
        for step in range(MAX_ITERATIONS_ESSAY):
            state["current_iteration"] = step + 1
            state["logs"].append(f"Starting iteration {step + 1}")

            memory = retrieve_memory(state["prompt"])

            # Planner
            state["logs"].append("Creating outline...")
            try:
                plan_response = essay_planner.run(f"Topic:\n{state['prompt']}\nMemory:\n{memory}", temperature=0.3)
                data = json.loads(plan_response)
                state["outline"] = data["outline"]
                state["logs"].append(f"Outline created with {len(data['outline'])} sections")
            except:
                state["logs"].append("Planner JSON parse failed")
                continue

            # Researcher
            state["logs"].append("Researching topic...")
            state["research"] = essay_researcher.run(f"Topic:\n{state['prompt']}\nOutline:\n{state['outline']}", temperature=0.3)

            # Writer
            state["logs"].append("Writing essay...")
            writer_prompt = f"Topic:\n{state['prompt']}\nOutline:\n{state['outline']}\nResearch:\n{state['research']}"
            state["essay"] = essay_writer.run(writer_prompt, temperature=0.4)

            # Critic
            state["logs"].append("Reviewing essay...")
            try:
                review = essay_critic.run(f"Topic:\n{state['prompt']}\nEssay:\n{state['essay']}", temperature=0.2)
                review_data = json.loads(review)
            except:
                state["logs"].append("Critic JSON parse failed")
                continue

            if review_data["pass"]:
                state["logs"].append("✓ Essay approved")
                state["status"] = "completed"
                state["feedback"] = review_data["feedback"]
                return
            else:
                state["logs"].append(f"✗ Feedback: {review_data['feedback']}")
                state["feedback"] = review_data["feedback"]
                state["prompt"] += f"\n\nRevise based on this feedback:\n{review_data['feedback']}"

        state["status"] = "completed"
        state["logs"].append("Reached iteration limit")
        
    except Exception as e:
        state["status"] = "error"
        state["logs"].append(f"Error: {str(e)}")


# ---------------------------------------------------
# App 3: Resume Builder
# ---------------------------------------------------

resume_analyzer = Agent("resume_analyzer", """
You are a resume analysis agent. Analyze the job description and extract key requirements.
Return JSON ONLY:
{
  "key_skills": ["skill1", "skill2"],
  "requirements": ["req1", "req2"],
  "priorities": ["what to emphasize"]
}
""")

resume_writer = Agent("resume_writer", """
You are a professional resume writer. Create an ATS-optimized resume.
- Tailor content to the job description
- Use action verbs and quantified achievements
- Format for ATS parsing
- Keep it concise and impactful
""")

resume_critic = Agent("resume_critic", """
You are a resume critic. Evaluate resume quality and ATS compatibility.
Return JSON ONLY:
{
  "pass": true or false,
  "feedback": "detailed critique"
}
""")

def create_resume_docx(resume_text, job_id):
    """Create a Word document from resume text"""
    doc = Document()
    
    # Set narrow margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)
    
    # Add content
    for line in resume_text.split('\n'):
        if line.strip():
            p = doc.add_paragraph(line)
            if line.isupper() or line.startswith('###'):
                p.runs[0].bold = True
                p.runs[0].font.size = Pt(14)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f"resume_{job_id}.docx")
    doc.save(output_path)
    return output_path

def run_resume_agent(job_description, current_resume, job_id):
    state = {
        "job_description": job_description,
        "current_resume": current_resume,
        "analysis": {},
        "resume": "",
        "feedback": "",
        "docx_path": None,
        "status": "running",
        "current_iteration": 0,
        "logs": []
    }
    resume_jobs[job_id] = state

    try:
        for step in range(MAX_ITERATIONS_RESUME):
            state["current_iteration"] = step + 1
            state["logs"].append(f"Starting iteration {step + 1}")

            # Analyzer
            state["logs"].append("Analyzing job description...")
            try:
                analysis_response = resume_analyzer.run(
                    f"Job Description:\n{state['job_description']}", 
                    temperature=0.2
                )
                state["analysis"] = json.loads(analysis_response)
                state["logs"].append("Analysis complete")
            except:
                state["logs"].append("Analyzer JSON parse failed")
                continue

            # Writer
            state["logs"].append("Tailoring resume...")
            writer_prompt = f"""
            Job Description:\n{state['job_description']}
            
            Current Resume:\n{state['current_resume']}
            
            Analysis:\n{json.dumps(state['analysis'], indent=2)}
            
            Create a tailored resume that highlights relevant experience and skills.
            """
            state["resume"] = resume_writer.run(writer_prompt, temperature=0.3)

            # Critic
            state["logs"].append("Reviewing resume...")
            try:
                review = resume_critic.run(
                    f"Job Description:\n{state['job_description']}\n\nResume:\n{state['resume']}", 
                    temperature=0.2
                )
                review_data = json.loads(review)
            except:
                state["logs"].append("Critic JSON parse failed")
                continue

            if review_data["pass"]:
                state["logs"].append("✓ Resume approved")
                state["status"] = "completed"
                state["feedback"] = review_data["feedback"]
                
                # Create DOCX
                state["logs"].append("Creating Word document...")
                state["docx_path"] = create_resume_docx(state["resume"], job_id)
                return
            else:
                state["logs"].append(f"✗ Feedback: {review_data['feedback']}")
                state["feedback"] = review_data["feedback"]

        state["status"] = "completed"
        state["logs"].append("Reached iteration limit")
        
        # Create DOCX even if not perfect
        if state["resume"]:
            state["docx_path"] = create_resume_docx(state["resume"], job_id)
        
    except Exception as e:
        state["status"] = "error"
        state["logs"].append(f"Error: {str(e)}")


# ---------------------------------------------------
# App 4: RAG Vector Store
# ---------------------------------------------------

@app.route('/api/rag/initialize', methods=['POST'])
def initialize_rag():
    initialize_model()
    return jsonify({'status': 'success' if is_model_ready else 'error'})

@app.route('/api/rag/store', methods=['POST'])
def store_rag():
    if not is_model_ready:
        return jsonify({'status': 'error', 'message': 'Model not ready'})
    
    data = request.json
    text = data.get('text', '')
    metadata = data.get('metadata', {})
    
    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided'})
    
    try:
        vector = embedding_model.encode(text, normalize_embeddings=True)
        memory_id = str(uuid.uuid4())
        
        memory_store.append({
            'id': memory_id,
            'text': text,
            'vector': vector.tolist(),
            'metadata': metadata,
            'timestamp': time.time()
        })
        
        return jsonify({'status': 'success', 'id': memory_id})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/rag/query', methods=['POST'])
def query_rag():
    if not is_model_ready:
        return jsonify({'status': 'error', 'message': 'Model not ready'})
    
    query = request.json.get('query', '')
    if not query:
        return jsonify({'status': 'error', 'message': 'No query provided'})
    
    try:
        query_vector = embedding_model.encode(query, normalize_embeddings=True)
        results = []
        
        for mem in memory_store:
            mem_vector = np.array(mem['vector'])
            score = float(cosine_similarity(query_vector, mem_vector))
            results.append({
                'text': mem['text'],
                'score': score,
                'metadata': mem.get('metadata', {}),
                'id': mem['id']
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = [r for r in results if r['score'] > 0.3][:5]
        
        return jsonify({'status': 'success', 'results': top_results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/rag/list')
def list_rag():
    memories = [
        {
            'id': m['id'],
            'text': m['text'][:200] + ('...' if len(m['text']) > 200 else ''),
            'timestamp': m['timestamp'],
            'metadata': m.get('metadata', {})
        }
        for m in memory_store
    ]
    return jsonify({'count': len(memory_store), 'memories': memories})

@app.route('/api/rag/delete/<memory_id>', methods=['DELETE'])
def delete_rag(memory_id):
    global memory_store
    memory_store = [m for m in memory_store if m['id'] != memory_id]
    return jsonify({'status': 'success', 'message': 'Memory deleted'})

@app.route('/api/rag/clear', methods=['POST'])
def clear_rag():
    global memory_store
    memory_store = []
    return jsonify({'status': 'success', 'message': 'Memory store cleared'})


# ---------------------------------------------------
# App 5: AI Job Search - COMPLETE IMPLEMENTATION
# ---------------------------------------------------

JOB_DB = os.path.join(OUTPUT_DIR, "jobs.db")

JOB_TITLES = [
    "AI Engineer",
    "Machine Learning Engineer",
    "Applied AI Engineer",
    "Backend Engineer AI",
    "Backend Python Engineer",
    "Software Engineer AI",
    "Software Engineer Machine Learning",
    "Full Stack Engineer AI",
    "Product Engineer AI",
    "Software Engineer",
    "Backend Engineer"
]

GOOD_KEYWORDS = [
    "llm", "machine learning", "ai engineer",
    "rag", "nlp", "deep learning", "python",
    "ai platform", "ai infrastructure", "genai",
    "ml engineer", "mlops"
]

BAD_KEYWORDS = [
    "trainer", "training sales", "customer success",
    "advisor", "support rep", "marketing coordinator",
    "writer", "translation", "data entry", "annotator"
]

EXCLUDE_SENIOR = [
    "principal", "staff engineer", "director", "vp", "chief",
    "head of", "lead engineer"
]

IRRELEVANT_ROLES = [
    "qa tester", "manual test",
    "electrical engineer", "hardware engineer",
    "sales engineer", "solutions engineer",
    "recruiter", "technical recruiter"
]

HIGH_SIGNAL_COMPANIES = [
    "openai", "anthropic", "google deepmind",
    "modal", "replicate", "cohere", "hugging face",
    "scale ai", "cursor", "perplexity"
]

# Database functions
def get_job_connection():
    conn = sqlite3.connect(JOB_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_job_db():
    conn = get_job_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        company TEXT NOT NULL,
        location TEXT,
        url TEXT UNIQUE NOT NULL,
        status TEXT DEFAULT 'new',
        score INTEGER DEFAULT 0,
        source TEXT,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_score ON jobs(score DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON jobs(created_at DESC)")
    conn.commit()
    conn.close()

def insert_jobs(jobs):
    conn = get_job_connection()
    cursor = conn.cursor()
    inserted = 0
    for job in jobs:
        try:
            cursor.execute("""
            INSERT INTO jobs (title, company, location, url, score, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                job.get("title", ""),
                job.get("company", ""),
                job.get("location", ""),
                job.get("url", ""),
                job.get("score", 0),
                job.get("source", "Unknown")
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    return inserted

def get_jobs(status_filter="all", search_query="", sort_by="date"):
    conn = get_job_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM jobs WHERE 1=1"
    params = []
    
    if status_filter != "all":
        query += " AND status = ?"
        params.append(status_filter)
    
    if search_query:
        query += " AND (title LIKE ? OR company LIKE ? OR location LIKE ?)"
        search_param = f"%{search_query}%"
        params.extend([search_param, search_param, search_param])
    
    sort_options = {
        "date": "created_at DESC",
        "score": "score DESC, created_at DESC",
        "company": "company ASC",
        "title": "title ASC"
    }
    query += f" ORDER BY {sort_options.get(sort_by, 'created_at DESC')}"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def update_status(job_id, status):
    conn = get_job_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE jobs SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", 
        (status, job_id)
    )
    conn.commit()
    conn.close()

def update_notes(job_id, notes):
    conn = get_job_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE jobs SET notes=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (notes, job_id)
    )
    conn.commit()
    conn.close()

def delete_job(job_id):
    conn = get_job_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    conn.commit()
    conn.close()

def mark_all_seen():
    conn = get_job_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE jobs SET status='seen', updated_at=CURRENT_TIMESTAMP WHERE status='new'"
    )
    conn.commit()
    conn.close()

def get_stats():
    conn = get_job_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as total FROM jobs")
    total = cursor.fetchone()["total"]
    
    cursor.execute("""
        SELECT status, COUNT(*) as count 
        FROM jobs 
        GROUP BY status 
        ORDER BY count DESC
    """)
    by_status = {row["status"]: row["count"] for row in cursor.fetchall()}
    
    cursor.execute("""
        SELECT source, COUNT(*) as count 
        FROM jobs 
        GROUP BY source 
        ORDER BY count DESC
    """)
    by_source = {row["source"]: row["count"] for row in cursor.fetchall()}
    
    cursor.execute("""
        SELECT company, COUNT(*) as count 
        FROM jobs 
        GROUP BY company 
        ORDER BY count DESC 
        LIMIT 10
    """)
    top_companies = [(row["company"], row["count"]) for row in cursor.fetchall()]
    
    cursor.execute("""
        SELECT COUNT(*) as count 
        FROM jobs 
        WHERE score >= 5
    """)
    high_score = cursor.fetchone()["count"]
    
    cursor.execute("""
        SELECT COUNT(*) as count 
        FROM jobs 
        WHERE created_at >= datetime('now', '-7 days')
    """)
    recent = cursor.fetchone()["count"]
    
    conn.close()
    
    return {
        "total": total,
        "by_status": by_status,
        "by_source": by_source,
        "top_companies": top_companies,
        "high_score": high_score,
        "recent": recent
    }

def export_jobs():
    conn = get_job_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    output = StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    
    return output.getvalue()

# Job search functions
def llm_call_jobs(system: str, user: str) -> str:
    """Make LLM call for job search"""
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return response.output_text

def generate_queries(job_titles: List[str]) -> List[str]:
    """Generate search queries from job titles"""
    result = llm_call_jobs(
        "You generate concise job search queries for job boards. Return 5 short queries, one per line.",
        f"Create job search queries from these titles: {', '.join(job_titles[:5])}"
    )
    return [q.strip("-• ").strip() for q in result.split("\n") if q.strip()]

def is_relevant(title: str, company: str, filters: Dict) -> bool:
    """Determine if job is relevant based strictly on UI filters"""
    t = title.lower()
    
    bad_keywords = filters.get("badKeywords", [])
    senior_keywords = filters.get("seniorKeywords", [])
    
    # Drop if it matches a bad keyword
    if any(bad in t for bad in bad_keywords):
        return False
    
    # Drop if it matches an excluded seniority level
    if any(senior in t for senior in senior_keywords):
        return False
    
    # If it passes exclusions, it stays. The scoring system will sort the best to the top.
    return True

def score_job(job: Dict, filters: Dict) -> int:
    """Score job relevance strictly based on UI filters"""
    score = 0
    title = (job.get("title") or "").lower()
    company = (job.get("company") or "").lower()
    
    high_signal_companies = filters.get("highSignalCompanies", [])
    good_keywords = filters.get("goodKeywords", [])
    
    # +5 points for High Signal Companies
    if any(c in company for c in high_signal_companies):
        score += 5
    
    # +2 points for EVERY good keyword matched in the title
    keyword_matches = sum(1 for k in good_keywords if k in title)
    score += (keyword_matches * 2)
    
    return max(0, score)

def search_remoteok(query: str, filters: Dict) -> List[Dict]:
    """Search RemoteOK API"""
    url = "https://remoteok.com/api"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"RemoteOK search failed: {e}")
        return []
    
    jobs = []
    for job in data[1:100]:
        title = job.get("position", "")
        company = job.get("company", "")
        
        if not title or not is_relevant(title, company, filters):
            continue
        
        jobs.append({
            "title": title,
            "company": company,
            "location": "Remote",
            "url": job.get("url", ""),
            "source": "RemoteOK"
        })
    return jobs

def search_linkedin_rss(job_titles: List[str], filters: Dict) -> List[Dict]:
    """Search LinkedIn jobs via jobs-guest API endpoint"""
    jobs = []
    
    for query in job_titles[:3]:
        try:
            url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
            params = {"keywords": query, "start": 0}
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code != 200:
                continue
            
            job_sections = re.findall(r'<li>.*?</li>', response.text, re.DOTALL)
            
            for section in job_sections[:20]:
                title_match = re.search(r'base-search-card__title[^>]*>([^<]+)<', section)
                company_match = re.search(r'base-search-card__subtitle[^>]*>([^<]+)<', section)
                url_match = re.search(r'href="(https://[^"]*linkedin\.com/jobs/view/[^"]*)"', section)
                
                if not title_match or not url_match:
                    continue
                
                title = html_module.unescape(re.sub(r'<[^>]+>', '', title_match.group(1))).strip()
                company = html_module.unescape(re.sub(r'<[^>]+>', '', company_match.group(1))).strip() if company_match else "LinkedIn Company"
                job_url = url_match.group(1)
                
                if not is_relevant(title, company, filters):
                    continue
                
                jobs.append({
                    "title": title,
                    "company": company,
                    "location": "LinkedIn",
                    "url": job_url,
                    "source": "LinkedIn"
                })
            time.sleep(3)
        except Exception as e:
            logger.error(f"LinkedIn search failed: {e}")
    return jobs

def search_hn_hiring(filters: Dict) -> List[Dict]:
    """Search Hacker News API for Who's Hiring thread"""
    try:
        search_url = "https://hn.algolia.com/api/v1/search"
        params = {"query": "Who is hiring", "tags": "story", "hitsPerPage": 1}
        response = requests.get(search_url, params=params, timeout=10)
        data = response.json()
        
        if not data.get("hits"): return []
        
        story_id = data["hits"][0]["objectID"]
        comments_url = f"https://hn.algolia.com/api/v1/items/{story_id}"
        thread_data = requests.get(comments_url, timeout=10).json()
        
        jobs = []
        for comment in thread_data.get("children", [])[:100]:
            text = comment.get("text", "")
            if not text or len(text) < 50: continue
            
            first_line = text.split('\n')[0].strip()
            first_line = html_module.unescape(re.sub(r'<[^>]+>', '', first_line))
            
            parts = [p.strip() for p in first_line.split('|')]
            company = parts[0] if len(parts) > 0 else "HN Company"
            title = parts[1] if len(parts) > 1 else first_line[:100]
            
            if not is_relevant(title, company, filters):
                continue
            
            jobs.append({
                "title": title,
                "company": company,
                "location": "Unknown",
                "url": f"https://news.ycombinator.com/item?id={comment.get('id')}",
                "source": "HN"
            })
        return jobs
    except Exception as e:
        logger.error(f"HN API search failed: {e}")
        return []

def run_search(job_titles: List[str], progress_callback=None, filters=None) -> List[Dict]:
    """Main search orchestrator"""
    if filters is None:
        filters = {}
        
    queries = generate_queries(job_titles)
    all_jobs = []
    
    # RemoteOK
    for idx, query in enumerate(queries[:3]):
        all_jobs.extend(search_remoteok(query, filters))
        time.sleep(2)
    
    # Hacker News
    all_jobs.extend(search_hn_hiring(filters))
    
    # LinkedIn
    all_jobs.extend(search_linkedin_rss(job_titles, filters))
    
    # Score & Dedup
    for job in all_jobs:
        job["score"] = score_job(job, filters)
    
    seen = set()
    unique = []
    for job in all_jobs:
        url = job.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(job)
    
    unique.sort(key=lambda x: x.get("score", 0), reverse=True)
    return unique

# ---------------------------------------------------
# Chatbot Data & Route
# ---------------------------------------------------

CHATBOT_DATA = {
    "about": {
        "name": "Blake Brandon",
        "role": "AI Systems Engineer",
        "focus": [
            "LLM orchestration",
            "RAG systems",
            "Multi-agent pipelines"
        ]
    },
    "projects": [
        {
            "name": "Multi-Agent Code Generator",
            "description": "AI-powered code generation with autonomous planning, research, and execution",
            "architecture": [
                "Planner agent for task decomposition",
                "Researcher agent for context gathering",
                "Executor agent for code generation",
                "Critic agent for iterative refinement",
                "RAG integration for contextual memory"
            ]
        },
        {
            "name": "AI Essay Writer",
            "description": "Multi-agent essay generation system with iterative improvement",
            "architecture": [
                "Planner for outline generation",
                "Researcher for information gathering",
                "Writer agent for content creation",
                "Critic for quality validation",
                "Feedback loop for refinement"
            ]
        },
        {
            "name": "RAG Vector Store",
            "description": "Retrieval-augmented generation with semantic search",
            "architecture": [
                "Sentence transformers for embeddings",
                "Cosine similarity for retrieval",
                "Vector store with normalized embeddings",
                "Context injection for agent enhancement"
            ]
        },
        {
            "name": "AI Job Search Tracker",
            "description": "Automated job discovery with AI-powered scoring and tracking",
            "architecture": [
                "Web scraping for job aggregation",
                "AI scoring for relevance ranking",
                "SQLite for persistence",
                "Status tracking and filtering system"
            ]
        }
    ]
}

def get_context(question):
    q = question.lower()

    if "workspace" in q:
        return CHATBOT_DATA["projects"][0]
    if "crm" in q:
        return CHATBOT_DATA["projects"][1]
    if "rpg" in q or "sprawl" in q:
        return CHATBOT_DATA["projects"][2]
    if "skill" in q or "experience" in q:
        return CHATBOT_DATA["about"]

    return CHATBOT_DATA

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    context = get_context(user_message)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""
You are an AI assistant representing Blake Brandon.

Be concise, structured, and technical.
Explain systems using architecture and decisions.

Context:
{context}
"""
            },
            {"role": "user", "content": user_message}
        ]
    )

    return jsonify({
        "reply": completion.choices[0].message.content
    })

# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

# Code Generator Routes
@app.route('/api/code/generate', methods=['POST'])
def generate_code():
    goal = request.json.get('goal', '')
    if not goal:
        return jsonify({'error': 'No goal provided'}), 400
    
    job_id = str(uuid.uuid4())
    thread = threading.Thread(target=run_code_agent, args=(goal, job_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route('/api/code/status/<job_id>')
def code_status(job_id):
    if job_id in code_jobs:
        return jsonify(code_jobs[job_id])
    return jsonify({'status': 'not_found'})

@app.route('/api/code/download/<job_id>')
def download_code(job_id):
    if job_id not in code_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = code_jobs[job_id]
    if not job.get('files'):
        return jsonify({'error': 'No files generated'}), 404
    
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for filepath in job['files']:
            arcname = os.path.basename(filepath)
            zf.write(filepath, arcname)
    
    memory_file.seek(0)
    return send_file(memory_file, download_name=f'code_{job_id}.zip', as_attachment=True)

# Essay Generator Routes
@app.route('/api/essay/generate', methods=['POST'])
def generate_essay():
    prompt = request.json.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    job_id = str(uuid.uuid4())
    thread = threading.Thread(target=run_essay_agent, args=(prompt, job_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route('/api/essay/status/<job_id>')
def essay_status(job_id):
    if job_id in essay_jobs:
        return jsonify(essay_jobs[job_id])
    return jsonify({'status': 'not_found'})

# Resume Builder Routes
@app.route('/api/resume/generate', methods=['POST'])
def generate_resume():
    job_description = request.json.get('job_description', '')
    current_resume = request.json.get('current_resume', '')
    
    if not job_description or not current_resume:
        return jsonify({'error': 'Job description and current resume required'}), 400
    
    job_id = str(uuid.uuid4())
    thread = threading.Thread(target=run_resume_agent, args=(job_description, current_resume, job_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route('/api/resume/status/<job_id>')
def resume_status(job_id):
    if job_id in resume_jobs:
        return jsonify(resume_jobs[job_id])
    return jsonify({'status': 'not_found'})

@app.route('/api/resume/download/<job_id>')
def download_resume(job_id):
    if job_id not in resume_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = resume_jobs[job_id]
    if not job.get('docx_path'):
        return jsonify({'error': 'Resume not ready'}), 404
    
    return send_file(job['docx_path'], as_attachment=True)

@app.route('/api/resume/text/<job_id>')
def resume_text(job_id):
    if job_id in resume_jobs:
        return jsonify({'resume': resume_jobs[job_id].get('resume', '')})
    return jsonify({'resume': ''})

# Job Search Routes
@app.route('/api/jobs/search', methods=['POST'])
def api_job_search():
    try:
        # Capture the dynamic filters sent from the UI
        filters = request.json or {}
        progress_file = os.path.join(OUTPUT_DIR, 'search_progress.json')
        
        def update_progress(message):
            with open(progress_file, 'w') as f:
                json.dump({'status': 'searching', 'message': message}, f)
        
        update_progress('Starting search...')
        
        # Pass filters into the search orchestrator
        jobs = run_search(JOB_TITLES, progress_callback=update_progress, filters=filters)
        
        with open(progress_file, 'w') as f:
            json.dump({'status': 'complete', 'message': f'Found {len(jobs)} jobs'}, f)
        
        count = insert_jobs(jobs)
        
        return jsonify({'status': 'success', 'found': len(jobs), 'inserted': count})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/jobs/progress')
def api_job_search_progress():
    try:
        progress_file = os.path.join(OUTPUT_DIR, 'search_progress.json')
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({'status': 'idle', 'message': ''})
    except:
        return jsonify({'status': 'idle', 'message': ''})

@app.route('/api/jobs/list')
def api_jobs_list():
    status_filter = request.args.get('status', 'all')
    search_query = request.args.get('q', '')
    sort_by = request.args.get('sort', 'date')
    jobs = get_jobs(status_filter, search_query, sort_by)
    stats = get_stats()
    return jsonify({'jobs': jobs, 'stats': stats})

@app.route('/api/jobs/update/<int:job_id>/<status>', methods=['POST'])
def api_job_update(job_id, status):
    valid_statuses = ["new", "interested", "applied", "interview", "offer", "rejected", "ignored", "seen"]
    if status in valid_statuses:
        update_status(job_id, status)
        return jsonify({'success': True})
    return jsonify({'error': 'Invalid status'}), 400

@app.route('/api/jobs/notes/<int:job_id>', methods=['POST'])
def api_job_notes(job_id):
    notes = request.json.get('notes', '')
    update_notes(job_id, notes)
    return jsonify({'success': True})

@app.route('/api/jobs/delete/<int:job_id>', methods=['DELETE'])
def api_job_delete(job_id):
    delete_job(job_id)
    return jsonify({'success': True})

@app.route('/api/jobs/bulk/seen', methods=['POST'])
def api_bulk_seen():
    mark_all_seen()
    return jsonify({'success': True})

@app.route('/api/jobs/add', methods=['POST'])
def api_job_add():
    data = request.json
    title = data.get('title', '').strip()
    company = data.get('company', '').strip()
    location = data.get('location', '').strip()
    url = data.get('url', '').strip()
    
    if not title or not company or not url:
        return jsonify({'error': 'Title, company, and URL are required'}), 400
    
    inserted = insert_jobs([{
        "title": title,
        "company": company,
        "location": location,
        "url": url,
        "source": "Manual",
        "score": 0
    }])
    
    return jsonify({'success': True, 'inserted': inserted})

@app.route('/api/jobs/export')
def api_job_export():
    csv_data = export_jobs()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=jobs_export.csv"}
    )

# Initialize
init_job_db()

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Unified AI Workspace - 5 Apps in One")
    print("   💻 Code Generator | 📝 Essay Writer | 📄 Resume Builder")
    print("   🧠 RAG Vector Store | 🔍 AI Job Search")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
