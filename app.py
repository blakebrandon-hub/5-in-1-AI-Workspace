import json
import os
import re
import threading
from datetime import datetime
import zipfile
from io import BytesIO
import uuid

from flask import Flask, render_template, request, jsonify, send_file, after_this_request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from openai import OpenAI
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import PyPDF2

import numpy as np
from sentence_transformers import SentenceTransformer

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
CORS(app, resources={r"/*": {"origins": "*"}})
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



@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


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
You are a research planning agent for academic technical essays.
Given the essay topic and requirements, break down the research into specific, actionable research tasks.
Return JSON ONLY in this format:
{
  "research_tasks": ["Research task 1", "Research task 2"],
  "essay_structure": ["Introduction", "Section 1", "Conclusion"]
}
""")

essay_researcher = Agent("essay_researcher", """
You are an expert research agent specializing in technical and academic topics.
For each research task provided, conduct thorough research and provide technical details, facts, expert perspectives, and examples.
Organize your research clearly with headings.
""")

essay_outliner = Agent("essay_outliner", """
You are an academic writing expert that creates detailed essay outlines.
Based on the topic, research findings, and suggested structure, create a comprehensive outline.
Return a detailed outline in markdown format with clear hierarchical structure.
""")

essay_writer = Agent("essay_writer", """
You are an expert academic writer specializing in technical essays at the college level.
Write a complete, polished technical essay based on the outline and research provided.
1500-2500 words typical length. Format with clear section headings using markdown.
Write the COMPLETE essay, not an outline or summary.
""")

essay_editor = Agent("essay_editor", """
You are an academic editor and writing instructor.
Evaluate the essay for thesis clarity, organization, evidence, technical accuracy, writing quality, and depth.
Return JSON ONLY:
{
  "pass": true or false,
  "score": {
    "thesis": 0-10, "organization": 0-10, "evidence": 0-10, 
    "technical_accuracy": 0-10, "writing_quality": 0-10, "depth": 0-10
  },
  "strengths": ["strength 1"],
  "improvements_needed": ["improvement 1"],
  "feedback": "overall assessment"
}
Pass = true only if the essay is well-developed, well-researched, and meets college standards.
""")

def create_word_document(essay_content, metadata, job_id):
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
    
    for line in essay_content.split('\n'):
        line = line.strip()
        if not line: continue
        
        if line.startswith('# '):
            p = doc.add_paragraph(line[2:].strip())
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.runs[0].font.size = Pt(16)
            p.runs[0].font.bold = True
            doc.add_paragraph()
        elif line.startswith('## '):
            p = doc.add_paragraph(line[3:].strip())
            p.runs[0].font.size = Pt(14)
            p.runs[0].font.bold = True
            doc.add_paragraph()
        elif line.startswith('### '):
            p = doc.add_paragraph(line[4:].strip())
            p.runs[0].font.size = Pt(12)
            p.runs[0].font.bold = True
            p.runs[0].font.italic = True
        else:
            clean_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            clean_line = re.sub(r'\*([^*]+)\*', r'\1', clean_line)
            p = doc.add_paragraph(clean_line)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.runs[0].font.size = Pt(12)
            p.runs[0].font.name = 'Times New Roman'
    
    output_folder = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(output_folder, exist_ok=True)
    topic_clean = re.sub(r'[^\w\s-]', '', metadata.get('topic', 'essay'))[:50]
    topic_clean = re.sub(r'[-\s]+', '_', topic_clean)
    filepath = os.path.join(output_folder, f"{topic_clean}.docx")
    doc.save(filepath)
    return filepath

def run_essay_agent(topic, requirements, job_id):
    state = {
        "topic": topic, "requirements": requirements, "research_tasks": [], "structure": [], 
        "research": "", "outline": "", "essay": "", "feedback": "", "file": None, 
        "status": "running", "current_iteration": 0, "current_phase": "Planning Research", 
        "logs": [], "scores": {}
    }
    essay_jobs[job_id] = state

    try:
        memory = retrieve_memory(topic)
        
        # Phase 1: Planning
        state["logs"].append("🎯 Planning research approach...")
        planning_prompt = f"Topic: {topic}\nRequirements:\n{requirements}\n"
        try:
            plan_response = essay_planner.run(planning_prompt)
            plan_data = json.loads(plan_response)
            state["research_tasks"] = plan_data.get("research_tasks", [])
            state["structure"] = plan_data.get("essay_structure", [])
            state["logs"].append(f"✓ Identified {len(state['research_tasks'])} research areas")
        except:
            state["logs"].append("✗ Planning failed")
            state["status"] = "error"
            return
        
        # Phase 2: Research
        state["current_phase"] = "Conducting Research"
        state["logs"].append("📚 Conducting deep research...")
        research_prompt = f"Topic: {topic}\nResearch Tasks:\n{chr(10).join(state['research_tasks'])}\nRequirements:\n{requirements}\nInternal Knowledge Base:\n{memory}"
        state["research"] = essay_researcher.run(research_prompt, temperature=0.4)
        state["logs"].append("✓ Research completed")
        
        # Phase 3: Outline
        state["current_phase"] = "Creating Outline"
        state["logs"].append("📝 Creating detailed outline...")
        outline_prompt = f"Topic: {topic}\nSuggested Structure:\n{chr(10).join(state['structure'])}\nResearch Findings:\n{state['research']}"
        state["outline"] = essay_outliner.run(outline_prompt)
        state["logs"].append("✓ Outline created")
        
        # Phase 4 & 5: Writing and Editing Loop
        for iteration in range(MAX_ITERATIONS_ESSAY):
            state["current_iteration"] = iteration + 1
            state["current_phase"] = f"Writing Essay (Draft {iteration + 1})"
            state["logs"].append(f"✍️ Writing draft {iteration + 1}...")
            
            writing_prompt = f"Topic: {topic}\nRequirements:\n{requirements}\nOutline:\n{state['outline']}\nResearch:\n{state['research']}\nPrevious Feedback:\n{state['feedback']}"
            state["essay"] = essay_writer.run(writing_prompt, temperature=0.5)
            
            state["current_phase"] = "Editorial Review"
            state["logs"].append("🔍 Editorial review in progress...")
            
            try:
                review_response = essay_editor.run(f"Topic: {topic}\nRequirements:\n{requirements}\nEssay:\n{state['essay']}")
                review_data = json.loads(review_response)
                state["scores"] = review_data.get("score", {})
                state["feedback"] = review_data.get("feedback", "")
                
                if review_data.get("pass", False):
                    state["logs"].append("✅ Essay approved by editor!")
                    break
                else:
                    improvements = review_data.get("improvements_needed", [])
                    state["logs"].append(f"⚠️ Revisions needed: {len(improvements)} issues")
            except:
                break
        
        # Phase 6: Document Creation
        state["current_phase"] = "Creating Document"
        state["logs"].append("📄 Creating Word document...")
        state["file"] = create_word_document(state["essay"], {"topic": topic}, job_id)
        
        state["status"] = "completed"
        state["current_phase"] = "Complete"
        state["logs"].append("🎉 Essay complete!")
        
    except Exception as e:
        state["status"] = "error"
        state["logs"].append(f"❌ Error: {str(e)}")

# ---------------------------------------------------
# App 3: Resume Builder Agents (NEW)
# ---------------------------------------------------

skill_extractor = Agent("skill_extractor", """
You are an expert technical recruiter. Analyze the provided file contexts (code snippets, old resumes).
Extract: 1. Core skills/tech stack 2. Project complexities 3. Experience level.
Return JSON ONLY: { "skills": ["skill1"], "projects": [{"name": "P1", "tech": "React", "desc": "Desc"}], "experience_level": "Mid", "inferred_role": "Full Stack" }
""")

resume_writer = Agent("resume_writer", """
You are an ATS resume writer. Draft a resume based on extracted skills, target role, and job description.
Requirements: Strong Summary, Technical Skills section, Experience/Projects section (use XYZ formula and action verbs for bullets).
Format in Markdown. DO NOT invent fake companies/dates.
""")

ats_editor = Agent("ats_editor", """
You are an ATS checker. Evaluate the resume draft.
Return JSON ONLY: { "pass": true/false, "score": {"impact": 0-10, "keywords": 0-10, "formatting": 0-10}, "improvements_needed": ["fix 1"], "feedback": "overall advice" }
Pass = true only if highly ATS-compliant and metric-driven.
""")

def create_resume_docx(resume_content, job_id):
    doc = Document()
    for section in doc.sections:
        section.top_margin, section.bottom_margin, section.left_margin, section.right_margin = Inches(0.5), Inches(0.5), Inches(0.5), Inches(0.5)
    
    for line in resume_content.split('\n'):
        line = line.strip()
        if not line: continue
        if line.startswith('# '):
            p = doc.add_paragraph(line[2:].strip())
            p.alignment, p.runs[0].font.size, p.runs[0].font.bold = WD_ALIGN_PARAGRAPH.CENTER, Pt(18), True
        elif line.startswith('## '):
            doc.add_paragraph()
            p = doc.add_paragraph(line[3:].strip().upper())
            p.runs[0].font.size, p.runs[0].font.bold = Pt(12), True
        elif line.startswith('### '):
            p = doc.add_paragraph(line[4:].strip())
            p.runs[0].font.size, p.runs[0].font.bold = Pt(11), True
        elif line.startswith('- ') or line.startswith('* '):
            clean_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line[2:])
            p = doc.add_paragraph(clean_line, style='List Bullet')
            p.runs[0].font.size = Pt(10)
        else:
            clean_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            p = doc.add_paragraph(clean_line)
            if "@" in line or "linkedin" in line.lower(): p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.runs[0].font.size = Pt(10)
            
    output_folder = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, "Optimized_Resume.docx")
    doc.save(filepath)
    return filepath

def run_resume_agent(target_role, job_desc, file_context, job_id):
    state = { "status": "running", "current_iteration": 0, "current_phase": "Extracting Context", "logs": [], "extracted_data": {}, "resume": "", "feedback": "", "file": None, "scores": {} }
    resume_jobs[job_id] = state

    try:
        memory_query = (target_role + " " + job_desc).strip() or "Software Engineering"
        memory = retrieve_memory(memory_query)
        combined_context = f"RAG Memory:\n{memory}\n\nUploaded Files Text:\n{file_context}"
        
        state["logs"].append("🧠 Analyzing uploaded files and code...")
        
        # FIX: Initialize the variable here so it exists even if the API call fails
        extracted_res = "Extraction failed or returned no data."
        
        try:
            role_prompt = f"Target Role: {target_role}" if target_role else "Target Role: Please infer the best role based on the context."
            extracted_res = skill_extractor.run(f"{role_prompt}\n\nContext Data:\n{combined_context}")
            
            extracted_data = json.loads(extracted_res)
            state["extracted_data"] = extracted_data
            state["logs"].append("✓ Extracted technical profile.")
            
            if not target_role and "inferred_role" in extracted_data:
                target_role = extracted_data["inferred_role"]
                state["logs"].append(f"✨ Auto-detected Target Role: {target_role}")
                
        except Exception as e:
            state["logs"].append("⚠️ Could not parse JSON cleanly, using raw text fallback.")
            state["extracted_data"] = extracted_res
            if not target_role: target_role = "Software Professional"

        for iteration in range(MAX_ITERATIONS_RESUME):
            state["current_iteration"] = iteration + 1
            state["current_phase"] = f"Drafting Resume (Draft {iteration + 1})"
            state["logs"].append(f"✍️ Writing resume draft {iteration + 1}...")
            
            prompt = f"Target Role: {target_role}\nJob Description:\n{job_desc}\nExtracted Profile:\n{state['extracted_data']}\nATS Feedback:\n{state['feedback']}"
            state["resume"] = resume_writer.run(prompt, temperature=0.4)
            
            state["current_phase"] = "ATS Review"
            state["logs"].append("🔍 Running ATS compatibility check...")
            try:
                review = json.loads(ats_editor.run(f"Target Role: {target_role}\nJob Description: {job_desc}\nResume Draft:\n{state['resume']}"))
                state["scores"], state["feedback"] = review.get("score", {}), review.get("feedback", "")
                if review.get("pass", False): 
                    state["logs"].append("✅ Resume passed ATS standards!")
                    break
                else:
                    state["logs"].append(f"⚠️ ATS Revisions needed: {len(review.get('improvements_needed', []))} issues found.")
            except: break
                
        state["current_phase"] = "Formatting Document"
        state["logs"].append("📄 Generating formatted Word document...")
        state["file"] = create_resume_docx(state["resume"], job_id)
        
        state["status"], state["current_phase"] = "completed", "Complete"
        state["logs"].append("🎉 Resume complete!")
        
    except Exception as e:
        state["status"], state["logs"] = "error", [f"Error: {str(e)}"]

# ---------------------------------------------------
# Core / General Routes
# ---------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------
# App 1 Routes (Code Generator)
# ---------------------------------------------------

@app.route('https://unified-ai-workspace.vercel.app/api/code/generate', methods=['POST', 'OPTIONS'])
def api_code_generate():
    if request.method == "OPTIONS":
        return "", 200

    goal = request.json.get('goal', '')
    if not goal:
        return jsonify({"error": "No goal provided"}), 400

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    threading.Thread(target=run_code_agent, args=(goal, job_id)).start()
    return job_id
    return jsonify({"job_id": job_id}
                  
                  )
@app.route('/api/code/status/<job_id>')
def api_code_status(job_id):
    if job_id not in code_jobs: return jsonify({"error": "Job not found"}), 404
    state = code_jobs[job_id]
    return jsonify({
        "status": state["status"], 
        "current_iteration": state["current_iteration"],
        "logs": state["logs"], 
        "tasks": state["tasks"], 
        "files": [os.path.basename(f) for f in state["files"]],
        "result": state["result"] if state["status"] == "completed" else "", 
        "feedback": state["feedback"]
    })

@app.route('/api/code/download/<job_id>')
def api_code_download(job_id):
    if job_id not in code_jobs or not code_jobs[job_id]["files"]:
        return jsonify({"error": "Job or files not found"}), 404
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fpath in code_jobs[job_id]["files"]: zf.write(fpath, os.path.basename(fpath))
    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name=f'agent_output_{job_id}.zip')


# ---------------------------------------------------
# App 2 Routes (Essay Generator)
# ---------------------------------------------------

@app.route('/api/essay/generate', methods=['POST'])
def api_essay_generate():
    data = request.json
    topic = data.get('topic')
    if not topic: return jsonify({"error": "No topic provided"}), 400
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    threading.Thread(target=run_essay_agent, args=(topic, data.get('requirements', ''), job_id)).start()
    return jsonify({"job_id": job_id})

@app.route('/api/essay/status/<job_id>')
def api_essay_status(job_id):
    if job_id not in essay_jobs: return jsonify({"error": "Job not found"}), 404
    state = essay_jobs[job_id]
    return jsonify({
        "status": state["status"], 
        "current_phase": state["current_phase"], 
        "logs": state["logs"],
        "research_tasks": state["research_tasks"], 
        "structure": state["structure"], 
        "scores": state["scores"],
        "has_file": state["file"] is not None, 
        "essay_preview": state["essay"][:500] if state["essay"] else "",
        "current_iteration": state["current_iteration"],
        "feedback": state["feedback"]
    })

@app.route('/api/essay/download/<job_id>')
def api_essay_download(job_id):
    state = essay_jobs.get(job_id)
    if not state or not state["file"]: return jsonify({"error": "File not found"}), 404
    return send_file(state["file"], as_attachment=True, download_name=os.path.basename(state["file"]))

@app.route('/api/essay/essay/<job_id>')
def api_essay_text(job_id):
    if job_id not in essay_jobs: return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "essay": essay_jobs[job_id]["essay"],
        "outline": essay_jobs[job_id]["outline"],
        "research": essay_jobs[job_id]["research"]
    })

# ---------------------------------------------------
# App 3 Routes (Resume Builder - NEW)
# ---------------------------------------------------
@app.route('/api/resume/generate', methods=['POST'])
def api_resume_generate():
    # Remove the default 'Software Engineer' so we know if it's truly blank
    target_role = request.form.get('target_role', '').strip() 
    job_desc = request.form.get('job_description', '')
    files = request.files.getlist('files')
    
    file_context = ""
    for f in files:
        if not f.filename: continue
        filename = secure_filename(f.filename)
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == '.pdf':
                reader = PyPDF2.PdfReader(f)
                text = "".join([page.extract_text() for page in reader.pages])
                file_context += f"\n--- PDF: {filename} ---\n{text}\n"
            elif ext == '.docx':
                doc = Document(f)
                text = "\n".join([p.text for p in doc.paragraphs])
                file_context += f"\n--- DOCX: {filename} ---\n{text}\n"
            else:
                text = f.read().decode('utf-8')
                file_context += f"\n--- CODE/TEXT: {filename} ---\n{text[:4000]}\n" 
        except Exception as e:
            print(f"Failed parsing {filename}: {e}")
            
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    threading.Thread(target=run_resume_agent, args=(target_role, job_desc, file_context, job_id)).start()
    return jsonify({"job_id": job_id})

@app.route('/api/resume/status/<job_id>')
def api_resume_status(job_id):
    if job_id not in resume_jobs: return jsonify({"error": "Not found"}), 404
    st = resume_jobs[job_id]
    return jsonify({
        "status": st["status"], "current_phase": st["current_phase"], 
        "current_iteration": st["current_iteration"], "logs": st["logs"], 
        "scores": st["scores"], "has_file": st["file"] is not None,
        "resume_preview": st["resume"][:500] if st["resume"] else "",
        "extracted_data": st["extracted_data"]
    })

@app.route('/api/resume/download/<job_id>')
def api_resume_download(job_id):
    if job_id not in resume_jobs or not resume_jobs[job_id]["file"]: return jsonify({"error": "Not found"}), 404
    return send_file(resume_jobs[job_id]["file"], as_attachment=True, download_name="Optimized_Resume.docx")

@app.route('/api/resume/text/<job_id>')
def api_resume_text(job_id):
    return jsonify({"resume": resume_jobs[job_id]["resume"]})

# ---------------------------------------------------
# App 4 Routes (RAG Vector Store)
# ---------------------------------------------------

@app.route('/api/rag/initialize', methods=['POST'])
def api_rag_initialize():
    try:
        initialize_model()
        return jsonify({'status': 'success', 'is_ready': is_model_ready})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rag/store', methods=['POST'])
def api_rag_store():
    global memory_store
    if not is_model_ready: return jsonify({'status': 'error', 'message': 'Model not initialized'}), 400
    text, meta = request.json.get('text', ''), request.json.get('metadata', {})
    if not text: return jsonify({'status': 'error', 'message': 'Text required'}), 400
    
    try:
        vector = embedding_model.encode(text, normalize_embeddings=True)
        memory = {'id': str(uuid.uuid4()), 'text': text, 'vector': vector.tolist(), 'metadata': meta, 'timestamp': datetime.now().timestamp()}
        memory_store.append(memory)
        return jsonify({'status': 'success', 'id': memory['id'], 'message': f'Stored: {text[:40]}...'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rag/query', methods=['POST'])
def api_rag_query():
    if not is_model_ready or len(memory_store) == 0:
        return jsonify({'status': 'success', 'results': [{'text': 'Model not ready or empty store.', 'score': 0}]})
    
    query_text = request.json.get('query', '')
    if not query_text: return jsonify({'status': 'error', 'message': 'Query required'}), 400
    
    try:
        query_vector = embedding_model.encode(query_text, normalize_embeddings=True)
        results = []
        for mem in memory_store:
            score = float(cosine_similarity(query_vector, np.array(mem['vector'])))
            results.append({'id': mem['id'], 'text': mem['text'], 'metadata': mem['metadata'], 'score': score, 'timestamp': mem['timestamp']})
        
        results.sort(key=lambda x: x['score'], reverse=True)
        relevant = [r for r in results if r['score'] > 0.40][:3]
        return jsonify({'status': 'success', 'results': relevant})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rag/delete/<memory_id>', methods=['DELETE'])
def api_rag_delete(memory_id):
    global memory_store
    for i, mem in enumerate(memory_store):
        if mem['id'] == memory_id:
            removed = memory_store.pop(i)
            return jsonify({'status': 'success', 'message': f'Removed: {removed["text"][:20]}...'})
    return jsonify({'status': 'error', 'message': 'Not found'}), 404

@app.route('/api/rag/clear', methods=['POST'])
def api_rag_clear():
    global memory_store
    memory_store = []
    return jsonify({'status': 'success', 'message': 'Memory store cleared'})

@app.route('/api/rag/list', methods=['GET'])
def api_rag_list():
    return jsonify({'status': 'success', 'count': len(memory_store), 'memories': [
        {'id': m['id'], 'text': m['text'], 'metadata': m['metadata'], 'timestamp': m['timestamp']} for m in memory_store
    ]})

@app.route('/api/rag/prune/age', methods=['POST'])
def api_rag_prune_age():
    global memory_store
    max_age_ms = request.json.get('max_age_ms', 3600000)
    now = datetime.now().timestamp()
    initial_count = len(memory_store)
    memory_store = [mem for mem in memory_store if (now - mem['timestamp']) * 1000 < max_age_ms]
    return jsonify({'status': 'success', 'message': f'Removed {initial_count - len(memory_store)} old entries', 'remaining': len(memory_store)})

@app.route('/api/rag/prune/size', methods=['POST'])
def api_rag_prune_size():
    global memory_store
    max_memories = request.json.get('max_memories', 200)
    if len(memory_store) <= max_memories:
        return jsonify({'status': 'success', 'message': 'No pruning needed', 'count': len(memory_store)})
    memory_store.sort(key=lambda x: x['timestamp'], reverse=True)
    removed = len(memory_store) - max_memories
    memory_store = memory_store[:max_memories]
    return jsonify({'status': 'success', 'message': f'Removed {removed} entries', 'remaining': len(memory_store)})

@app.route('/api/rag/prune/duplicates', methods=['POST'])
def api_rag_prune_duplicates():
    global memory_store
    threshold = request.json.get('threshold', 0.95)
    pruned = []
    for mem in memory_store:
        is_duplicate = False
        mem_vector = np.array(mem['vector'])
        for existing in pruned:
            if cosine_similarity(mem_vector, np.array(existing['vector'])) > threshold:
                is_duplicate = True
                break
        if not is_duplicate: pruned.append(mem)
    removed = len(memory_store) - len(pruned)
    memory_store = pruned
    return jsonify({'status': 'success', 'message': f'Removed {removed} duplicates', 'remaining': len(memory_store)})


# ---------------------------------------------------
# App 5: Job Search (AI-Powered Job Aggregator)
# ---------------------------------------------------

import requests
import time
import html as html_lib
from typing import List, Dict
import sqlite3

# Job Search Database
JOB_DB = os.path.join(OUTPUT_DIR, "jobs.db")

def init_job_db():
    conn = sqlite3.connect(JOB_DB)
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
    conn.commit()
    conn.close()

# Filter Keywords
GOOD_KEYWORDS = ["llm", "machine learning", "ai engineer", "rag", "nlp", "deep learning", "python", "ai platform", "genai", "ml engineer", "mlops"]
BAD_KEYWORDS = ["trainer", "training sales", "customer success", "advisor", "support rep", "marketing coordinator", "writer", "translation", "data entry", "annotator"]
HIGH_SIGNAL_COMPANIES = ["openai", "anthropic", "google deepmind", "modal", "replicate", "cohere", "hugging face", "scale ai", "cursor", "perplexity"]

def is_relevant_job(title: str, company: str = "") -> bool:
    t = title.lower()
    c = company.lower()
    if any(cmp in c for cmp in HIGH_SIGNAL_COMPANIES):
        if "engineer" in t or "developer" in t:
            return True
    if any(bad in t for bad in BAD_KEYWORDS):
        return False
    very_senior = ["principal", "staff engineer", "director", "vp", "chief", "head of"]
    if any(s in t for s in very_senior):
        return False
    if not any(m in t for m in ["engineer", "developer"]):
        return False
    return True

def score_job(job: Dict) -> int:
    score = 0
    title = (job.get("title") or "").lower()
    company = (job.get("company") or "").lower()
    
    if any(c in company for c in HIGH_SIGNAL_COMPANIES):
        score += 5
    
    source = job.get("source", "")
    if source == "LinkedIn":
        score += 2
    elif source == "HN":
        score += 3
    elif source == "RemoteOK":
        score += 1
    
    ai_keywords = sum(1 for k in GOOD_KEYWORDS if k in title)
    score += min(ai_keywords * 2, 6)
    
    if "python" in title:
        score += 2
    if "backend" in title:
        score += 2
    
    return max(0, score)

def search_remoteok(query: str) -> List[Dict]:
    url = "https://remoteok.com/api"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        jobs = []
        for job in data[1:100]:
            title = job.get("position", "")
            company = job.get("company", "")
            if not title or not is_relevant_job(title, company):
                continue
            jobs.append({
                "title": title,
                "company": company,
                "location": "Remote",
                "url": job.get("url", ""),
                "source": "RemoteOK"
            })
        return jobs
    except:
        return []

def run_job_search(job_titles: List[str]) -> List[Dict]:
    all_jobs = []
    
    # Search RemoteOK for first 2 queries
    for query in job_titles[:2]:
        try:
            remote_jobs = search_remoteok(query)
            all_jobs.extend(remote_jobs)
            time.sleep(2)
        except:
            pass
    
    # Score and deduplicate
    for job in all_jobs:
        job["score"] = score_job(job)
    
    seen = set()
    unique = []
    for job in all_jobs:
        url = job.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(job)
    
    unique.sort(key=lambda x: x.get("score", 0), reverse=True)
    return unique

def insert_jobs_db(jobs):
    conn = sqlite3.connect(JOB_DB)
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
        except:
            pass
    conn.commit()
    conn.close()
    return inserted

def get_jobs_db(status_filter="all"):
    conn = sqlite3.connect(JOB_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if status_filter == "all":
        cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    else:
        cursor.execute("SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC", (status_filter,))
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def update_job_status(job_id, status):
    conn = sqlite3.connect(JOB_DB)
    cursor = conn.cursor()
    cursor.execute("UPDATE jobs SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (status, job_id))
    conn.commit()
    conn.close()

# Job Search Store
job_search_jobs = {}

def run_job_search_async(job_id, job_titles):
    state = {"status": "running", "progress": "Starting search...", "jobs_found": 0}
    job_search_jobs[job_id] = state
    
    try:
        state["progress"] = "Searching RemoteOK..."
        jobs = run_job_search(job_titles)
        
        state["progress"] = "Saving jobs to database..."
        inserted = insert_jobs_db(jobs)
        
        state["jobs_found"] = len(jobs)
        state["status"] = "completed"
        state["progress"] = f"Complete! Found {len(jobs)} jobs, added {inserted} new ones."
    except Exception as e:
        state["status"] = "error"
        state["progress"] = f"Error: {str(e)}"

# App 5 Routes (Job Search)
@app.route('/api/jobs/search', methods=['POST'])
def api_job_search():
    job_titles = [
        "AI Engineer",
        "Machine Learning Engineer",
        "Backend Python Engineer",
        "Software Engineer AI"
    ]
    job_id = str(uuid.uuid4())
    thread = threading.Thread(target=run_job_search_async, args=(job_id, job_titles))
    thread.daemon = True
    thread.start()
    return jsonify({'job_id': job_id})

@app.route('/api/jobs/status/<job_id>')
def api_job_search_status(job_id):
    if job_id in job_search_jobs:
        return jsonify(job_search_jobs[job_id])
    return jsonify({'status': 'not_found'})

@app.route('/api/jobs/list')
def api_jobs_list():
    status_filter = request.args.get('status', 'all')
    jobs = get_jobs_db(status_filter)
    return jsonify({'jobs': jobs})

@app.route('/api/jobs/update/<int:job_id>/<status>', methods=['POST'])
def api_job_update(job_id, status):
    update_job_status(job_id, status)
    return jsonify({'success': True})

# Initialize job database
init_job_db()
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Unified AI Workspace - 5 Apps in One")
    print("   💻 Code Generator | 📝 Essay Writer | 📄 Resume Builder")
    print("   🧠 RAG Vector Store | 🔍 AI Job Search")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
