from flask import Flask, request, render_template
from AI_Resume_Rank import evaluate_resumes
import tempfile, os

app = Flask(__name__)
UPLOAD_FOLDER = './resumes'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('AI_Resume_Rank.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    job_description = request.form.get('job_description', '').strip()
    job_keywords = set([k.strip() for k in request.form.get('keywords', '').split(',') if k.strip()])
    files = request.files.getlist('resumes')

    if not files or not job_description:
        return render_template('results.html', error="Please upload at least one resume and provide a job description.")

    # Save uploads
    paths = []
    for f in files:
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_FOLDER)
        f.save(tmp.name)
        tmp.close()
        paths.append(tmp.name)

    # Evaluate
    scores = evaluate_resumes(paths, job_description, job_keywords)

    # Cleanup
    for p in paths:
        try: os.remove(p)
        except: pass

    # Prepare for template
    result = [{"idx": i, "score": round(s, 2)} for i, s in scores]
    return render_template('results.html', scores=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
