import os

from docx import Document
from flask import Flask, render_template, request
from openai import OpenAI
from pypdf import PdfReader
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {"pdf", "docx"}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def get_client():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("API key is not configured on this server.")
    return OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_docx(file):
    document = Document(file)
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("main.html", result="File must be 10 MB or smaller."), 413


@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        job = request.form.get("job", "").strip()
        resume_file = request.files.get("resume_file")

        if not job:
            result = "Please enter a target job title."
        elif not resume_file or not resume_file.filename:
            result = "Please select a resume file."
        elif not allowed_file(secure_filename(resume_file.filename)):
            result = "Only PDF and DOCX files are supported."
        else:
            extension = resume_file.filename.rsplit(".", 1)[1].lower()
            try:
                resume_text = read_docx(resume_file) if extension == "docx" else read_pdf(resume_file)
                if not resume_text.strip():
                    raise ValueError("No text was found in this file. Scanned image PDFs are unsupported.")

                response = get_client().chat.completions.create(
                    model="qwen-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional Chinese resume editor. Optimize the resume for "
                                "the target position, emphasize measurable achievements, remove redundant "
                                "content, and return a clear Chinese resume ready for use."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Target job title: {job}\n\nResume source text:\n{resume_text}",
                        },
                    ],
                )
                result = response.choices[0].message.content or "No usable AI response was returned."
            except (ValueError, RuntimeError) as error:
                result = str(error)
            except Exception:
                app.logger.exception("Resume optimization failed")
                result = "Optimization failed. Please try again later."

    return render_template("main.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
