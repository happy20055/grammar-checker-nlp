from flask import Flask, request, render_template_string
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from textblob import TextBlob
import torch

app = Flask(__name__)

# Load grammar correction model and tokenizer
model_name = "prithivida/grammar_error_correcter_v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def spell_correct(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

def grammar_correct(text):
    inputs = tokenizer.encode("grammar: " + text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def correct_text(text):
    spell_fixed = spell_correct(text)
    grammar_fixed = grammar_correct(spell_fixed)
    return spell_fixed, grammar_fixed

# Simple HTML template for the web interface
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>Grammar & Spell Checker</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        textarea { width: 100%; height: 120px; font-size: 16px; }
        .result { margin-top: 20px; }
        label { font-weight: bold; }
        input[type=submit] { padding: 10px 20px; font-size: 16px; }
    </style>
</head>
<body>
    <h1>Grammar & Spell Checker</h1>
    <form method="POST">
        <label for="input_text">Enter Text:</label><br>
        <textarea id="input_text" name="input_text" required>{{ request.form.get('input_text', '') }}</textarea><br><br>
        <input type="submit" value="Correct Text">
    </form>

    {% if original %}
    <div class="result">
        <p><strong>Original Text:</strong><br>{{ original }}</p>
        <p><strong>After Spell Correction:</strong><br>{{ spell_corrected }}</p>
        <p><strong>After Grammar Correction:</strong><br>{{ grammar_corrected }}</p>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    original = None
    spell_corrected = None
    grammar_corrected = None

    if request.method == "POST":
        original = request.form.get("input_text")
        spell_corrected, grammar_corrected = correct_text(original)

    return render_template_string(HTML_TEMPLATE,
                                  original=original,
                                  spell_corrected=spell_corrected,
                                  grammar_corrected=grammar_corrected)

if __name__ == "__main__":
    app.run(debug=True)
