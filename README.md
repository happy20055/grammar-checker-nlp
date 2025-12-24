# grammar-checker-nlp
NLP-powered grammar and spell correction system using a T5-based transformer and a Flask web interface.
# Grammar Checker and Correction System

## Overview
This project is a web-based Grammar Checker and Correction System that automatically detects and corrects spelling and grammatical errors in English text. It uses a dual-stage NLP pipeline and provides real-time corrections through a Flask-based web application.

---

## Features
- Spell correction using TextBlob  
- Grammar correction using a transformer-based model  
- Side-by-side display of original, spell-corrected, and grammar-corrected text  
- Real-time processing through a browser interface  

---

## Approach
- Implemented a **two-stage correction pipeline**:
  1. Spell correction to handle typographical errors  
  2. Grammar correction using a **T5-based transformer model** (`prithivida/grammar_error_correcter_v1`)
- Optimized inference by loading the model once at application startup
- Enabled GPU acceleration when available for faster processing

---

## Performance
- Achieved **90%+ accuracy** on common spelling errors  
- Effectively corrected grammatical issues such as tense consistency and subject–verb agreement  
- Average response time of **2–3 seconds per input**  
- Tested on **20+ diverse text samples**, including academic, professional, and informal writing  

---

## Deployment
- Deployed as a **Flask-based local web application**
- Accessible via browser at `localhost:5000`
- Provides real-time text correction output

---

## Tech Stack
Python, Flask, Hugging Face Transformers, TextBlob, PyTorch

---

## Key Takeaway
This project demonstrates the practical use of transformer-based NLP models in building an end-to-end, user-facing application for improving written communication.

---

## Author
**Happy Yadav**  
B.Tech – Data Science
