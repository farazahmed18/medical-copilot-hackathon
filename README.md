<h1 align="center">⚕️ Medical Guidelines Copilot</h1>

<div align="center">
  <img src="https://img.shields.io/badge/LLM-Llama%203.3-blue?style=for-the-badge&logo=meta" alt="Llama 3.3">
  <img src="https://img.shields.io/badge/Orchestration-LangChain-green?style=for-the-badge&logo=langchain" alt="LangChain">
  <img src="https://img.shields.io/badge/Database-ChromaDB-red?style=for-the-badge&logo=googlecloud" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Inference-Groq-orange?style=for-the-badge&logo=lightning" alt="Groq">
</div>

<hr>

<h2> Project Overview</h2>
<p>
  The <b>Medical Guidelines Copilot</b> is an AI-powered assistant designed to provide clinicians with instant, cited answers from complex medical guidelines.
</p>

<h3> Key Features</h3>
<ul>
  <li><b>Hybrid Fallback Guardrail:</b> A custom-engineered logic that forces the AI to rely on the internal vector database first. It explicitly disclaims the transition to general knowledge if documents miss.</li>
  <li><b>Conversational Memory:</b> Maintains state across multi-turn dialogues, allowing for contextual follow-up questions without repeating the subject.</li>
  <li><b>Transparent Citation Engine:</b> Automatically extracts and displays source metadata, including exact text chunks and page numbers, directly in the UI.</li>
  <li><b>Ultra-Low Latency:</b> Utilizing Llama 3.3 via Groq to achieve inference speeds under 400ms.</li>
  <li><b>End-to-End Observability:</b> Fully integrated with LangSmith for real-time tracing and performance monitoring.</li>
</ul>

<hr>

<h2> Technical Architecture</h2>
<p>The application follows a modular RAG (Retrieval-Augmented Generation) workflow:</p>
<ol>
  <li><b>Ingestion:</b> PyPDFLoader parses clinical guidelines into a RecursiveCharacterTextSplitter.</li>
  <li><b>Embedding:</b> Text is vectorized using the <code>all-MiniLM-L6-v2</code> HuggingFace model.</li>
  <li><b>Vector Store:</b> High-dimensional vectors are stored in a locally persistent ChromaDB instance.</li>
  <li><b>Retrieval:</b> LangChain handles the similarity search and conversational contextualization.</li>
  <li><b>Generation:</b> The context-augmented prompt is processed by Llama 3.3 (70B) for precise medical answering.</li>
</ol>

<hr>

<h2> Security & Setup</h2>
<p>This project follows industry-standard security protocols for API key management. <b>Secrets are never hard-coded into the codebase.</b></p>

<h3>Local Installation:</h3>
<ol>
  <li><b>Clone the Repository:</b>
    <pre><code>git clone https://github.com/farazahmed18/medical-copilot-hackathon.git</code></pre>
  </li>
  <li><b>Environment Setup:</b>
    <ul>
      <li>Create a <code>.env</code> file in the root directory.</li>
      <li>Refer to <code>.env.example</code> for required keys (<code>GROQ_API_KEY</code>, <code>LANGCHAIN_API_KEY</code>).</li>
    </ul>
  </li>
  <li><b>Install Dependencies:</b>
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li><b>Launch the App:</b>
    <pre><code>streamlit run app.py</code></pre>
  </li>
</ol>

<hr>

<h2> Observability & Reliability</h2>
<p>To solve the "black box" problem of traditional AI in healthcare, this project implements <b>LangSmith Tracing</b>:</p>
<ul>
  <li><b>Source Verification:</b> Every response is cross-referenced against the local vectorstore with exact page numbers.</li>
  <li><b>Traceability:</b> Developers can audit the exact document chunks retrieved for every user query to ensure zero-hallucination.</li>
</ul>

<hr>

<div align="center">
  <p><b>Developed by Faraz Ahmed Siddiqui</b><br>
  AI Engineer & Data Scientist | Dubai</p>
  <i>Developed for the AI Engineering Hackathon - 2026</i>
</div>
