# 📚 AutoQuestion Generator

**Generate Reading Passages and Multiple-Choice Questions using Local LLMs**

This project, developed as part of a thesis, demonstrates how Large Language Models (LLMs) running locally via [Ollama](https://ollama.com/) can be used to automatically generate language learning materials. It creates reading passages tailored to specific topics, languages, proficiency levels, and styles, and then generates relevant multiple-choice comprehension questions based on the generated text.

This tool showcases a simple yet effective **agentic workflow**: it first tasks an LLM with generating text, then analyzes that text (for English), and finally tasks the LLM again to create questions based on the initial output. This multi-step process, combined with validation techniques, helps ensure the quality and relevance of the generated content.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Ollama](https://img.shields.io/badge/-Ollama-000000?style=flat&logo=ollama&logoColor=white)


## ✨ Key Features

*   **Text Generation:** Creates reading passages based on user-defined:
    *   Topic
    *   Language (English, Spanish, French, German, Italian, Danish, Russian)
    *   Proficiency Level (Basic, Intermediate, Advanced)
    *   Writing Style (Formal, Informal, Humorous, etc.)
*   **Question Generation:** Automatically creates multiple-choice questions (with answers) based *only* on the generated text.
*   **Multiple LLM Providers:** 
    *   **Local LLM Powered:** Uses models running locally via [Ollama](https://ollama.com/), ensuring privacy and offline capability. Supports switching between different downloaded Ollama models.
    *   **Cloud LLM Integration:** Supports [Groq](https://groq.com/) for blazing fast generation when you need speed over local privacy.
*   **Quality Control & Agentic Workflow:**
    *   **Iterative Refinement (English):** For English text, it uses `textstat` to calculate the Gunning Fog readability index and iteratively re-prompts the LLM if the generated text doesn't match the target difficulty level.
    *   **Structured Output:** Prompts the LLM to return questions in a specific JSON format.
    *   **Pydantic Validation:** Uses Pydantic models (`Question`, `Questions` in `backend.py`) to rigorously validate the structure and types of the JSON data received from the LLM, preventing errors from malformed responses.
*   **Text Analysis:** Provides basic metrics like word count, sentence count, estimated reading time, and Gunning Fog score (English only).
*   **Interactive Quiz:** Allows users to answer the generated questions within the Streamlit interface.
*   **Fuzzy Answer Checking:** Uses fuzzy string matching (`fuzzywuzzy`) to assess user answers, allowing for minor variations.
*   **Web Interface:** Built with [Streamlit](https://streamlit.io/) for an easy-to-use graphical interface.

## 🎓 Relevance to Language Education

This tool serves as a practical example of how AI can assist language educators and learners:

*   **Content Creation:** Quickly generate supplementary reading materials and comprehension checks tailored to specific curriculum needs or student interests.
*   **Personalized Learning:** Create texts and quizzes adjusted to individual student proficiency levels.
*   **Efficiency:** Reduce the time educators spend on creating basic reading and assessment materials.
*   **Accessibility:** Leverage the power of LLMs locally, without relying on paid cloud services or sharing potentially sensitive data.

## ⚙️ How it Works (Simplified Agentic Workflow)

1.  **User Input:** The user specifies parameters (topic, language, level, style, number of questions, LLM model) in the Streamlit frontend.
2.  **Text Generation (`backend.TextGenerator`):**
    *   Constructs a detailed prompt based on user input.
    *   Sends the prompt to the selected LLM provider (Ollama or Groq).
    *   **(English Only):** If the language is English, it calculates the Gunning Fog score of the generated text using `textstat`. If the score is outside the target range for the selected level, it refines the prompt (asking for simpler/more complex text) and asks the LLM to regenerate. This loop continues for a limited number of iterations (`MAX_ITERATIONS`) or until suitable text is generated.
    *   Stores the generated text.
3.  **Text Analysis (`backend.TextMetrics`):**
    *   Calculates metrics (word count, etc.) for the generated text.
4.  **Question Generation (`backend.QuestionGenerator`):**
    *   Constructs a prompt including the generated text, requesting a specific number of multiple-choice questions in JSON format.
    *   Sends the prompt to the LLM provider, explicitly requesting JSON output.
    *   Receives the JSON response from the LLM.
    *   **Crucially, validates the received JSON** against the `Questions` Pydantic model. If validation fails, an error is raised. This ensures the data structure is correct before further processing.
5.  **Frontend Display (`frontend.py`):**
    *   Displays the generated text, metrics, and the validated questions (with radio buttons for choices).
6.  **User Interaction & Checking:**
    *   The user answers the questions in the Streamlit form.
    *   On submission, the frontend compares the user's selected answer text to the correct answer text using `fuzzywuzzy`'s ratio similarity.
    *   Displays the results and score.

## 🚀 Getting Started

### Prerequisites

*   **Python:** Version 3.8 or higher.
*   **Pip:** Python package installer.
*   **Ollama:** The engine for running local LLMs (optional if you only use Groq).
*   **Groq API Key:** For using Groq's cloud-based LLM (optional if you only use Ollama).

### 1. Install Ollama (Optional)

Follow the official instructions for your operating system: [https://ollama.com/](https://ollama.com/)

*   **macOS:** Download the application.
*   **Linux:** Run the install script: `curl -fsSL https://ollama.com/install.sh | sh`
*   **Windows:** Download the installer.

After installation, ensure Ollama is running. You might need to start the Ollama application (macOS/Windows) or ensure the service is running (Linux).

### 2. Download an Ollama LLM Model (Optional)

You need at least one model compatible with Ollama. This project defaults to `gemma:2b`, a relatively small and fast model suitable for testing. You can download it via the command line:

```bash
ollama pull gemma:2b
```

Other models you might consider (check Ollama's model library for more):

*   `mistral:latest` (Good balance of performance and capability)
*   `llama3:8b` (More powerful, requires more resources)

You can list your downloaded models using:

```bash
ollama list
```

The application will attempt to fetch the list of available models, but having at least one downloaded beforehand is recommended.

### 3. Get a Groq API Key (Optional)

For faster generation using Groq's powerful cloud LLMs, you'll need to set up an API key:

1. Create a Groq account at [https://console.groq.com/signup](https://console.groq.com/signup)
2. After signing in, navigate to [https://console.groq.com/keys](https://console.groq.com/keys)
3. Click "Create API Key" and copy your new API key

There are two ways to provide your Groq API key to the application:

**Method 1: Environment Variable File**
Create a `.env` file in the project root with the following content:
```
GROQ_API_KEY=your_api_key_here
```

**Method 2: Input in the Application UI**
You can input your API key directly in the application when you select Groq as the provider.

### 4. Clone the Repository

```bash
git clone https://github.com/your-user-name/OllamaQuestionGenerator.git # Replace with your repo URL
cd autoquestion-generator
```

### 5. Install Dependencies


```bash
pip install -r requirements.txt
```
*(Note: `python-Levenshtein` improves the speed of `fuzzywuzzy`)*

### 6. Run the Streamlit App

Navigate to the repository directory in your terminal and run:

```bash
streamlit run frontend.py
```

This should automatically open the application in your default web browser.

## 💻 Usage

1.  Open the application in your browser (usually `http://localhost:8501`).
2.  Use the sidebar (**⚙️ Configuration**) to select:
    * Your preferred provider (Ollama for local generation or Groq for cloud generation)
    * If using Ollama: select the Ollama model (it should list the models you have downloaded)
    * If using Groq: enter your API key if not provided via `.env` file
    * The number of questions to generate
3.  In the main area (**📝 Input Parameters**), enter the main topic for the reading passage.
4.  Select the target language, proficiency level, and writing style.
5.  Click the **✨ Generate Content** button.
6.  Wait for the text and questions to be generated (progress bar will be shown).
7.  Review the **📖 Generated Text** and its analysis metrics.
8.  Answer the **❓ Multiple Choice Questions** in the quiz form.
9.  Click **Submit Answers** to see your results.

## 🔧 Configuration

Most configuration is done via the Streamlit interface:

*   **Provider:** Choose between Ollama (local) or Groq (cloud) LLMs.
*   **LLM Model:** 
    * For Ollama: Select any model downloaded on your system.
    * For Groq: Choose between available Groq models (`llama-3.1-8b-instant` or `meta-llama/llama-4-scout-17b-16e-instruct`).
*   **Number of Questions:** Adjust the quantity of MCQs to generate (1-10).
*   **Input Parameters:** Topic, Language, Level, Style.

Constants like `MAX_ITERATIONS` (for English text refinement) or `SIMILARITY_THRESHOLDS` (for answer checking) can be adjusted directly in `backend.py`.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgements

*   The developers of Ollama, Groq, Streamlit, Pydantic, Textstat, and FuzzyWuzzy.