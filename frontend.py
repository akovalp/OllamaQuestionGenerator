# frontend.py
from typing import List, Optional, Dict, Any
import streamlit as st
import time
from pydantic import BaseModel  # Keep Pydantic for AppState


# Import necessary components from the backend module
import backend
# Import Question model specifically for AppState type hinting
from backend import Question

# Import fuzzy matching for direct use in the frontend answer check loop
from fuzzywuzzy import fuzz

# --- Frontend Specific Pydantic Model for Session State ---


class AppState(BaseModel):
    """Model for storing application state in Streamlit's session state."""
    topic: str = ""
    language: str = "English"
    level: str = "Basic"
    style: str = "Formal"
    selected_model: str = backend.DEFAULT_MODEL
    num_questions: int = backend.NUM_QUESTIONS
    generated_text: str = ""
    # Use the Question model imported from backend
    questions: List[Question] = []
    # Stores user's selected radio button option (e.g., "1) Paris")
    answers: List[Optional[str]] = []
    iterations: int = 0
    metrics: Dict[str, Any] = {}
    quiz_submitted: bool = False  # Flag to know if quiz answers were submitted

# --- Session State Initialization ---


def init_session_state():
    """Initialize session state variables using the AppState model."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState().model_dump()  # Store as dict

    # Ensure keys exist even if model defaults change later
    # (This helps prevent KeyError if AppState model evolves)
    app_state_dict = st.session_state.app_state
    for key, default_value in AppState().model_dump().items():
        if key not in app_state_dict:
            app_state_dict[key] = default_value

    # Ensure answers list matches the number of questions
    num_q = len(app_state_dict.get("questions", []))
    if len(app_state_dict.get("answers", [])) != num_q:
        app_state_dict["answers"] = [None] * num_q


# --- Main Streamlit Application ---
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="AutoQuestion Generator")
    st.title("📚 AutoQuestion Generator")
    st.info("""
        Generate reading passages and multiple-choice questions using local LLMs via Ollama.
        1.  Select parameters and model in the sidebar.
        2.  Enter a topic and click 'Generate Content'.
        3.  (Optional) Answer the generated questions and submit.
        *Note: Detailed difficulty analysis (Gunning Fog) only for texts in English.*
    """)

    # Initialize session state
    init_session_state()
    app_state = st.session_state.app_state  # Use dict directly

    # Fetch available models from backend ONCE per session or if list is empty
    # This prevents calling ollama.list() on every rerun
    if "available_models" not in st.session_state or not st.session_state.available_models:
        with st.spinner("Fetching available Ollama models..."):
            st.session_state.available_models = backend.get_available_models()
            # Ensure the currently selected model is valid, fallback if necessary
            if app_state["selected_model"] not in st.session_state.available_models:
                st.warning(
                    f"Previously selected model '{app_state['selected_model']}' not found. Switching to default '{backend.DEFAULT_MODEL}'.")
                app_state["selected_model"] = backend.DEFAULT_MODEL

    available_models = st.session_state.available_models

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection - update AppState on change
        selected_model = st.selectbox(
            "LLM Model",
            available_models,
            index=available_models.index(
                app_state["selected_model"]) if app_state["selected_model"] in available_models else 0,
            key="model_select",  # Use a distinct key
            help="Select the Ollama model for generation. Ensure it's available locally."
        )
        # Update state only if changed to avoid unnecessary reruns
        if selected_model != app_state["selected_model"]:
            app_state["selected_model"] = selected_model
            st.rerun()  # Rerun to ensure consistency if model changes affect defaults/logic

        # Update AppState on change for sliders/selects
        num_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=10,
            value=app_state["num_questions"],
            key="num_questions_slider",
            help="Select how many questions to generate."
        )
        if num_questions != app_state["num_questions"]:
            app_state["num_questions"] = num_questions

    # --- Main Area for Inputs and Outputs ---
    st.header("📝 Input Parameters")

    # Use a form for input grouping
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            # Use state for default values
            topic = st.text_input(
                "Main Topic", value=app_state["topic"], key="topic_input")
            language = st.selectbox("Language",
                                    ["English", "Spanish", "French", "German",
                                     "Italian", "Danish", "Russian"],
                                    index=["English", "Spanish", "French", "German", "Italian", "Danish", "Russian"].index(
                                        app_state["language"]) if app_state["language"] in ["English", "Spanish", "French", "German", "Italian", "Danish", "Russian"] else 0,
                                    key="language_select")
        with col2:
            level = st.selectbox("Language Proficiency Level",
                                 ["Basic", "Intermediate", "Advanced"],
                                 index=["Basic", "Intermediate", "Advanced"].index(app_state["level"]) if app_state["level"] in [
                                     "Basic", "Intermediate", "Advanced"] else 0,
                                 key="level_select")
            style = st.selectbox("Desired Writing Style/Tone",
                                 ["Formal", "Informal", "Humorous", "Serious",
                                  "Friendly", "Easy to read", "Neutral"],  # Added Neutral
                                 index=["Formal", "Informal", "Humorous", "Serious", "Friendly", "Easy to read", "Neutral"].index(
                                     app_state["style"]) if app_state["style"] in ["Formal", "Informal", "Humorous", "Serious", "Friendly", "Easy to read", "Neutral"] else 0,
                                 key="style_select")

        # Submit button for the form
        submitted = st.form_submit_button("✨ Generate Content")

    # Input validation check after form submission
    if submitted and not topic.strip():
        st.error("⚠️ Please enter a topic before generating content.")
        st.stop()  # Stop execution if topic is missing

    # --- Content Generation Logic ---
    if submitted:
        # Update state with the latest form values
        app_state["topic"] = topic
        app_state["language"] = language
        app_state["level"] = level
        app_state["style"] = style
        app_state["quiz_submitted"] = False  # Reset quiz submission status

        # Instantiate backend classes with the selected model from state
        text_generator = backend.TextGenerator(
            model=app_state["selected_model"])
        question_generator = backend.QuestionGenerator(
            model=app_state["selected_model"])

        # Display progress
        progress_text = f"Generating content for '{app_state['topic']}' using {app_state['selected_model']}..."
        progress_bar = st.progress(0, text=progress_text)
        st.session_state.generation_error = None  # Clear previous errors

        try:
            # Step 1: Generate Text
            progress_bar.progress(
                10, text=f"Generating text ({app_state['language']} - {app_state['level']} - {app_state['style']})...")
            text_response = text_generator.generate_text(
                app_state["topic"], app_state["language"], app_state["level"], app_state["style"])

            app_state["generated_text"] = text_response["generated_text"]
            app_state["iterations"] = text_response.get("iterations", 1)

            # Step 2: Calculate Metrics
            progress_bar.progress(40, text="Calculating text metrics...")
            metrics = backend.TextMetrics.calculate_metrics(
                app_state["generated_text"], app_state["language"])
            app_state["metrics"] = metrics

            # Step 3: Generate Questions
            if app_state["generated_text"]:
                progress_bar.progress(
                    60, text=f"Generating {app_state['num_questions']} questions...")
                questions_data = question_generator.generate_questions(
                    app_state["generated_text"],
                    num_questions=app_state["num_questions"],
                    language=app_state["language"]
                )
                # Validate and store questions using Pydantic model from backend
                app_state["questions"] = [backend.Question(
                    **q) for q in questions_data.get("questions", [])]
                # Reset answers for new questions
                app_state["answers"] = [None] * len(app_state["questions"])
            else:
                st.warning(
                    "Text generation failed, cannot generate questions.")
                app_state["questions"] = []
                app_state["answers"] = []

            progress_bar.progress(100, text="Content generation complete!")
            time.sleep(1)
            progress_bar.empty()

        except Exception as e:
            st.session_state.generation_error = f"❌ Error during generation: {str(e)}"
            progress_bar.empty()
            # No st.stop() here, allow UI to render the error message below

        # Force rerun AFTER generation attempt to update UI elements correctly
        st.rerun()

    # Display error message if generation failed
    if hasattr(st.session_state, 'generation_error') and st.session_state.generation_error:
        st.error(st.session_state.generation_error)

    # --- Displaying Results ---
    if app_state.get("generated_text"):
        st.markdown("---")
        st.header("📖 Generated Text")
        with st.expander("Click to view/hide the generated text", expanded=True):
            st.markdown(app_state["generated_text"])

            # Display metrics
            st.markdown("**Text Analysis:**")
            metrics = app_state.get("metrics", {})
            cols = st.columns(4)
            cols[0].metric("Word Count", metrics.get("word_count", "N/A"))
            # Display Gunning Fog only if available and not N/A
            gunning_fog = metrics.get('difficulty_gunning_fog', 'N/A')
            cols[1].metric("Difficulty (Gunning Fog)", gunning_fog if gunning_fog !=
                           "N/A (English only)" and gunning_fog != "N/A (Too short)" else "N/A")
            cols[2].metric("Sentence Count", metrics.get(
                "sentence_count", "N/A"))
            cols[3].metric("Est. Reading Time (min)", f"{metrics.get('reading_time_min', 'N/A'):.1f}" if isinstance(
                metrics.get('reading_time_min'), float) else "N/A")
    if app_state.get("questions"):
        st.markdown("---")
        st.header(
            f"❓ Multiple Choice Questions ({len(app_state['questions'])} generated)")

        # Use a separate form for submitting answers
        # Reset quiz submission status if the form is displayed again after submission
        # This happens implicitly because submitting the form causes a rerun
        # We use the quiz_submitted flag to show results *after* submission

        with st.form("quiz_form"):
            # Ensure the answers list in state matches the number of questions
            if len(app_state["answers"]) != len(app_state["questions"]):
                app_state["answers"] = [None] * len(app_state["questions"])

            for idx, question_dict in enumerate(app_state["questions"]):
                # question_dict is already a Question object, no need to convert
                question_obj = question_dict

                st.markdown(f"**{idx + 1}. {question_obj.question}**")

                # Create options with indices for radio button display consistency
                # Ensure choices are strings
                options_with_labels = [
                    f"{i+1}) {str(choice)}" for i, choice in enumerate(question_obj.choices)]

                # Find the index of the currently stored answer to set the default value
                current_answer_value = app_state["answers"][idx]
                try:
                    default_index = options_with_labels.index(
                        current_answer_value) if current_answer_value in options_with_labels else 0
                except ValueError:
                    default_index = 0  # Should not happen if state is managed correctly

                # Use index based assignment to session state list
                # Store the full selected option string (e.g., "1) Paris")
                selected_option = st.radio(
                    f"Select answer for question {idx + 1}",
                    options=options_with_labels,
                    key=f"question_{idx}",  # Unique key per question
                    index=default_index,  # Set default based on state
                    label_visibility="collapsed"
                )
                # Update the state immediately within the form context (Streamlit handles this)
                app_state["answers"][idx] = selected_option
                st.markdown("---")  # Separator between questions

            submitted_answers = st.form_submit_button("Submit Answers")

        # --- Answer Checking Logic ---
        # Check if answers were submitted in the *current* run OR if quiz_submitted flag is set from previous run
        if submitted_answers:
            app_state["quiz_submitted"] = True
            st.rerun()  # Rerun to display results outside the form

        if app_state["quiz_submitted"]:  # Check the flag after potential rerun
            st.header("📝 Results")
            correct_count = 0
            total_questions = len(app_state["questions"])

            # Retrieve the similarity threshold from backend constants
            threshold = backend.SIMILARITY_THRESHOLDS.get(
                app_state["language"], backend.SIMILARITY_THRESHOLDS["default"])

            all_answered = all(ans is not None for ans in app_state["answers"])
            if not all_answered:
                st.warning("Please answer all questions before submitting.")

            else:  # Proceed with checking if all answered
                for idx, question_dict in enumerate(app_state["questions"]):
                    # question_dict is already a Question object, no need to convert
                    question_obj = question_dict

                    # e.g., "1) Paris"
                    user_answer_full = app_state["answers"][idx]
                    # Safely extract the answer part after ") "
                    user_answer = user_answer_full.split(
                        ") ", 1)[1] if ") " in user_answer_full else user_answer_full

                    correct_answer = question_obj.answer

                    # Use fuzzy matching imported from fuzzywuzzy
                    similarity_ratio = fuzz.ratio(
                        user_answer.lower().strip(), correct_answer.lower().strip())
                    # Alternative: use backend helper function if defined
                    # similarity_ratio = backend.check_answer_similarity(user_answer, correct_answer, app_state["language"])

                    with st.container(border=True):
                        st.markdown(
                            f"**Question {idx+1}:** {question_obj.question}")
                        st.write(f"Your answer: `{user_answer}`")
                        st.write(f"Correct answer: `{correct_answer}`")

                        if similarity_ratio >= threshold:
                            st.success("✅ Correct!")
                            correct_count += 1
                        else:
                            st.error("Incorrect 😔")
                        # Optional: Show similarity score for debugging
                        # st.write(f"(Similarity: {similarity_ratio}%)")

                st.subheader(
                    f"Final Score: {correct_count} out of {total_questions}")
                if total_questions > 0:
                    score_percentage = (correct_count / total_questions) * 100
                    st.progress(int(score_percentage)/100,
                                text=f"{score_percentage:.1f}%")

                    if score_percentage >= 80:
                        st.balloons()
                        st.success("🎉 Excellent work!")
                    elif score_percentage >= 50:
                        st.info("👍 Good effort, keep practicing!")
                    else:
                        st.warning(
                            "🤔 Keep trying! Review the text and try again.")


# --- Run the App ---
if __name__ == "__main__":
    main()
