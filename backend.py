# backend.py
import os
import json
from fuzzywuzzy import fuzz  # Keep fuzz here if you add backend answer checking
import textstat
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import ollama  # Import the official Ollama Python library
import time
from functools import wraps

# --- Constants ---
MAX_ITERATIONS = 10
# Default number of questions (can be overridden by frontend)
NUM_QUESTIONS = 5
# Changed Default model to smaller one for faster generation
DEFAULT_MODEL = "gemma3:12b"

NUM_CHOICES = 5  # Default number of answer choices for questions


LEVEL_RANGES = {
    "Basic": (0, 6),
    "Intermediate": (6, 12),
    "Advanced": (12, 25),
}

SIMILARITY_THRESHOLDS = {
    "English": 85,
    "Spanish": 80,
    "French": 80,
    "German": 80,
    "Italian": 80,
    "Danish": 80,
    "Russian": 75,
    "default": 80
}

# --- Pydantic Models for Data Structure ---


class GeneratedText(BaseModel):
    """Model for storing generated text and its metadata."""
    failed_texts: Optional[List[str]] = None
    generated_text: str
    score: Optional[float] = None
    level: str
    language: str
    style: str
    iterations: Optional[int] = None


class Question(BaseModel):
    """Model for storing a question with choices and answer."""
    question: str
    choices: List[str]
    answer: str


class Questions(BaseModel):
    """Model for storing a list of questions."""
    questions: List[Question]


# --- Ollama Interaction ---


def get_available_models() -> List[str]:
    """Gets the list of available models from Ollama."""
    try:
        models_response = ollama.list()
        available_models = [model['model']
                            for model in models_response.get('models', [])]
        if not available_models:
            # Fallback if list is empty but ollama responded
            available_models = [DEFAULT_MODEL]
        return available_models
    except Exception as e:
        print(
            f"Warning: Could not fetch models from Ollama: {e}. Using default.")
        # Return a default list if Ollama isn't running or reachable
        return [DEFAULT_MODEL]


# Fetch models once when the backend module is loaded
AVAILABLE_MODELS = get_available_models()


# --- Core Logic Classes ---

class TextGenerator:
    """Class for generating text based on given parameters."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model if model in AVAILABLE_MODELS else DEFAULT_MODEL
        if model not in AVAILABLE_MODELS:
            print(
                f"Warning: Specified model '{model}' not found in Ollama list. Falling back to '{self.model}'.")

    def check_model_availability(self) -> bool:
        """Check if the selected model is available in Ollama (based on initial fetch)."""
        # This check is now based on the list fetched at module load time.
        # For a dynamic check, call get_available_models() again.
        return self.model in AVAILABLE_MODELS

    def generate_text(self, topic: str, language: str, level: str, style: str) -> Dict[str, Any]:
        """Generate text based on the given parameters."""
        if not topic.strip():
            raise ValueError("Topic cannot be empty")

        # Log which model is used
        print(f"Generating text with model: {self.model}")

        if language == "English":
            return self._generate_text_english(topic, language, level, style)
        else:
            return self._generate_text_other_languages(topic, language, level, style)

    def _generate_text_english(self, topic: str, language: str, level: str, style: str) -> Dict[str, Any]:
        """Generate English text with readability scoring."""
        iterations = 0
        best_text = None
        best_score = None
        best_difference = None
        failed_texts = []

        for _ in range(MAX_ITERATIONS):
            iterations += 1
            prompt = self._create_english_prompt(
                topic, level, language, style, best_score)

            try:
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    system="You are a professional language teacher tasked to create a reading passage. Do not give any output besides the text do not include things like 'ok here is your text' or 'here is the text'.",
                )
                generated_text = response['response']

                try:
                    score = textstat.gunning_fog(generated_text)
                except (AttributeError, ValueError):  # Catch potential errors in textstat
                    print(
                        "Warning: textstat.gunning_fog failed. Using fallback difficulty estimation.")
                    score = self._estimate_difficulty(generated_text)

                low, high = LEVEL_RANGES.get(level, (0, 25))
                range_center = (high + low) / 2
                score_difference = abs(score - range_center)

                if low <= score <= high:
                    best_text = generated_text
                    best_score = score
                    break  # Found suitable text
                else:
                    failed_texts.append(
                        f"Iteration {iterations} (score {score:.2f}): {generated_text}")
                    if best_difference is None or score_difference < best_difference:
                        best_text = generated_text
                        best_score = score
                        best_difference = score_difference
            except Exception as e:
                print(
                    f"Error during text generation iteration {iterations}: {e}")
                if best_text:  # If we got something usable before the error, use it
                    break
                else:  # If error on first try or no usable text found, raise it
                    raise Exception(
                        f"Failed to generate text using model {self.model}: {e}")

        if best_text is None:

            raise Exception(
                f"Could not generate suitable text after {iterations} iterations using model {self.model}.")

        result = GeneratedText(
            generated_text=best_text,
            score=best_score,
            level=level,
            language=language,
            style=style,
            iterations=iterations,
            failed_texts=failed_texts  # Include the failed texts in the response
        )
        return result.model_dump()

    def _generate_text_other_languages(self, topic: str, language: str, level: str, style: str) -> Dict[str, Any]:
        """Generate text in languages other than English."""
        prompt = self._create_other_language_prompt(
            topic, level, language, style)
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                system="You are an AI assistant tasked to do exactly what the prompt says. Output only the requested text in the specified language."
            )
            generated_text = response['response']

            # Basic validation: Check if the response seems empty or is just an error message
            if not generated_text or len(generated_text) < 20:
                raise ValueError("Generated text is too short or empty.")

            result = GeneratedText(
                generated_text=generated_text,
                level=level,
                language=language,
                style=style,
                iterations=1  # No iterative refinement for non-English yet
            )
            return result.model_dump()
        except Exception as e:
            raise Exception(
                f"Error generating non-English text with model {self.model}: {e}")

    def _create_english_prompt(self, topic: str, level: str, language: str, style: str,
                               previous_score: Optional[float] = None) -> str:
        """Create a prompt for English text generation."""
        words_range = {"Basic": "150-250",
                       "Intermediate": "250-400", "Advanced": "400-600"}
        prompt = f"""
        Create a {level.lower()} level reading passage in {language} about "{topic}" with a {style.lower()} tone.
        Guidelines:
        - Target Word Count: {words_range.get(level, "300-400")} words.
        - Vocabulary/Complexity: Use vocabulary and sentence structures appropriate for the {level} level ({'simple sentences/common words' if level == 'Basic' else ('more complex sentences/some specialized terms' if level == 'Intermediate' else 'complex structures/domain-specific vocabulary')}).
        - Content: Ensure cohesive paragraphs with clear transitions. Develop the topic with appropriate depth for the level.
        - Style: Maintain a consistent {style.lower()} tone throughout.
        - Output: Provide ONLY the reading passage text, nothing else. No introductory phrases, explanations, or formatting beyond paragraphs.
        """
        if previous_score is not None:
            low, high = LEVEL_RANGES.get(level, (0, 20))
            if previous_score < low:
                prompt += f"\n- NOTE: The previous attempt scored {previous_score:.2f} (Gunning Fog), which was too simple for the target range {low}-{high}. Please generate a significantly more complex text."
            elif previous_score > high:
                prompt += f"\n- NOTE: The previous attempt scored {previous_score:.2f} (Gunning Fog), which was too complex for the target range {low}-{high}. Please generate a significantly simpler text."
            # Score was outside range but not strictly above/below (edge case?)
            else:
                prompt += f"\n- NOTE: The previous attempt scored {previous_score:.2f} (Gunning Fog). Aim closer to the middle of the target range {low}-{high}."

        return prompt.strip()

    def _create_other_language_prompt(self, topic: str, level: str, language: str, style: str) -> str:
        """Create a prompt for text generation in languages other than English."""
        words_range = {"Basic": "150-250",
                       "Intermediate": "250-400", "Advanced": "400-600"}
        return f"""
        Create a {level.lower()} level reading passage strictly in the {language} language about "{topic}" with a {style.lower()} tone.
        Guidelines:
        - Language: The entire text MUST be in {language}.
        - Target Word Count: Approximately {words_range.get(level, "300-400")} words.
        - Vocabulary/Complexity: Use vocabulary and sentence structures appropriate for a {level} learner of {language}.
        - Content: Ensure cohesive paragraphs with clear transitions. Develop the topic with appropriate depth for the level.
        - Style: Maintain a consistent {style.lower()} tone throughout.
        - Output: Provide ONLY the reading passage text in {language}, nothing else. No introductory phrases, explanations, or formatting beyond paragraphs.
        """.strip()

    def _estimate_difficulty(self, text: str) -> float:
        """Fallback method to estimate text difficulty."""
        try:
            words = text.split()
            num_words = len(words)
            if num_words == 0:
                return 0.0
            # Count sentences ending with . ! ?
            sentences = text.count('.') + text.count('!') + text.count('?')
            num_sentences = max(1, sentences)  # Avoid division by zero

            avg_words_per_sentence = num_words / num_sentences

            # Simple complex word estimation (e.g., length > 6 or 7) - less accurate than syllables
            complex_words = sum(1 for word in words if len(
                word) > 6)  # Adjust threshold as needed
            percent_complex = (complex_words / num_words) * \
                100 if num_words else 0

            # Simplified Fog Index approximation
            return 0.4 * (avg_words_per_sentence + percent_complex)
        except Exception as e:
            print(f"Error during fallback difficulty estimation: {e}")
            return 0.0  # Return a neutral value on error


class QuestionGenerator:
    """Class for generating questions based on text."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model if model in AVAILABLE_MODELS else DEFAULT_MODEL
        if model not in AVAILABLE_MODELS:
            print(
                f"Warning: Specified model '{model}' not found in Ollama list. Falling back to '{self.model}'.")

    def generate_questions(self, generated_text: str, num_questions: int, language: str, choices_num: int) -> Dict[str, Any]:
        """Generate questions based on the given text."""
        if not generated_text.strip():
            raise ValueError("Generated text cannot be empty")

        # Refined prompt requesting JSON output matching the Pydantic model
        prompt = f"""
        Reading Passage ({language}):
        ---
        {generated_text}
        ---

        Task: Based *only* on the reading passage above, generate exactly {num_questions} multiple-choice comprehension questions.
        For each question:
        1.  Provide the question itself.
        2.  Provide exactly {choices_num} plausible answer choices (options). One choice must be the correct answer based on the text.
        3.  Clearly indicate the correct answer.
        4.  All questions, choices, and the answer text MUST be in the same language as the reading passage ({language}).

        Format the output as a JSON object containing a single key "questions", which is a list of question objects.
        Each question object should have the keys "question" (string), "choices" (list of {choices_num} strings), and "answer" (string - the correct choice text).

        Example JSON structure:
        {{
          "questions": [
            {{
              "question": "Sample question in {language}?",
              "choices": ["Choice A in {language}", "Choice B in {language}", "Choice C in {language}"],
              "answer": "Choice B in {language}"
            }},
            // ... more question objects
          ]
        }}

        Generate the JSON output now based on the provided passage.
        """

        # Log model used
        print(f"Generating questions with model: {self.model}")

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system",
                        "content": f"You are an AI assistant specialized in creating multiple-choice comprehension questions based on provided text. Respond ONLY with the requested JSON object containing the questions. Ensure all text content (questions, choices, answers) is in {language}."},
                    {"role": "user", "content": prompt}
                ],
                format='json',  # Request JSON format directly
                # Lower temperature for more predictable JSON structure
                options={"temperature": 0.5}
            )

            # The response content should be a JSON string
            json_response_str = response['message']['content']

            # Attempt to parse the JSON string
            try:
                # Validate the parsed data against the Pydantic model
                validated_data = Questions.model_validate_json(
                    json_response_str)
                # Ensure the correct number of questions were generated
                if len(validated_data.questions) != num_questions:
                    print(
                        f"Warning: Requested {num_questions} questions, but model generated {len(validated_data.questions)}. Using generated questions.")
                    # You might want to retry or handle this case more robustly
                return validated_data.model_dump()  # Return Pydantic model output as dict
            except (json.JSONDecodeError, ValidationError) as json_error:
                print(
                    f"Error: Failed to parse or validate JSON response from model {self.model}.")
                print(f"Model Response String: {json_response_str}")
                raise Exception(
                    f"Could not parse valid JSON questions from the model: {json_error}")

        except Exception as e:
            # Catch errors from ollama.chat or JSON parsing
            raise Exception(
                f"Error generating questions with model {self.model}: {e}")


class TextMetrics:
    """Class for calculating text metrics."""

    @staticmethod
    def calculate_metrics(text: str, language: str) -> Dict[str, Any]:
        """Calculate metrics for the given text."""
        metrics = {}
        word_count = len(text.split())
        metrics["word_count"] = word_count

        if language == "English":
            try:
                # Use textstat if available and text is suitable
                if word_count > 10:  # Basic check if text is long enough for stats
                    metrics["difficulty_gunning_fog"] = round(
                        textstat.gunning_fog(text), 2)
                    metrics["sentence_count"] = textstat.sentence_count(text)
                    # Calculate reading time in minutes
                    reading_time_seconds = textstat.reading_time(
                        text, ms_per_char=14.6)  # Standard ~200 WPM
                    metrics["reading_time_min"] = round(
                        reading_time_seconds / 60, 1)
                else:
                    metrics["difficulty_gunning_fog"] = "N/A (Too short)"
                    metrics["sentence_count"] = text.count(
                        '.') + text.count('!') + text.count('?')  # Basic count
                    # Estimate for short text
                    metrics["reading_time_min"] = "~0.1"
            except Exception as e:
                print(
                    f"Warning: textstat calculation failed for English text: {e}. Metrics might be incomplete.")
                # Provide basic fallbacks if textstat fails
                if "sentence_count" not in metrics:
                    metrics["sentence_count"] = text.count(
                        '.') + text.count('!') + text.count('?')
                if "reading_time_min" not in metrics:
                    # Simple fallback: average reading speed of 200 words per minute
                    metrics["reading_time_min"] = round(
                        word_count / 200, 1) if word_count > 0 else 0.0
                if "difficulty_gunning_fog" not in metrics:
                    # Use the fallback difficulty estimator from TextGenerator if needed
                    # (Creating a temporary instance - consider making _estimate_difficulty static or moving it)
                    temp_gen = TextGenerator()
                    metrics["difficulty_gunning_fog"] = round(temp_gen._estimate_difficulty(
                        text), 2) if word_count > 10 else "N/A (Too short)"
                    metrics[
                        "difficulty_gunning_fog"] = f"{metrics['difficulty_gunning_fog']} (est.)"

        else:  # For non-English languages
            metrics["sentence_count"] = text.count(
                '.') + text.count('!') + text.count('?')
            metrics["reading_time_min"] = round(
                word_count / 180, 1) if word_count > 0 else 0.0  # Slightly slower WPM estimate
            metrics["difficulty_gunning_fog"] = "N/A (English only)"

        return metrics

# Example of how to check answer similarity (can be called from frontend)


def check_answer_similarity(user_answer: str, correct_answer: str, language: str) -> int:
    """Calculates fuzzy similarity between two answers."""
    return fuzz.ratio(user_answer.lower().strip(), correct_answer.lower().strip())
