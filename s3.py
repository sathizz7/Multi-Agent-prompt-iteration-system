import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Streamlit app
st.title("Multi Agent Prompt Iteration System")

# Constants
TARGET_F1_SCORE = 0.95
MAX_ITERATIONS = 15  # Prevent infinite loops in case of slow convergence

# Helper function to clean dataset
def clean_dataset(dataset):
    try:
        dataset['cleaned_category'] = dataset['category'].str.replace(r'^\d+\.\s*', '', regex=True)
        dataset['cleaned_category'] = dataset['cleaned_category'].str.split('\n').str[0]
        dataset = dataset.drop(columns=["category"])
        dataset["category"] = dataset['cleaned_category']
        dataset = dataset.drop(columns=["cleaned_category"])
        return dataset
    except Exception as e:
        st.error(f"Error cleaning dataset: {e}")
        return None

# AGENTS
class PromptCreationAgent:
    def __init__(self, client):
        self.client = client
        self.prompt = "You are a Classification agent, Carefully read the prompt and output the best-fitting category."
        self.category_prompts = {}

    def create_initial_prompt(self, categories):
        try:
            for category in categories:
                response = self.client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant that generates definitions based on a given category."},
                        {"role": "user", "content": f"category: {category}\nGenerate a concise and clear definition based on the given category."},
                    ],
                    temperature=1,
                    max_tokens=256,
                    top_p=1,
                )
                definition = response.choices[0].message.content.strip()
                self.category_prompts[category] = definition
            return self.generate_final_prompt()
        except Exception as e:
            st.error(f"Error creating initial prompt: {e}")
            return None

    def generate_final_prompt(self):
        prompt = self.prompt + "\n"
        for category, definition in self.category_prompts.items():
            prompt += f"{category}: {definition}\n"
        return prompt

    def refine_prompt(self, feedback):
        try:
            for category, feed in feedback.items():
                response = self.client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant that generates refined definitions based on a given topic and feedback."},
                        {"role": "user", "content": f"Feedback: {feed}\nGenerate a concise and clear definition based on the given topic and feedback."},
                    ]
                )
                refined_prompt = response.choices[0].message.content
                self.category_prompts[category] = refined_prompt
            return self.generate_final_prompt()
        except Exception as e:
            st.error(f"Error refining prompt: {e}")
            return None

class EvaluationAgent:
    def evaluate(self, true_labels, predicted_labels, categories):
        try:
            macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
            confusion = confusion_matrix(true_labels, predicted_labels, labels=categories)
            report = classification_report(true_labels, predicted_labels, target_names=categories, zero_division=1)
            return macro_f1, confusion, report
        except Exception as e:
            st.error(f"Error during evaluation: {e}")
            return None, None, None

class FeedbackCreationAgent:
    def __init__(self, client):
        self.client = client

    def generate_feedback(self, true_labels, predicted_labels, categories):
        feedback = {}
        try:
            for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)):
                if true != pred:
                    f = self.client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "you are a feedback agent."},
                            {"role": "user", "content": f"As the true label {true} was classified with incorrect predicted label {pred} provide more definitions for true label to improve"},
                            {"role": "user", "content": f"Write a feedback or advise prompt to improve the definitions for true label : {true} with examples"},
                        ],
                        max_tokens=1024,
                        temperature=1
                    )
                    feedback[true] = f.choices[0].message.content
            return feedback
        except Exception as e:
            st.error(f"Error generating feedback: {e}")
            return None

class ClassificationAgent:
    def __init__(self, model_name="distilbert-base-uncased", alpha=0.9):
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertModel.from_pretrained(model_name)
            self.memory = {}  # Stores learned category embeddings (list per category)
            self.alpha = alpha  # Decay factor for moving average update
        except Exception as e:
            st.error(f"Error initializing ClassificationAgent: {e}")

    def get_embedding(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)                 # EMBED OF CLS token representation (input sentance)
            return outputs.last_hidden_state[:, 0, :].numpy()  # CLS token representation
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            return None

    def classify_posts(self, prompt, posts, categories):
        try:
            prompt_embedding = self.get_embedding(prompt)
            if prompt_embedding is None:
                return None

            category_embeddings = []
            for category in categories:
                if category not in self.memory:
                    self.memory[category] = [self.get_embedding(category)]  # Store multiple representations

                category_vector = np.mean(self.memory[category], axis=0)

                # 🔥 New: Refine category embedding by incorporating the prompt
                category_vector = 0.8 * category_vector + 0.2 * prompt_embedding  

                category_embeddings.append(category_vector)

            category_embeddings = np.vstack(category_embeddings)  # Stack category vectors

            predictions = []
            for post_text in posts:
                post_embedding = self.get_embedding(post_text)
                if post_embedding is None:
                    continue

                # Compute similarity with category embeddings
                post_similarities = cosine_similarity(post_embedding, category_embeddings)[0]
                
                #New: Compute similarity between prompt and each category
                prompt_similarities = cosine_similarity(prompt_embedding, category_embeddings)[0]

                #New: Weight final similarity score (balancing post-category & prompt-category)
                final_similarities = 0.7 * post_similarities + 0.3 * prompt_similarities  

                best_category_idx = np.argmax(final_similarities)
                best_category = categories[best_category_idx]

                # Limit memory growth
                if len(self.memory[best_category]) >= 5:
                    self.memory[best_category].pop(0)  

                # Update category memory using moving average
                updated_embedding = self.alpha * np.mean(self.memory[best_category], axis=0) + (1 - self.alpha) * post_embedding
                self.memory[best_category].append(updated_embedding)

                predictions.append(best_category)

            return predictions
        except Exception as e:
            st.error(f"Error during classification: {e}")
            return None

    def correct_misclassification(self, post_text, correct_category):
        try:
            post_embedding = self.get_embedding(post_text)
            if correct_category in self.memory:
                self.memory[correct_category].append(post_embedding)  # Add post to correct category
                for category in self.memory:
                    if category != correct_category:
                        self.memory[category] = [e - 0.1 * post_embedding for e in self.memory[category]]
            else:
                self.memory[correct_category] = [post_embedding]  # Initialize category if missing
        except Exception as e:
            st.error(f"Error correcting misclassification: {e}")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (Excel file)", type=["xlsx"])
if uploaded_file is not None:
    try:
        dataset = pd.read_excel(uploaded_file)
        dataset = clean_dataset(dataset)
        if dataset is not None:
            st.write("Dataset Preview:")
            st.write(dataset.head())

            # Input API key
            api_key = st.text_input("Enter your Groq API Key", type="password")
            if api_key:
                try:
                    client = Groq(api_key=api_key)

                    # Initialize agents
                    prompt_agent = PromptCreationAgent(client)
                    classification_agent = ClassificationAgent()
                    evaluation_agent = EvaluationAgent()
                    feedback_agent = FeedbackCreationAgent(client)

                    # Extract posts and categories
                    posts = dataset["post_text"].tolist()
                    true_labels = dataset["category"].tolist()
                    categories = list(dataset["category"].unique())

                    # Iterative classification and refinement
                    prompt = prompt_agent.create_initial_prompt(categories)
                    best_refined_prompts = {}  # Store best refined definitions per category
                    iteration = 0

                    while iteration < MAX_ITERATIONS:  # Prevent infinite loops
                        iteration += 1
                        st.write(f"Iteration {iteration}")

                        # Classification
                        predicted_labels = classification_agent.classify_posts(prompt, posts, categories)
                        if predicted_labels is None:
                            break

                        # Evaluation
                        macro_f1, confusion, report = evaluation_agent.evaluate(true_labels, predicted_labels, categories)
                        if macro_f1 is None:
                            break
                        st.write(f"Macro F1 Score: {macro_f1}")

                        if macro_f1 >= TARGET_F1_SCORE:
                            st.write("Target F1 score achieved!")

                            # Save final category prompts to a file
                            final_category_prompts = "\n".join([f"{category}: {prompt}" for category, prompt in prompt_agent.category_prompts.items()])
                            st.download_button(
                                label="Download Final Category Prompts",
                                data=final_category_prompts,
                                file_name="final_category_prompts.txt",
                                mime="text/plain"
                            )

                            # Save predictions to a CSV file
                            results_df = pd.DataFrame({
                                "Post": posts,
                                "True Category": true_labels,
                                "Predicted Category": predicted_labels
                            })
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Classification Results",
                                data=csv_data,
                                file_name="classification_results.csv",
                                mime="text/csv"
                            )

                            # Save classification report
                            st.download_button(
                                label="Download Classification Report",
                                data=report,
                                file_name="classification_report.txt",
                                mime="text/plain"
                            )

                            break  # Stop iteration when the target F1 score is achieved

                        # Handle misclassifications
                        for post_text, true_label, pred_label in zip(posts, true_labels, predicted_labels):
                            if true_label != pred_label:
                                classification_agent.correct_misclassification(post_text, true_label)

                        # Generate feedback for misclassifications
                        feedback = feedback_agent.generate_feedback(true_labels, predicted_labels, categories)
                        if feedback is None:
                            break
                        prompt = prompt_agent.refine_prompt(feedback)
                        if prompt is None:
                            break

                except Exception as e:
                    st.error(f"Error initializing Groq client or agents: {e}")
    except Exception as e:
        st.error(f"Error reading dataset: {e}")