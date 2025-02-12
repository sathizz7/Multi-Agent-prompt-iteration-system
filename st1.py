import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Set the title of the webpage
st.title("Multi Agent Prompt Iteration System")

# Input for Groq API key
api_key = st.text_input("Enter your Groq API key", type="password")

if not api_key:
    st.warning("Please enter a valid Groq API key to proceed.")
    st.stop()

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (Excel file)", type=["xlsx"])

if uploaded_file is None:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

try:
    # Load dataset
    dataset = pd.read_excel(uploaded_file)

    # Check if required columns exist
    if "category" not in dataset.columns or "post_text" not in dataset.columns:
        st.error("The dataset must contain 'category' and 'post_text' columns.")
        st.stop()

    # Clean dataset
    dataset['cleaned_category'] = dataset['category'].str.replace(r'^\d+\.\s*', '', regex=True)
    dataset['cleaned_category'] = dataset['cleaned_category'].str.split('\n').str[0]
    dataset = dataset.drop(columns=["category"])
    dataset["category"] = dataset['cleaned_category']
    dataset = dataset.drop(columns=["cleaned_category"])

    # Initialize Groq client
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        st.stop()

    # Define agents
    class PromptCreationAgent:
        def __init__(self):
            self.prompt = "You are a Classification agent, Carefully read the prompt and output the best-fitting category."

        def create_initial_prompt(self, categories):
            label = []
            gen_prom = []
            dic = {}

            for category in categories:
                try:
                    response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "You are an AI assistant that generates definitions based on a given category."},
                            {"role": "user", "content": f"category: {category}\nGenerate a concise and clear definition based on the given category."},
                        ],
                        temperature=1,
                        max_tokens=256,
                        top_p=1,
                    )
                    response = response.choices[0].message.content.strip()
                    label.append(category)
                    gen_prom.append(response)
                except Exception as e:
                    st.error(f"Error generating prompt for category {category}: {e}")
                    continue

            dic = dict(zip(label, gen_prom))

            for category, definition in dic.items():
                self.prompt += f"{category}: {definition}\n"
            return self.prompt

        def refine_prompt(self, feedback):
            for feed in feedback:
                try:
                    response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "You are an AI assistant that generates refined definitions based on a given topic and feedback."},
                            {"role": "user", "content": f"Feedback: {feed}\nGenerate a concise and clear definition based on the given topic and feedback."},
                        ],
                    )
                    self.prompt = response.choices[0].message.content
                except Exception as e:
                    st.error(f"Error refining prompt: {e}")
                    continue
            return self.prompt

    class EvaluationAgent:
        def evaluate(self, true_labels, predicted_labels, categories):
            try:
                macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
                confusion = confusion_matrix(true_labels, predicted_labels, labels=categories)
                report = classification_report(true_labels, predicted_labels, target_names=categories)
                return macro_f1, confusion, report
            except Exception as e:
                st.error(f"Error during evaluation: {e}")
                return None, None, None

    class FeedbackCreationAgent:
        def generate_feedback(self, true_labels, predicted_labels, categories):
            feedback = []
            true_label = []

            for true, pred in zip(true_labels, predicted_labels):
                if true != pred:
                    try:
                        f = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {"role": "system", "content": "you are a feedback agent."},
                                {"role": "user", "content": f"As the true label {true} was classified with incorrect predicted label {pred} provide more definitions for true label to improve"},
                                {"role": "user", "content": f"Write a feedback or advise prompt to improve the definitions for true label : {true} with examples"},
                            ],
                            max_tokens=1024,
                            temperature=1,
                        )
                        true_label.append(true)
                        if f is not None:
                            feedback.append(f.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Error generating feedback: {e}")
                        continue

            return feedback

    class ClassificationAgent:
        def __init__(self, model_name="distilbert-base-uncased", alpha=0.9):
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertModel.from_pretrained(model_name)
            self.memory = {}  # Stores learned category embeddings (list per category)
            self.alpha = alpha  # Decay factor for moving average update

        def get_embedding(self, text):
            try:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
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
                    category_embeddings.append(category_vector)

                category_embeddings = np.vstack(category_embeddings)  # Stack category vectors

                predictions = []
                for post_text in posts:
                    post_embedding = self.get_embedding(post_text)
                    if post_embedding is None:
                        continue

                    similarities = cosine_similarity(post_embedding, category_embeddings)[0]
                    best_category_idx = np.argmax(similarities)
                    best_category = categories[best_category_idx]

                    if len(self.memory[best_category]) >= 5:  # Limit memory growth
                        self.memory[best_category].pop(0)  # Remove oldest embedding

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
                if post_embedding is None:
                    return

                if correct_category in self.memory:
                    self.memory[correct_category].append(post_embedding)  # Add post to correct category

                    for category in self.memory:
                        if category != correct_category:
                            self.memory[category] = [e - 0.1 * post_embedding for e in self.memory[category]]
                else:
                    self.memory[correct_category] = [post_embedding]  # Initialize category if missing
            except Exception as e:
                st.error(f"Error correcting misclassification: {e}")

    # Initialize agents
    prompt_agent = PromptCreationAgent()
    classification_agent = ClassificationAgent()
    evaluation_agent = EvaluationAgent()
    feedback_agent = FeedbackCreationAgent()

    # Extract posts and categories
    posts = dataset["post_text"].tolist()
    true_labels = dataset["category"].tolist()
    categories = list(dataset["category"].unique())

    # Iterative classification and refinement
    prompt = prompt_agent.create_initial_prompt(categories)
    best_refined_prompts = {}  # Store best refined definitions per category
    iteration = 0

    while True:  # Infinite loop until F1 score target is achieved
        iteration += 1
        st.write(f"Iteration {iteration}")

        # Classification
        predicted_labels = classification_agent.classify_posts(prompt, posts, categories)
        if predicted_labels is None:
            st.error("Classification failed. Please check the logs.")
            break

        # Evaluation
        macro_f1, confusion, report = evaluation_agent.evaluate(true_labels, predicted_labels, categories)
        if macro_f1 is None:
            st.error("Evaluation failed. Please check the logs.")
            break

        st.write(f"Macro F1 Score: {macro_f1}")

        # Display and save classification report
        st.text_area("Classification Report", report, height=300)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        st.success("Classification report saved to classification_report.txt")

        # Check for success
        if macro_f1 >= 0.95:
            st.success("Target F1 score achieved!")

            # Save predictions to a CSV file
            results_df = pd.DataFrame({
                "Post": posts,
                "True Category": true_labels,
                "Predicted Category": predicted_labels
            })
            results_df.to_csv("classification_results.csv", index=False)
            st.success("Predictions saved to classification_results.csv")

            # Save best refined definitions to a file
            with open("best_refined_prompts.txt", "w") as f:
                for category, definition in best_refined_prompts.items():
                    f.write(f"{category}: {definition}\n")
            st.success("Best refined prompts saved to best_refined_prompts.txt")

            break

        # Handle misclassifications
        for post_text, true_label, pred_label in zip(posts, true_labels, predicted_labels):
            if true_label != pred_label:
                classification_agent.correct_misclassification(post_text, true_label)

        # Feedback
        feedback = feedback_agent.generate_feedback(true_labels, predicted_labels, categories)
        prompt = prompt_agent.refine_prompt(feedback)

        # Store best refined definitions for each category
        for category in categories:
            best_refined_prompts[category] = prompt  # Store latest refined prompt

        st.write("Prompt refined.")

    # Download buttons
    st.download_button(
        label="Download Best Refined Prompt",
        data=open("best_refined_prompts.txt", "rb").read(),
        file_name="best_refined_prompts.txt",
        mime="text/plain",
    )

    st.download_button(
        label="Download Classification Report",
        data=open("classification_report.txt", "rb").read(),
        file_name="classification_report.txt",
        mime="text/plain",
    )

    st.download_button(
        label="Download Classification Results",
        data=open("classification_results.csv", "rb").read(),
        file_name="classification_results.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")