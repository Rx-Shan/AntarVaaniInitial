import gradio as gr
from questionnaire import QuestionnaireApp
from sample_questions import SAMPLE_QUESTIONS
from utils import load_therapists
from llm_setup import initialize_llm, create_or_load_vector_db
from qa_chain import setup_qa_chain

# File paths (Updated to your machine)
CSV_PATH = "D:\\Code\\AntarVaani\\updated_mental_health_professionals.csv"
PDF_PATH = "D:\\Code\\AntarVaani\\data\\mental_health_Document.pdf"
DB_PATH = "chroma_db"

# 1. Run the CLI Questionnaire
app = QuestionnaireApp(SAMPLE_QUESTIONS)
json_data, context_str = app.run()

# 2. Load LLM and Therapists
llm = initialize_llm()
therapists = load_therapists(CSV_PATH)

therapist_info = "\n".join([
    f"Name: {t['Name']}, Specialization: {t['Specialization']}, Experience: {t['Experience']} years\nContact: {t['Contact']}\nApproach: {t['Approach']}"
    for t in therapists
])

# 3. Generate Initial Message
initial_message = llm.invoke(f"""
1. Greet the user.
2. Analyze their questionnaire responses.
3. Recommend the most suitable therapist.
4. Provide contact information.

User Responses:
{context_str}

Available Therapists:
{therapist_info}
""").content

# 4. Setup VectorDB and QA Chain
vector_db = create_or_load_vector_db(PDF_PATH, DB_PATH)
qa_chain = setup_qa_chain(vector_db, llm)

# 5. Gradio Chatbot Function
def chatbot(user_input, chat_history):
    if not chat_history:
        chat_history.append(("AntarVaani", initial_message))
    response = qa_chain.run(user_input)
    chat_history.append(("User", user_input))
    chat_history.append(("AntarVaani", response))
    return chat_history

# 6. Launch Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AntarVaani - Your Mental Health Companion")
    chatbot_ui = gr.Chatbot(label="AntarVaani Chat")
    msg_input = gr.Textbox(label="Your Message:")
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear Chat")

    send_btn.click(chatbot, inputs=[msg_input, chatbot_ui], outputs=chatbot_ui)
    clear_btn.click(lambda: [("AntarVaani", initial_message)], outputs=chatbot_ui)

demo.launch()
