# import gradio as gr
# from questionnaire import QuestionnaireApp
# from sample_questions import SAMPLE_QUESTIONS
# from utils import load_therapists
# from llm_setup import initialize_llm, create_or_load_vector_db
# from qa_chain import setup_qa_chain

# # File paths (Updated to your machine)
# CSV_PATH = "D:\\Code\\AntarVaani\\updated_mental_health_professionals.csv"
# PDF_PATH = "D:\\Code\\AntarVaani\\data\\mental_health_Document.pdf"
# DB_PATH = "chroma_db"

# # 1. Run the CLI Questionnaire
# app = QuestionnaireApp(SAMPLE_QUESTIONS)
# json_data, context_str = app.run()

# # 2. Load LLM and Therapists
# llm = initialize_llm()
# therapists = load_therapists(CSV_PATH)

# therapist_info = "\n".join([
#     f"Name: {t['Name']}, Specialization: {t['Specialization']}, Experience: {t['Experience']} years\nContact: {t['Contact']}\nApproach: {t['Approach']}"
#     for t in therapists
# ])

# # 3. Generate Initial Message
# initial_message = llm.invoke(f"""
# 1. Greet the user.
# 2. Analyze their questionnaire responses.
# 3. Recommend the most suitable therapist.
# 4. Provide contact information.

# User Responses:
# {context_str}

# Available Therapists:
# {therapist_info}
# """).content

# # 4. Setup VectorDB and QA Chain
# vector_db = create_or_load_vector_db(PDF_PATH, DB_PATH)
# qa_chain = setup_qa_chain(vector_db, llm)

# # 5. Gradio Chatbot Function
# def chatbot(user_input, chat_history):
#     if not chat_history:
#         chat_history.append(("AntarVaani", initial_message))
#     response = qa_chain.run(user_input)
#     chat_history.append(("User", user_input))
#     chat_history.append(("AntarVaani", response))
#     return chat_history

# # 6. Launch Gradio UI
# with gr.Blocks() as demo:
#     gr.Markdown("# 🤖 AntarVaani - Your Mental Health Companion")
#     chatbot_ui = gr.Chatbot(label="AntarVaani Chat")
#     msg_input = gr.Textbox(label="Your Message:")
#     send_btn = gr.Button("Send")
#     clear_btn = gr.Button("Clear Chat")

#     send_btn.click(chatbot, inputs=[msg_input, chatbot_ui], outputs=chatbot_ui)
#     clear_btn.click(lambda: [("AntarVaani", initial_message)], outputs=chatbot_ui)

# demo.launch()
import gradio as gr
from sample_questions import SAMPLE_QUESTIONS
from utils import load_therapists
from llm_setup import initialize_llm
from qa_chain import create_or_load_vector_db, setup_qa_chain

CSV_PATH = "D:\\Code\\AntarVaani\\updated_mental_health_professionals.csv"
PDF_PATH = "D:\\Code\\AntarVaani\\data\\mental_health_Document.pdf"
DB_PATH = "chroma_db"

llm = initialize_llm()
therapists = load_therapists(CSV_PATH)
therapist_info = "\n".join([
    f"Name: {t['Name']}, Specialization: {t['Specialization']}, Experience: {t['Experience']} years\n"
    f"Contact: {t['Contact']}\nApproach: {t['Approach']}"
    for t in therapists
])

vector_db = create_or_load_vector_db(PDF_PATH, DB_PATH)
qa_chain = setup_qa_chain(vector_db, llm)

def format_answers(answers_dict):
    lines = []
    for q, a in answers_dict.items():
        lines.append(f"Q: {q}\nA: {a}")
    return "\n".join(lines)

def generate_initial_message(answers):
    context_str = format_answers(answers)
    prompt = f"""
    1. Greet the user.
    2. Analyze their questionnaire responses.
    3. Recommend the most suitable therapist.
    4. Provide contact information.

    User Responses:
    {context_str}

    Available Therapists:
    {therapist_info}
    """
    return llm.invoke(prompt).content

def chatbot(user_input, chat_history, initial_message):
    if not chat_history:
        chat_history = [("AntarVaani", initial_message)]
    response = qa_chain.run(user_input)
    chat_history.append(("User", user_input))
    chat_history.append(("AntarVaani", response))
    return chat_history

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AntarVaani - Your Mental Health Companion")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Please fill out this questionnaire:")
            question_inputs = {}
            for q, opts in SAMPLE_QUESTIONS.items():
                question_inputs[q] = gr.Radio(choices=opts, label=q, interactive=True)

            submit_btn = gr.Button("Submit Questionnaire")

        with gr.Column(scale=1):
            chatbot_ui = gr.Chatbot(label="AntarVaani Chat")
            msg_input = gr.Textbox(label="Your Message:")
            send_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear Chat")

    answers_state = gr.State({})
    initial_message_state = gr.State("")
    chat_history_state = gr.State([])

    def on_submit(*answers):
        answers_dict = {q: ans for q, ans in zip(SAMPLE_QUESTIONS.keys(), answers)}
        initial_msg = generate_initial_message(answers_dict)
        return answers_dict, initial_msg, [("AntarVaani", initial_msg)]

    submit_btn.click(
        on_submit,
        inputs=[question_inputs[q] for q in SAMPLE_QUESTIONS.keys()],
        outputs=[answers_state, initial_message_state, chat_history_state]
    )

    def on_send(message, chat_history, initial_message):
        if chat_history is None or len(chat_history) == 0:
            chat_history = [("AntarVaani", initial_message)]
        response = qa_chain.run(message)
        chat_history.append(("User", message))
        chat_history.append(("AntarVaani", response))
        return chat_history

    send_btn.click(
        on_send,
        inputs=[msg_input, chat_history_state, initial_message_state],
        outputs=chat_history_state
    ).then(lambda: "", None, msg_input)

    clear_btn.click(lambda: [], outputs=chat_history_state)

demo.launch()
