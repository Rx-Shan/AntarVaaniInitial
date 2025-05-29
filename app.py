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
from llm_setup import initialize_llm, create_or_load_vector_db
from qa_chain import setup_qa_chain

# Paths
CSV_PATH = "updated_mental_health_professionals.csv" #"D:\\Code\\AntarVaani\\updated_mental_health_professionals.csv"
PDF_PATH = "data/mental_health_Document.pdf" #D:\\Code\\AntarVaani\\data\\mental_health_Document.pdf
DB_PATH = "chroma_db"

# Load core components
llm = initialize_llm()
vector_db = create_or_load_vector_db(PDF_PATH, DB_PATH)
qa_chain = setup_qa_chain(vector_db, llm)
therapists = load_therapists(CSV_PATH)

therapist_info = "\n".join([
    f"Name: {t['Name']}, Specialization: {t['Specialization']}, Experience: {t['Experience']} years\nContact: {t['Contact']}\nApproach: {t['Approach']}"
    for t in therapists
])

# Questionnaire logic
def submit_questionnaire(*answers):
    questions = list(SAMPLE_QUESTIONS.keys())
    responses = list(zip(questions, answers))
    
    context_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in responses])
    json_context = {
        "metadata": {"total_questions": len(SAMPLE_QUESTIONS), "answered": len(responses)},
        "answers": [{"question": q, "selected_answer": a} for q, a in responses]
    }

    # LLM initial message
    initial_msg = llm.invoke(f"""
1. Greet the user.
2. Analyze their questionnaire responses.
3. Recommend the most suitable therapist.
4. Provide contact information.

User Responses:
{context_str}

Available Therapists:
{therapist_info}
""").content

    return (
        gr.update(visible=False),     # hide questionnaire
        gr.update(visible=True),      # show chatbot
        responses,                    # store in response state
        [("AntarVaani", initial_msg)], # initialize chat
        gr.update(value=initial_msg)   # update state
    )

# Chat interaction logic
def chatbot(user_input, chat_history, initial_msg):
    if not chat_history:
        chat_history.append(("AntarVaani", initial_msg))
    chat_history.append(("User", user_input))
    response = qa_chain.run(user_input)
    chat_history.append(("AntarVaani", response))
    return chat_history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AntarVaani - Your Mental Health Companion")

    # Define State Components inside the app scope
    response_state = gr.State([])
    chat_history_state = gr.State([])
    initial_message_state = gr.State("")

    # Questionnaire Section
    with gr.Column(visible=True) as questionnaire_section:
        gr.Markdown("### 📋 Mental Health Questionnaire")
        question_widgets = []
        for question, options in SAMPLE_QUESTIONS.items():
            radio = gr.Radio(label=question, choices=options)
            question_widgets.append(radio)
        submit_btn = gr.Button("Submit Questionnaire")

    # Chatbot Section
    with gr.Column(visible=False) as chatbot_section:
        chatbot_ui = gr.Chatbot(label="AntarVaani Chat")
        user_msg = gr.Textbox(label="Your Message:")
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear Chat")

    # Button logic
    submit_btn.click(
        submit_questionnaire,
        inputs=question_widgets,
        outputs=[
            questionnaire_section,
            chatbot_section,
            response_state,
            chatbot_ui,
            initial_message_state
        ]
    )

    send_btn.click(
        chatbot,
        inputs=[user_msg, chatbot_ui, initial_message_state],
        outputs=chatbot_ui
    )

    clear_btn.click(
        lambda initial: [("AntarVaani", initial)],
        inputs=[initial_message_state],
        outputs=chatbot_ui
    )

demo.launch()


