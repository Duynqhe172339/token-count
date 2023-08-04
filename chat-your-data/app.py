import os
from typing import Optional, Tuple

import gradio as gr
import pickle
from query_data import get_chain
from threading import Lock
from langchain.callbacks import get_openai_callback

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)


def set_openai_api_key(api_key: str):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        chain = get_chain(vectorstore)
        os.environ["OPENAI_API_KEY"] ="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"
        return chain

from langchain.callbacks import get_openai_callback

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            with get_openai_callback() as cb: # Use context manager here
                output = chain({"question": inp, "chat_history": history})["answer"]
                total_tokens = cb.total_tokens # Get total tokens here
                prompt_tokens = cb.prompt_tokens # Get prompt tokens here
                completion_tokens = cb.completion_tokens # Get completion tokens here
                total_cost = cb.total_cost # Get total cost here
                print(f"Tokens Used: {total_tokens}") # Print or return total tokens as you wish
                print(f"Prompt Tokens: {prompt_tokens}") # Print or return prompt tokens as you wish
                print(f"Completion Tokens: {completion_tokens}") # Print or return completion tokens as you wish
                print(f"Total Cost (USD): {total_cost}") # Print or return total cost as you wish
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Chat-Your-Data (State-of-the-Union)</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask questions about the most recent state of the union",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "What did the president say about Kentaji Brown Jackson",
            "Did he mention Stephen Breyer?",
            "What was his stance on Ukraine",
        ],
        inputs=message,
    )

    gr.HTML("Demo application of a LangChain chain.")

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )

block.launch(debug=True)