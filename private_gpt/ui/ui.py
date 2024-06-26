"""This file should be imported if and only if you want to run the UI locally."""

import itertools
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from deep_translator import GoogleTranslator

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

import pickle
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import util
import private_gpt.ui.load_text_model as load_text_model
import uuid

from dotenv import load_dotenv
# REPLICATE_API_TOKEN = ...  in private_gpt/ui/.env 
load_dotenv()
# os.environ["REPLICATE_API_TOKEN"] # gets loaded using load_dotenv()

from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.replicate.base import (
    REPLICATE_MULTI_MODAL_LLM_MODELS,
)

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "Yamaha DocBot"

SOURCES_SEPARATOR = "\n\n Sources: \n"

MODES = ["Query Files", "Search Files", "LLM Chat (no context from files)"]

IS_JAP = False
LANGUAGES = ["English", "Japanese"]

def eng2jap(text: str) -> str:
    translated = GoogleTranslator(source="auto", target="ja").translate(text=text)
    return translated

def gen_from_vision(pdf_name,message,img_name: str | None = None, save_image_only: bool = True):
    # return "In the end, say thank you!"
    def load_database_pickle(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            data = pickle.load(f)
        # print(data['summary'])
        return data['pdf_name'], data['images'], data['embeddings']

    pdf_name, images, embeddings = load_database_pickle(f'/home/ub/Downloads/ash_temp/hack/yamaha-doc-bot/local_data/{pdf_name}.pkl')
    
    save_path=f'/tmp/hackathon/{img_name}'
    def save_image(query):
        # text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1', device='cuda:3')
        text_model=load_text_model.text_model

        query_emb = text_model.encode(query)

        # Instantiate a FAISS index
        # index = faiss.IndexFlatL2(embeddings.shape[1])

        # Add the embeddings to the index
        # index.add(embeddings)

        # Query
        # query_embedding = query_emb

        # Perform a k-nearest neighbor search
        k = 5  # Number of nearest neighbors to retrieve
        # distances, indices = index.search(np.array([query_embedding]), k)
        # print(indices, distances)
        # Then, we use the util.semantic_search function, which computes the cosine-similarity
        # between the query embedding and all image embeddings.
        # It then returns the top_k highest ranked images, which we output
        hits = util.semantic_search(query_emb, embeddings, top_k=k)[0]
        print(hits)
        ##################
        # page_idx=int(indices[0][0])  # least distance (top 1)

        # context_img=images[page_idx]
        # if len(hits):
        context_img=images[hits[0]['corpus_id']]
        context_img.save(save_path)  # Saves the Context image(having least distance in index.search)
        
    save_image(message)
    if(save_image_only):
        return ''


    multi_modal_llm = ReplicateMultiModal(
        model=REPLICATE_MULTI_MODAL_LLM_MODELS["llava-13b"],
        max_new_tokens=200,
        temperature=0.1,
    )

    def vision_gen_response(query,img_name: str | None = None):
        ##################
        # prompt = f"describe every component of the image in detail. Also answer this question in detail: Q:{query}"
        prompt = f"Answer this question in detail: Q:{query}"

        llava_response = multi_modal_llm.complete(
            prompt=prompt,
            image_documents=[ImageDocument(image_path=save_path)]  #img_paths[indices[0][0]])],
        )
        # return img_paths[indices[0][0]], img_embeddings[indices[0][0]], indices[0][0], llava_response.text
        return llava_response.text
    
    return vision_gen_response(message,img_name)

class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> list["Source"]:
        curated_sources = []

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"

            source = Source(file=file_name, page=page_label, text=chunk.text)
            curated_sources.append(source)
            curated_sources = list(
                dict.fromkeys(curated_sources).keys()
            )  # Unique sources only

        return curated_sources

def yeild_resp(message):
    yield message

@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service

        # Cache the UI blocks
        self._ui_block = None

        self._selected_filename = None

        # Initialize system prompt based on default mode
        self.mode = MODES[0]
        self._system_prompt = self._get_default_system_prompt(self.mode)

    def _chat(self, message: str, history: list[list[str]], mode: str, *_: Any) -> Any:
        def yield_deltas(completion_gen: CompletionGen, img_name: str | None = None) -> Iterable[str]:
            full_response: str = ""
            stream = completion_gen.response
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response
                time.sleep(0.02)

            if completion_gen.sources:

                import pdfplumber,pickle
                def list_pages(file_path: str) -> list:
                    pages_num=[]
                    with pdfplumber.open(file_path) as pdf:
                        for i in range(len(pdf.pages)):
                            page_num=pdf.pages[i].extract_text().split('\n')[-1]
                            if(sum([not v.isnumeric() for v in page_num.split('-')])):
                                page_num=str(i+1)

                            # print(page_num)
                            pages_num.append(page_num)
                    # print(pages_num)
                    return pages_num
                

                def display_images(pdf_name,page_idx,img_name):
    
                    # return "In the end, say thank you!"
                    def load_database_pickle(pickle_filename):
                        with open(pickle_filename, 'rb') as f:
                            data = pickle.load(f)
                        # print(data['summary'])
                        return data['pdf_name'], data['images'], data['embeddings']

                    # pdf_name, images, embeddings = load_database_pickle(f'/home/vinayak/Desktop/IIT/8/dl/project/yamaha-doc-bot/local_data/{pdf_name}.pkl')
                    pdf_name, images, embeddings = load_database_pickle(f'/home/ub/Downloads/ash_temp/hack/yamaha-doc-bot/local_data/{pdf_name}.pkl')

                    save_path=f'/tmp/hackathon/{img_name}'
                    images[page_idx-1].save(save_path)

                    return
                
                full_response += SOURCES_SEPARATOR
                cur_sources = Source.curate_sources(completion_gen.sources)
                sources_text = "\n\n\n"
                
                disp_img=f'<img src="context_images/{img_name}">' if img_name else ''
                used_files = set()
                for index, source in enumerate(cur_sources, start=1):
                    if f"{source.file}-{source.page}" not in used_files:
                        

                        first_page=7
                        npzfile=list_pages(str(f'/home/ub/Downloads/ash_temp/hack/yamaha-doc-bot/local_data/pdfs/{source.file}'))
                        print(npzfile)
                        map_pages=dict()
                        for i in range(len(npzfile)):
                            map_pages[npzfile[i]]=i+1-first_page+1
                        print(map_pages)
                        # print(f"map_pages['4-5']={map_pages['4-5']}")

                        page_idx=map_pages[source.page]
                        # page_idx=map_pages[source.page]

                        id=str(uuid.uuid4())
                        fname=source.file
                        display_images(fname,page_idx,f'{id}.png')
                        img_txt=f"{index}. {source.file} (<a href='context_images/{id}.png'> page {source.page} </a>) \n\n"

                        sources_text = (
                            sources_text
                            # + f"{index}. {source.file} (page {source.page}) \n\n"
                            + img_txt
                            + disp_img
                        )
                        used_files.add(f"{source.file}-{source.page}")
                full_response += sources_text
            yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = list(
                itertools.chain(
                    *[
                        [
                            ChatMessage(content=interaction[0], role=MessageRole.USER),
                            ChatMessage(
                                # Remove from history content the Sources information
                                content=interaction[1].split(SOURCES_SEPARATOR)[0],
                                role=MessageRole.ASSISTANT,
                            ),
                        ]
                        for interaction in history
                    ]
                )
            )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        # If a system prompt is set, add it as a system message
        if self._system_prompt:
            all_messages.insert(
                0,
                ChatMessage(
                    content=self._system_prompt,
                    role=MessageRole.SYSTEM,
                ),
            )
        match mode:
            case "Query Files":

                # Use only the selected file for the query
                context_filter = None
                if self._selected_filename is not None:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    context_filter=context_filter,
                )
                # print("CONTEXT-FILTER:--------------------------------\n",context_filter, '--------\n---*****--\n')
                # print("query_stream:--------------------------------\n",query_stream, '--------\n---*****--\n')
                # yield from yield_deltas(query_stream)
                ############ DEBUG: ###########
                # for deltas in yield_deltas(query_stream):
                #     print(f'deltas: {deltas}','\n-----\n')
                #     yield deltas
                # # import ...
                #     yield '\nHi man! Just testing sending response from Image embedding model\n'
                ###########
                ###########CHANGED
                resps=""
                for response in yield_deltas(query_stream,None): # don't display image in output
                    resps = response
                
                ############## Generate additional response from gen_from_vision() #############
                additional_response=[]
                img_name=str(uuid.uuid4())+'.png'
                print('\nBefore image',img_name,'-'*20,'\n')
                cur_sources = Source.curate_sources(query_stream.sources)
                used_files = set()
                for index, source in enumerate(cur_sources, start=1):
                    if f"{source.file}" not in used_files:

                        additional_response=gen_from_vision(source.file,message,img_name,save_image_only=False)

                        used_files.add(f"{source.file}")  # Unique files only
                
                # print(additional_response)

                full_response = "`This was your response just now:` "+ resps + "\n`Show the above list of sources (only pages) in your final response.`\n `Now here is some additional information from context for you, if you missed it, do not apologize just respond to the query precisely:` " + ''.join(additional_response)
                # print(query_stream)
                # print(f'MESSAGE IS THISSSS: \n\n{message}\n\n\n\n\n\n')
                # full_response = f"This was your current response: {resps}\nThis response may contain information about multiple queries, but you only have to answer about the query '{query_stream}'. Here I have a response from another agent for the same context, but it is only related to the query asked: {''.join(additional_response)}"
                # full_response = f"This was your current response: {resps}\n\nThis response may contain information about multiple queries, but you only have to answer about the query '{message}'. Applying a multimodal rag algorithm, here is refined and summarized information from the context itself about the query asked: {''.join(additional_response)}"
                # full_response = f"This was your current response: {resps}\nThis response may contain information about multiple queries. But you only have to answer about the query '{message}'. Applying confirmational analysis, the refined information about the query asked is: {''.join(additional_response)}."
                ####################################################

                # additional_response = gen_from_vision(message, response)
                # # Append the additional response to the original response
                # full_response = "This was your response just now: "+ resps + "\n\n Now I have some additional information for you: " + additional_response

                # print(f'FULL RESPONSE: -----{full_response}---------')

                # Send the combined response back to the LLM
                # new_message = ChatMessage(content=full_response, role=MessageRole.USER)
                all_messages.append(ChatMessage(content=full_response, role=MessageRole.USER))
                next_response = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    context_filter=context_filter,
                )
                final_resp = ""
                
                img_name=str(uuid.uuid4())+'.png'
                print('\nAfter image',img_name,'-'*20,'\n')
                for stream in yield_deltas(next_response,None):
                    final_resp = stream
                
                # final_resp += f'\n<img src="context_images/{img_name}">' if img_name else ''

                cur_sources = Source.curate_sources(next_response.sources)
                used_files = set()
                for index, source in enumerate(cur_sources, start=1):
                    if f"{source.file}" not in used_files:

                        additional_response=gen_from_vision(source.file,final_resp,img_name,save_image_only=True)

                        used_files.add(f"{source.file}")  # Unique files only
                
                # print(type(final_resp))
                
                # if(flag):
                #     translator = Translator()
                #     translated = translator.translate(final_resp, dest='ja')

                if IS_JAP:
                    yield from yeild_resp(eng2jap(final_resp))
                else:
                    yield from yeild_resp(final_resp)

                # yield from yield_deltas(next_response)
                ###########CHANGED
            case "LLM Chat (no context from files)":
                llm_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=False,
                )
                yield from yield_deltas(llm_stream)

            case "Search Files":
                response = self._chunks_service.retrieve_relevant(
                    text=message, limit=4, prev_next_chunks=0
                )

                sources = Source.curate_sources(response)

                yield "\n\n\n".join(
                    f"{index}. **{source.file} "
                    f"(page {source.page})**\n "
                    f"{source.text}"
                    for index, source in enumerate(sources, start=1)
                )

    # On initialization and on mode change, this function set the system prompt
    # to the default prompt based on the mode (and user settings).
    @staticmethod
    def _get_default_system_prompt(mode: str) -> str:
        p = ""
        match mode:
            # For query chat mode, obtain default system prompt from settings
            case "Query Files":
                p = settings().ui.default_query_system_prompt
            # For chat mode, obtain default system prompt from settings
            case "LLM Chat (no context from files)":
                p = settings().ui.default_chat_system_prompt
            # For any other mode, clear the system prompt
            case _:
                p = ""
        return p

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        logger.info(f"Setting system prompt to: {system_prompt_input}")
        self._system_prompt = system_prompt_input

    def _set_current_mode(self, mode: str) -> Any:
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        # Update placeholder and allow interaction if default system prompt is set
        if self._system_prompt:
            return gr.update(placeholder=self._system_prompt, interactive=True)
        # Update placeholder and disable interaction if no default system prompt is set
        else:
            return gr.update(placeholder=self._system_prompt, interactive=False)

    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            files.add(file_name)
        return [[row] for row in files]

    def _upload_file(self, files: list[str]) -> None:
        logger.debug("Loading count=%s files", len(files))
        paths = [Path(file) for file in files]

        # remove all existing Documents with name identical to a new file upload:
        file_names = [path.name for path in paths]
        doc_ids_to_delete = []
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"] in file_names
            ):
                doc_ids_to_delete.append(ingested_document.doc_id)
        if len(doc_ids_to_delete) > 0:
            logger.info(
                "Uploading file(s) which were already ingested: %s document(s) will be replaced.",
                len(doc_ids_to_delete),
            )
            for doc_id in doc_ids_to_delete:
                self._ingest_service.delete(doc_id)

        self._ingest_service.bulk_ingest([(str(path.name), path) for path in paths])

    def _delete_all_files(self) -> Any:
        ingested_files = self._ingest_service.list_ingested()
        logger.debug("Deleting count=%s files", len(ingested_files))
        for ingested_document in ingested_files:
            self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _delete_selected_file(self) -> Any:
        logger.debug("Deleting selected %s", self._selected_filename)
        # Note: keep looping for pdf's (each page became a Document)
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"]
                == self._selected_filename
            ):
                self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _set_curent_lang(self, lang) -> None:
        global IS_JAP
        IS_JAP = (lang == "Japanese")
        # print(self._is_jap)

    def _deselect_selected_file(self) -> Any:
        self._selected_filename = None
        return [
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _selected_a_file(self, select_data: gr.SelectData) -> Any:
        self._selected_filename = select_data.value
        return [
            gr.components.Button(interactive=True),
            gr.components.Button(interactive=True),
            gr.components.Textbox(self._selected_filename),
        ]

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Monochrome(primary_hue=slate),
            css=".logo { "
            "display:flex;"
            "background-color: rgb(28, 15, 85);"
            "height: 80px;"
            # "border-radius: 8px;"
            "align-content: center;"
            "justify-content: center;"
            "align-items: center;"
            "}"
            ".logo b { font-size: 2.5rem; color: #c5c4c4 }"
            ".contain { display: flex !important; flex-direction: column !important; }"
            "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
            "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
            "#col { height: calc(100vh - 112px - 16px) !important; }",
        ) as blocks:
            with gr.Row():
                gr.HTML(f"<div class='logo'><b>Yamaha DocBot</b></div>")
                # gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=DocBot></div")

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    mode = gr.Radio(
                        MODES,
                        label="Mode",
                        value="Query Files",
                        visible=False,
                    )
                    lang = gr.Radio(
                        LANGUAGES,
                        label="Language",
                        value="English"
                    )
                    lang.change(
                        self._set_curent_lang,
                        inputs=lang
                    )
                    upload_button = gr.components.UploadButton(
                        "Upload File(s)",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                    )
                    ingested_dataset = gr.List(
                        self._list_ingested_files,
                        headers=["File name"],
                        label="Ingested Files",
                        height=235,
                        interactive=False,
                        render=False,  # Rendered under the button
                    )
                    upload_button.upload(
                        self._upload_file,
                        inputs=upload_button,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.change(
                        self._list_ingested_files,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.render()
                    deselect_file_button = gr.components.Button(
                        "De-select selected file", size="sm", interactive=False
                    )
                    selected_text = gr.components.Textbox(
                        "All files", label="Selected for Query or Deletion", max_lines=1
                    )
                    delete_file_button = gr.components.Button(
                        "🗑️ Delete selected file",
                        size="sm",
                        visible=settings().ui.delete_file_button_enabled,
                        interactive=False,
                    )
                    delete_files_button = gr.components.Button(
                        "⚠️ Delete ALL files",
                        size="sm",
                        visible=settings().ui.delete_all_files_button_enabled,
                    )
                    deselect_file_button.click(
                        self._deselect_selected_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    ingested_dataset.select(
                        fn=self._selected_a_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_file_button.click(
                        self._delete_selected_file,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_files_button.click(
                        self._delete_all_files,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    system_prompt_input = gr.Textbox(
                        placeholder=self._system_prompt,
                        label="System Prompt",
                        lines=2,
                        interactive=True,
                        render=False,
                    )
                    # When mode changes, set default system prompt
                    mode.change(
                        self._set_current_mode, inputs=mode, outputs=system_prompt_input
                    )
                    # On blur, set system prompt to use in queries
                    system_prompt_input.blur(
                        self._set_system_prompt,
                        inputs=system_prompt_input,
                    )

                    def get_model_label() -> str | None:
                        """Get model label from llm mode setting YAML.

                        Raises:
                            ValueError: If an invalid 'llm_mode' is encountered.

                        Returns:
                            str: The corresponding model label.
                        """
                        # Get model label from llm mode setting YAML
                        # Labels: local, openai, openailike, sagemaker, mock, ollama
                        config_settings = settings()
                        if config_settings is None:
                            raise ValueError("Settings are not configured.")

                        # Get llm_mode from settings
                        llm_mode = config_settings.llm.mode

                        # Mapping of 'llm_mode' to corresponding model labels
                        model_mapping = {
                            "llamacpp": config_settings.llamacpp.llm_hf_model_file,
                            "openai": config_settings.openai.model,
                            "openailike": config_settings.openai.model,
                            "sagemaker": config_settings.sagemaker.llm_endpoint_name,
                            "mock": llm_mode,
                            "ollama": config_settings.ollama.llm_model,
                        }

                        if llm_mode not in model_mapping:
                            print(f"Invalid 'llm mode': {llm_mode}")
                            return None

                        return model_mapping[llm_mode]

                with gr.Column(scale=7, elem_id="col"):
                    # Determine the model label based on the value of PGPT_PROFILES
                    model_label = get_model_label()
                    if model_label is not None:
                        label_text = (
                            f"LLM: {settings().llm.mode} | Model: {model_label}"
                        )
                    else:
                        label_text = f"LLM: {settings().llm.mode}"

                    _ = gr.ChatInterface(
                        self._chat,
                        chatbot=gr.Chatbot(
                            label=None and label_text,
                            show_copy_button=True,
                            elem_id="chatbot",
                            render=False,
                            avatar_images=(
                                None,
                                AVATAR_BOT,
                            ),
                        ),
                        additional_inputs=[mode, upload_button, system_prompt_input],
                    )
        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
