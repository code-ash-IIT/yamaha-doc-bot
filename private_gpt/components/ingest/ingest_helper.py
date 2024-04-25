import logging
from pathlib import Path

from llama_index.core.readers import StringIterableReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

import io
import fitz  # PyMuPDF library

class DocumentWithBlobs(Document):
    doc_blobs: list[bytes] = []

    def __init__(self, **data):
        super().__init__(**data)
        self.doc_blobs = data.get('doc_blobs', [])

class PyMuPDFReader:
    """PDF parser using PyMuPDF."""

    def __init__(self, return_full_document: bool = False):
        """
        Initialize PyMuPDFReader.
        """
        self.return_full_document = return_full_document

    def load_data(self, file_path: str) -> list:
        """Parse file."""
        docs = []

        # Load the file using PyMuPDF
        with open(file_path, "rb") as fp:
            pdf_doc = fitz.open(stream=fp.read(), filetype="pdf")

            # This block returns a whole PDF as a single Document
            if self.return_full_document:
                text = ""
                metadata = {"file_name": file_path}

                for page in range(len(pdf_doc)):
                    # Extract the text from the page
                    page_text = pdf_doc[page].get_text()
                    text += page_text

                docs.append(Document(text=text, metadata=metadata))

            # This block returns each page of a PDF as its own Document
            else:
                # Iterate over every page
                for page in range(len(pdf_doc)):
                    # Extract the text from the page
                    page_text = pdf_doc[page].get_text()

                    # # Extract images from the page
                    # page_images = []
                    # for img in pdf_doc.get_page_images(page):
                    #     xref = img[0]
                    #     image = pdf_doc.extract_image(xref)
                    #     image_bytes = image["image"]
                    #     page_images.append(image_bytes)

                    metadata = {"file_name": file_path, "page_num": page}

                    # Create a document for text
                    text_doc = Document(text=page_text, metadata=metadata)
                    docs.append(text_doc)

                    # # Create a document for images
                    # if page_images:
                    #     image_doc = Document(text="", doc_blobs=page_images, metadata=metadata)
                    #     docs.append(image_doc)

        return docs


# Inspired by the `llama_index.core.readers.file.base` module
def _try_loading_included_file_formats() -> dict[str, type[BaseReader]]:
    try:
        from llama_index.readers.file.docs import (  # type: ignore
            DocxReader,
            HWPReader,
            PDFReader,
        )
        from llama_index.readers.file.epub import EpubReader  # type: ignore
        from llama_index.readers.file.image import ImageReader  # type: ignore
        from llama_index.readers.file.ipynb import IPYNBReader  # type: ignore
        from llama_index.readers.file.markdown import MarkdownReader  # type: ignore
        from llama_index.readers.file.mbox import MboxReader  # type: ignore
        from llama_index.readers.file.slides import PptxReader  # type: ignore
        from llama_index.readers.file.tabular import PandasCSVReader  # type: ignore
        from llama_index.readers.file.video_audio import (  # type: ignore
            VideoAudioReader,
        )
    except ImportError as e:
        raise ImportError("`llama-index-readers-file` package not found") from e

    default_file_reader_cls: dict[str, type[BaseReader]] = {
        ".hwp": HWPReader,
        # ".pdf": PDFReader,
        ".pdf": PyMuPDFReader,
        ".docx": DocxReader,
        ".pptx": PptxReader,
        ".ppt": PptxReader,
        ".pptm": PptxReader,
        ".jpg": ImageReader,
        ".png": ImageReader,
        ".jpeg": ImageReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
        ".csv": PandasCSVReader,
        ".epub": EpubReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".ipynb": IPYNBReader,
    }
    return default_file_reader_cls


# Patching the default file reader to support other file types
FILE_READER_CLS = _try_loading_included_file_formats()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
    }
)


class IngestionHelper:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    def transform_file_into_documents(
        file_name: str, file_data: Path
    ) -> list[Document]:
        documents = IngestionHelper._load_file_to_documents(file_name, file_data)
        for document in documents:
            document.metadata["file_name"] = file_name
        IngestionHelper._exclude_metadata(documents)
        return documents

    @staticmethod
    def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        extension = Path(file_name).suffix
        reader_cls = FILE_READER_CLS.get(extension)
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )
            # Read as a plain text
            string_reader = StringIterableReader()
            return string_reader.load_data([file_data.read_text()])

        logger.debug("Specific reader found for extension=%s", extension)
        return reader_cls().load_data(file_data)

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
