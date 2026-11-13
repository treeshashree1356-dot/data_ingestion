"""Website loader using LangChain WebBaseLoader."""

from typing import Any
import requests

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from ingestion.document_loader import BaseDocumentLoader, DocumentLoadingError
from utils.validators import InvalidURLError, validate_url


class WebDocumentLoader(BaseDocumentLoader):
    """Loader for web documents using LangChain WebBaseLoader."""

    DEFAULT_USER_AGENT = "DataIngestionPipeline/1.0 (Mozilla/5.0 compatible)"

    def load(
        self,
        source: str | list[str],
        continue_on_failure: bool = False,
        **kwargs: Any,
    ) -> list[Document]:
        """Extract text from one or more website URLs into LangChain Documents.

        Args:
            source: A single URL string or a list of URL strings.
            continue_on_failure: If True, skip failed URLs and return successfully loaded ones.
            **kwargs: Extra parameters passed to WebBaseLoader.

        Returns:
            List of LangChain Document objects.

        Raises:
            DocumentLoadingError: If a URL is malformed, inaccessible, or empty (when continue_on_failure is False).
        """
        urls = [source] if isinstance(source, str) else list(source)
        if not urls:
            raise DocumentLoadingError("No URLs provided for web loading.")

        documents: list[Document] = []
        errors: list[str] = []

        for url in urls:
            try:
                # 1. Validate URL format
                try:
                    cleaned_url = validate_url(url)
                except (InvalidURLError, Exception) as val_err:
                    raise DocumentLoadingError(f"Malformed URL '{url}': {val_err}") from val_err

                # 2. Configure WebBaseLoader with User-Agent
                requests_kwargs = kwargs.get("requests_kwargs", {})
                headers = requests_kwargs.setdefault("headers", {})
                headers.setdefault("User-Agent", self.DEFAULT_USER_AGENT)

                loader = WebBaseLoader(
                    web_path=cleaned_url,
                    requests_kwargs=requests_kwargs,
                    **{k: v for k, v in kwargs.items() if k != "requests_kwargs"},
                )

                # 3. Load documents
                loaded = loader.load()

                if not loaded or all(d.page_content.strip() == "" for d in loaded):
                    raise DocumentLoadingError(
                        f"Webpage at '{cleaned_url}' has no extractable text content."
                    )

                # 4. Standardize metadata
                for doc in loaded:
                    doc.metadata["source"] = cleaned_url
                    doc.metadata["source_type"] = "web"
                    doc.metadata["char_count"] = len(doc.page_content)
                    documents.append(doc)

            except DocumentLoadingError as dle:
                errors.append(str(dle))
                if not continue_on_failure:
                    raise dle
            except (requests.RequestException, Exception) as exc:
                err_msg = f"Failed to access URL '{url}': {exc}"
                errors.append(err_msg)
                if not continue_on_failure:
                    raise DocumentLoadingError(err_msg) from exc

        if not documents and errors:
            raise DocumentLoadingError(f"All URL extractions failed: {'; '.join(errors)}")

        return documents
