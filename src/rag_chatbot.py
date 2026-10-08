"""Provide retrieval-augmented chat over uploaded AURA OS retail orders.

The service converts validated CSV rows into Chroma documents, retrieves relevant
order evidence for each question, and uses GPT-4o to produce a cited answer with
JSON-safe chart data.
"""

from __future__ import annotations

import logging
import math
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)


class AURAOrdersRAG:
    """Index uploaded order rows and answer questions with cited evidence."""

    def __init__(self, persist_directory: str = "./.aura_chroma") -> None:
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.persist_directory / "current_collection.txt"
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the AURA OS retail analyst. Answer only from the supplied order rows. "
                    "Do not invent values or claim a result is the maximum across the full dataset "
                    "unless the evidence proves it. If the evidence is insufficient or a date range "
                    "is ambiguous, say so. Cite supporting rows using their [S#] source IDs.",
                ),
                ("human", "Order rows:\n{context}\n\nQuestion: {question}"),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.collection_name = self._read_collection_name()
        self.vector_store = self._create_vector_store(self.collection_name)

    def _read_collection_name(self) -> str:
        if self._manifest_path.is_file():
            name = self._manifest_path.read_text(encoding="utf-8").strip()
            if name.startswith("aura_orders_"):
                return name
        return "aura_orders"

    def _create_vector_store(self, collection_name: str) -> Chroma:
        return Chroma(
            collection_name=collection_name,
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
        )

    @staticmethod
    def _safe_value(value: Any) -> str:
        if pd.isna(value):
            return "Unknown"
        return str(value)

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _resolve_columns(dataframe: pd.DataFrame) -> dict[str, str]:
        available = {str(column).strip().casefold(): str(column) for column in dataframe.columns}
        aliases = {
            "date": ("order date", "date"),
            "product": ("product name", "product"),
            "sales": ("sales",),
            "profit": ("profit",),
        }
        resolved = {
            field: next((available[name] for name in names if name in available), "")
            for field, names in aliases.items()
        }
        missing = [field for field, column in resolved.items() if not column]
        if missing:
            required = "Date/Order Date, Product/Product Name, Sales, and Profit"
            raise ValueError(f"Orders CSV must include {required}. Missing: {', '.join(missing)}.")
        return resolved

    @classmethod
    def _row_to_text(cls, row: dict[str, Any], columns: dict[str, str]) -> str:
        date = cls._safe_value(row.get(columns["date"]))
        product = cls._safe_value(row.get(columns["product"]))
        sales = cls._safe_value(row.get(columns["sales"]))
        profit = cls._safe_value(row.get(columns["profit"]))
        text = f"On {date}, {product} sold for {sales} with profit {profit}."

        details = []
        for field in ("Order ID", "Category", "Sub-Category", "Region"):
            value = row.get(field)
            if value is not None and not pd.isna(value):
                details.append(f"{field}: {value}.")
        return " ".join([text, *details])

    def build_documents(self, csv_path: str, source_name: str | None = None) -> list[Document]:
        """Read the CSV and create one searchable document per order row."""
        try:
            dataframe = pd.read_csv(csv_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            dataframe = pd.read_csv(csv_path, encoding="cp1252")
        dataframe.columns = [str(column).strip() for column in dataframe.columns]
        if dataframe.empty:
            raise ValueError("Orders CSV is empty. Upload a file containing order rows.")

        columns = self._resolve_columns(dataframe)
        file_name = source_name or Path(csv_path).name
        documents = []
        for row_index, row in dataframe.iterrows():
            values = row.to_dict()
            metadata: dict[str, str | int | float] = {
                "source": file_name,
                "row_number": int(row_index) + 2,
                "product": self._safe_value(values.get(columns["product"])),
                "date": self._safe_value(values.get(columns["date"])),
            }
            for field in ("sales", "profit"):
                number = self._number(values.get(columns[field]))
                if number is not None:
                    metadata[field] = number
            for field in ("Order ID", "Category", "Sub-Category", "Region"):
                value = values.get(field)
                if value is not None and not pd.isna(value):
                    metadata[field.lower().replace(" ", "_")] = str(value)

            documents.append(
                Document(page_content=self._row_to_text(values, columns), metadata=metadata)
            )
        return documents

    def ingest_csv(self, csv_path: str, source_name: str | None = None) -> int:
        """Replace the active index with one document for each uploaded CSV row."""
        documents = self.build_documents(csv_path, source_name=source_name)
        if not documents:
            raise ValueError("No order rows were found in the CSV.")

        new_collection_name = f"aura_orders_{uuid.uuid4().hex[:16]}"
        new_vector_store = self._create_vector_store(new_collection_name)
        try:
            for start in range(0, len(documents), 500):
                batch = documents[start : start + 500]
                identifiers = [f"row-{start + offset}" for offset in range(len(batch))]
                new_vector_store.add_documents(batch, ids=identifiers)

            temporary_manifest = self._manifest_path.with_suffix(".tmp")
            temporary_manifest.write_text(new_collection_name, encoding="utf-8")
            temporary_manifest.replace(self._manifest_path)
        except Exception:
            try:
                new_vector_store.delete_collection()
            except Exception:
                logger.warning("Could not remove incomplete Chroma collection %s", new_collection_name)
            raise

        previous_vector_store = self.vector_store
        self.vector_store = new_vector_store
        self.collection_name = new_collection_name
        try:
            previous_vector_store.delete_collection()
        except Exception:
            logger.warning("Could not remove previous Chroma collection %s", self.collection_name)
        return len(documents)

    def ask(self, question: str, k: int = 5) -> dict[str, Any]:
        """Retrieve order rows and return a grounded answer, citations, and chart data."""
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")
        if not 1 <= k <= 20:
            raise ValueError("Retrieval count must be between 1 and 20.")

        results = self.vector_store.similarity_search(clean_question, k=k)
        if not results:
            return {
                "answer": "No matching order rows were found. Upload an Orders CSV before asking a question.",
                "sources": [],
                "chart_data": [],
            }

        context = "\n".join(
            f"[S{index}] {document.page_content}"
            for index, document in enumerate(results, start=1)
        )
        answer = self.chain.invoke({"context": context, "question": clean_question}).strip()
        sources = []
        for index, document in enumerate(results, start=1):
            metadata = document.metadata
            sources.append(
                {
                    "id": f"S{index}",
                    "source": metadata.get("source", "Uploaded Orders CSV"),
                    "row_number": metadata.get("row_number"),
                    "product": metadata.get("product", "Unknown"),
                    "date": metadata.get("date", "Unknown"),
                    "sales": self._number(metadata.get("sales")),
                    "profit": self._number(metadata.get("profit")),
                    "text": document.page_content,
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "chart_data": self._build_chart_data(results),
        }

    @classmethod
    def _build_chart_data(cls, results: list[Document]) -> list[dict[str, Any]]:
        """Build JSON-safe chart records from retrieved rows."""
        return [
            {
                "label": document.metadata.get("product", "Unknown"),
                "date": document.metadata.get("date", "Unknown"),
                "sales": cls._number(document.metadata.get("sales")),
                "profit": cls._number(document.metadata.get("profit")),
            }
            for document in results
        ]


if __name__ == "__main__":
    AURAOrdersRAG()
    print("AURA OS orders RAG service is ready.")
