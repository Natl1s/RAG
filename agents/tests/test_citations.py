from __future__ import annotations

import unittest

from agents.citations import (
    SourceRef,
    append_sources,
    build_source_index,
    humanize_citations,
)
from agents.state import ContextChunk


def make_chunk(
    doc_id: str,
    chunk_id: str,
    label: str,
    *,
    title: str = "",
    chunk_ids: list[str] | None = None,
) -> ContextChunk:
    metadata: dict[str, object] = {"source_label": label}
    if title:
        metadata["document_title"] = title
    if chunk_ids is not None:
        metadata["chunk_ids"] = chunk_ids
    return ContextChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        text="Текст",
        metadata=metadata,
        subquery_ids=[1],
    )


class BuildSourceIndexTest(unittest.TestCase):
    def test_maps_doc_chunk_and_merged_chunk_ids(self) -> None:
        chunk = make_chunk(
            "law.txt",
            "law.txt_chunk_7",
            "152-ФЗ, ст. 9",
            chunk_ids=["law.txt_chunk_7", "law.txt_chunk_8"],
        )

        index = build_source_index([chunk])

        for key in (
            "law.txt",
            "law.txt_chunk_7",
            "law.txt_chunk_8",
            "152-ФЗ, ст. 9",
        ):
            self.assertIn(key, index)
            self.assertEqual(index[key].label, "152-ФЗ, ст. 9")


class HumanizeCitationsTest(unittest.TestCase):
    def test_replaces_doc_id_with_label(self) -> None:
        chunk = make_chunk("doc-1", "chunk-1", "Договор аренды № А-08/2024")

        answer, refs = humanize_citations(
            "Аренда 450 000 рублей [doc_id: doc-1].",
            [chunk],
        )

        self.assertEqual(answer, "Аренда 450 000 рублей [Договор аренды № А-08/2024].")
        self.assertEqual([ref.label for ref in refs], ["Договор аренды № А-08/2024"])

    def test_collapses_merged_chunks_of_one_document(self) -> None:
        chunk = make_chunk(
            "law.txt",
            "law.txt_chunk_7",
            "152-ФЗ, ст. 9",
            chunk_ids=["law.txt_chunk_7", "law.txt_chunk_8"],
        )

        answer, refs = humanize_citations(
            "Текст [doc_id: law.txt_chunk_7, law.txt_chunk_8].",
            [chunk],
        )

        self.assertEqual(answer, "Текст [152-ФЗ, ст. 9].")
        self.assertEqual([ref.label for ref in refs], ["152-ФЗ, ст. 9"])

    def test_handles_multiple_documents(self) -> None:
        law = make_chunk("law.txt", "law-1", "152-ФЗ, ст. 9")
        code = make_chunk("koap.txt", "koap-1", "КоАП, ст. 13.11")

        answer, refs = humanize_citations(
            "Норма [doc_id: law-1]; ответственность [chunk_id: koap-1].",
            [law, code],
        )

        self.assertEqual(
            answer,
            "Норма [152-ФЗ, ст. 9]; ответственность [КоАП, ст. 13.11].",
        )
        self.assertEqual(
            [ref.label for ref in refs],
            ["152-ФЗ, ст. 9", "КоАП, ст. 13.11"],
        )

    def test_keeps_unknown_citation(self) -> None:
        chunk = make_chunk("doc-1", "chunk-1", "Документ")

        answer, refs = humanize_citations(
            "Текст [doc_id: unknown].",
            [chunk],
        )

        self.assertEqual(answer, "Текст [doc_id: unknown].")
        self.assertEqual(refs, [])

    def test_deduplicates_repeated_sources(self) -> None:
        chunk = make_chunk("doc-1", "chunk-1", "Документ")

        _, refs = humanize_citations(
            "Раз [doc_id: doc-1]. Два [doc_id: doc-1].",
            [chunk],
        )

        self.assertEqual([ref.label for ref in refs], ["Документ"])

    def test_normalizes_source_prefix(self) -> None:
        chunk = make_chunk("doc-1", "chunk-1", "Документ")

        answer, refs = humanize_citations(
            "Текст [source: Документ].",
            [chunk],
        )

        self.assertEqual(answer, "Текст [Документ].")
        self.assertEqual([ref.label for ref in refs], ["Документ"])

    def test_collects_labels_already_present_in_answer(self) -> None:
        chunk = make_chunk("doc-1", "chunk-1", "Документ")

        answer, refs = humanize_citations(
            "Текст [Документ].",
            [chunk],
        )

        self.assertEqual(answer, "Текст [Документ].")
        self.assertEqual([ref.label for ref in refs], ["Документ"])


class AppendSourcesTest(unittest.TestCase):
    def test_appends_sources_block(self) -> None:
        answer = append_sources(
            "Ответ.",
            [
                SourceRef("152-ФЗ, ст. 9", "Статья 9. Согласие субъекта"),
                SourceRef("Договор аренды № А-08/2024", "Договор аренды"),
            ],
        )

        self.assertIn("Ответ.", answer)
        self.assertIn("Источники:", answer)
        self.assertIn("- 152-ФЗ, ст. 9 — Статья 9. Согласие субъекта", answer)
        self.assertIn("- Договор аренды № А-08/2024 — Договор аренды", answer)

    def test_omits_title_when_it_matches_label(self) -> None:
        answer = append_sources(
            "Ответ.",
            [SourceRef("152-ФЗ, ст. 9", "152-ФЗ, ст. 9")],
        )

        self.assertIn("- 152-ФЗ, ст. 9\n", answer + "\n")
        self.assertNotIn("— 152-ФЗ", answer)

    def test_omits_long_contract_header(self) -> None:
        answer = append_sources(
            "Ответ.",
            [
                SourceRef(
                    "Договор аренды № А-08/2024",
                    "ДОГОВОР АРЕНДЫ НЕЖИЛОГО ПОМЕЩЕНИЯ № А-08/2024 "
                    "Московская область, г. Химки — 1 августа 2024 г. "
                    "ООО «Северный Склад», именуемое в дальнейшем «Арендодатель»",
                )
            ],
        )

        self.assertIn("- Договор аренды № А-08/2024", answer)
        self.assertNotIn("Московская область", answer)

    def test_does_not_duplicate_existing_block(self) -> None:
        text = "Ответ.\n\nИсточники:\n- 152-ФЗ, ст. 9"

        answer = append_sources(text, [SourceRef("КоАП, ст. 13.11")])

        self.assertEqual(answer, text)

    def test_returns_answer_when_no_sources(self) -> None:
        self.assertEqual(append_sources("Ответ.", []), "Ответ.")


if __name__ == "__main__":
    unittest.main()
