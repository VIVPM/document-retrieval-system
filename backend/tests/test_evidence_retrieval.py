"""Bounded evidence retrieval must preserve source identity and never scan a packet."""

import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from core.answer_generator import build_prompt, build_sources
from core.chunker import _token_len
from core.evidence import (MAX_EVIDENCE_CHUNKS, MAX_EVIDENCE_TOKENS,
                           bounded_evidence, source_block)
from core.models import ChunkMetadata
from core.retriever import HybridRetriever


def chunk(index=0, doc='doc_0', text='Gross earnings | $4,375.00', page=0):
    return ChunkMetadata(chunk_id=f'chat#{doc}_chunk_{index}', doc_id=doc,
                         doc_type='Pay Slip', filename='packet.pdf', chunk_index=index,
                         page_start=page, page_end=page, text=text)


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.retriever = HybridRetriever.__new__(HybridRetriever)
        self.retriever.chat_id = 'chat'
        self.retriever.id_prefix = 'chat#'
        self.retriever.namespace = 'test-account'
        self.retriever.pc_index = MagicMock()
        self.retriever.alpha = 0.5
        self.retriever.chunk_count = 45

    def test_deduplicate_without_merging_people_or_pay_periods(self):
        candidates = [(chunk(), 1), (chunk(), 1), (chunk(1), 1),
                      (chunk(doc='doc_1'), 1), (chunk(2, page=1), 1)]
        selected, _ = bounded_evidence(candidates)
        self.assertEqual(len(selected), 3)
        self.assertEqual([c.doc_id for c, _ in selected], ['doc_0', 'doc_1', 'doc_0'])

    def test_hard_chunk_and_token_limits_keep_whole_chunks(self):
        candidates = [(chunk(i, text=f'Row {i} | ' + 'amount ' * 600), 1) for i in range(30)]
        selected, tokens = bounded_evidence(candidates)
        self.assertLessEqual(len(selected), MAX_EVIDENCE_CHUNKS)
        self.assertLessEqual(tokens, MAX_EVIDENCE_TOKENS)
        self.assertLessEqual(_token_len(''.join(source_block(c) for c, _ in selected)), MAX_EVIDENCE_TOKENS)
        self.assertEqual(selected[0][0].text, candidates[0][0].text)
        tiny, _ = bounded_evidence([(chunk(i, text=f'Row {i}'), 1) for i in range(30)])
        self.assertEqual(len(tiny), MAX_EVIDENCE_CHUNKS)

    def test_oversized_table_is_skipped_not_truncated(self):
        large = chunk(text='amount ' * (MAX_EVIDENCE_TOKENS + 1))
        small = chunk(1, text='Pay Frequency | Semi-monthly')
        selected, _ = bounded_evidence([(large, 1), (small, 0)])
        self.assertEqual(selected, [(small, 0)])

    def test_linked_fetch_is_bounded_and_missing_ids_are_normal(self):
        seed = chunk(5)
        related = chunk(1, text='Employee | Alice\nPay Frequency | Semi-monthly')
        metadata = {**asdict(related), 'chat_id': 'chat'}
        self.retriever.pc_index.fetch.return_value = {'vectors': {related.chunk_id: {'metadata': metadata}}}
        links = self.retriever._linked_chunks([(seed, 1)])
        self.assertEqual(links, [(related, 0.0)])
        args = self.retriever.pc_index.fetch.call_args.kwargs
        self.assertEqual(args['namespace'], 'test-account')
        self.assertEqual(args['_request_timeout'], 10)
        self.assertNotIn('timeout', args)
        self.assertTrue(all(vid.startswith('chat#doc_0_chunk_') for vid in args['ids']))
        self.assertLessEqual(len(args['ids']), 36)
        self.retriever.pc_index.list.assert_not_called()
        self.retriever.pc_index.query.assert_not_called()

    def test_wrong_chat_document_or_type_is_rejected(self):
        related = chunk(1)
        for field, wrong in [('chat_id', 'other-chat'), ('doc_id', 'doc_9'), ('doc_type', 'Bank Statement')]:
            with self.subTest(field=field):
                metadata = {**asdict(related), 'chat_id': 'chat', field: wrong}
                self.retriever.pc_index.fetch.return_value = {'vectors': {related.chunk_id: {'metadata': metadata}}}
                self.assertEqual(self.retriever._linked_chunks([(chunk(5), 1)]), [])

    def test_optional_fetch_failure_preserves_search_results(self):
        self.retriever.pc_index.fetch.side_effect = TimeoutError('test timeout')
        with self.assertLogs('drs.retriever', level='WARNING'):
            self.assertEqual(self.retriever._linked_chunks([(chunk(5), 1)]), [])

    def test_broader_search_keeps_explicit_filter_and_reports_budget(self):
        match = {'id': chunk().chunk_id, 'score': 0.8, 'metadata': asdict(chunk())}
        with patch.object(self.retriever, '_query_pinecone', return_value=[match]) as search, \
             patch.object(self.retriever, '_linked_chunks', return_value=[(chunk(1, text='Pay Frequency | Semi-monthly'), 0)]):
            result = self.retriever.retrieve('monthly gross', k=6, filter_doc_type='Pay Slip', return_details=True)
        search.assert_called_once_with(query='monthly gross', top_k=12, filter_doc_type='Pay Slip')
        self.assertEqual(len(result['results']), 2)
        self.assertLessEqual(result['retrieval_stats']['context_tokens'], MAX_EVIDENCE_TOKENS)
        self.retriever.pc_index.list.assert_not_called()

    def test_prompt_identifies_documents_and_related_citations(self):
        selected = [(chunk(), 0.9), (chunk(1, text='Pay Frequency | Semi-monthly'), 0)]
        prompt = build_prompt('Compare monthly income', selected)
        self.assertIn('Document: doc_0', prompt)
        self.assertIn('Pages: 1-1', prompt)
        self.assertIn('not proof of account ownership', prompt)
        self.assertIn('An anonymous amount in a different Document ID cannot', prompt)
        self.assertIn('not a\n   complete inventory', prompt)
        sources = build_sources(selected)
        self.assertEqual([s['relevance'] for s in sources], ['90.00%'])

    def test_source_cards_are_top_search_matches_only(self):
        hits = [(chunk(i), 0.9 - i / 100) for i in range(8)]
        linked = [(chunk(20 + i, text=f'Row {i}'), 0.0) for i in range(8)]
        sources = build_sources(hits + linked)
        self.assertEqual(len(sources), 6)
        self.assertEqual(sources[0]['relevance'], '90.00%')
        self.assertNotIn('Related evidence', [s['relevance'] for s in sources])
        self.assertIn('Row 7', build_prompt('q', hits + linked))
        self.assertEqual(len(build_sources(hits + linked, k=3)), 3)

    def test_answer_sources_stay_separate_from_prose(self):
        selected = [(chunk(), 0.9)]
        for summarize in (False, True):
            prompt = build_prompt('Explain with page citations', selected, summarize=summarize)
            self.assertIn('Source citations are rendered separately by the application', prompt)
            self.assertIn('even when the question asks for citations', prompt)
            self.assertIn('Pages: 1-1', prompt)
        self.assertEqual(build_sources(selected)[0]['pages'], '1-1')



if __name__ == '__main__':
    unittest.main()
