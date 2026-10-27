"""Offline transport regressions; native consumer acceptance is independent."""
import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
import providers


def complete(output):
    return {'id': 'r1', 'status': 'completed', 'output': output, 'usage': None,
            'error': None, 'incomplete_details': None}


def message(text, phase='final_answer'):
    return {'type': 'message', 'role': 'assistant', 'status': 'completed',
            'phase': phase, 'content': [{'type': 'output_text', 'text': text}]}


def call(name='pin', arguments='{"string":"literal"}', id='c1'):
    return {'type': 'function_call', 'call_id': id, 'name': name,
            'arguments': arguments, 'status': 'completed'}


class ResponseTests(unittest.TestCase):
    def test_final_is_never_a_command(self):
        response = providers.validate_response(complete([message('(shell dangerous)')]))
        self.assertEqual(response.calls, [])
        self.assertEqual(response.final_text, '(shell dangerous)')

    def test_full_transport_is_retained(self):
        raw = complete([{'type': 'reasoning', 'encrypted_content': 'opaque'},
                        message('analysis', 'commentary'), message('final')])
        raw['usage'] = {'input_tokens': 7, 'output_tokens': 9}
        response = providers.validate_response(raw)
        self.assertEqual(response.raw, raw)
        raw['output'].clear()
        self.assertEqual(len(response.raw['output']), 3)
        self.assertEqual(response.commentary, 'analysis')
        self.assertEqual(response.final_text, 'final')

    def test_unknown_batch_rejected_before_exposing_calls(self):
        response = providers.validate_response(complete([call(), call('unknown', '{}', 'c2')]))
        self.assertEqual(response.status, 'malformed')
        self.assertEqual(response.calls, [])

    def test_bad_arguments(self):
        for arguments in ('{"string":3}', '{', '{"other":"x"}',
                          '{"string":"a","string":"b"}'):
            with self.subTest(arguments=arguments):
                self.assertEqual(providers.validate_response(complete([call(arguments=arguments)])).status,
                                 'malformed')

    def test_duplicate_ids(self):
        self.assertEqual(providers.validate_response(complete([call(), call()])).calls, [])

    def test_dynamic_schema_and_argument_order(self):
        providers.register_skill('custom', 'Description', ['second', 'first'])
        try:
            schema = next(t for t in providers.tool_schemas() if t['name'] == 'custom')
            self.assertEqual(schema['parameters']['required'], ['second', 'first'])
            response = providers.validate_response(complete([call('custom', '{"first":"A","second":"B"}')]))
            self.assertEqual(response.calls, [['c1', 'custom', ['B', 'A']]])
        finally:
            providers.remove_skill('custom')

    def test_noncompleted_status_and_unknown_usage(self):
        for status in ('incomplete', 'cancelled', 'failed', 'in_progress', 'queued'):
            raw = complete([])
            raw.update(status=status, incomplete_details={'reason': 'max_output_tokens'})
            response = providers.validate_response(raw)
            self.assertEqual(response.status, status)
            self.assertIsNone(response.raw['usage'])
            self.assertEqual(response.raw['incomplete_details'], raw['incomplete_details'])

    def test_refusal_preserved(self):
        item = message('unused')
        item['content'] = [{'type': 'refusal', 'refusal': 'Cannot comply'}]
        self.assertEqual(providers.validate_response(complete([item])).refusal, 'Cannot comply')

    def test_legacy_explicit_only(self):
        class Legacy(providers.LLMProvider):
            def chat(self, *args):
                return 'write-file "file name" "two\nlines"\nsend literal'
        raw = Legacy().respond('prompt')
        # JSON legacy quoted values use escaped newlines.
        raw['text'] = 'write-file "file name" two\nlines\nsend literal'
        response = providers.validate_response(raw)
        self.assertEqual(response.kind, 'legacy_commands')
        self.assertEqual(response.calls[0][2], ['file name', 'two\nlines'])
        self.assertEqual(providers.validate_response('send literal').status, 'malformed')

    def test_start_and_transport_failures(self):
        class Broken(providers.LLMProvider):
            def start(self):
                raise RuntimeError('start failed')
        providers.registerLLMProvider('broken-test', Broken())
        self.assertEqual(providers.llmProviderStart('broken-test'), 'start_error')
        self.assertEqual(providers.llmProviderRespond('p').status, 'start_error')
        class Transport(providers.LLMProvider):
            def respond(self, *args):
                raise RuntimeError('transport failed')
        providers.registerLLMProvider('transport-test', Transport())
        providers.llmProviderStart('transport-test')
        self.assertEqual(providers.llmProviderRespond('p', '6').status, 'provider_error')

    def test_private_exception_text_is_not_response_history(self):
        secret = "PRIVATE_SENTINEL credential=request-secret"
        class Broken(providers.LLMProvider):
            def start(self):
                raise RuntimeError(secret)
        providers.registerLLMProvider('private-start-test', Broken())
        providers.llmProviderStart('private-start-test')
        record = providers.response_record(providers.llmProviderRespond('new input'))
        self.assertNotIn(secret, record)
        self.assertIn('RuntimeError', record)
        self.assertIn('start_error', record)
        class BrokenTransport(providers.LLMProvider):
            def respond(self, *args):
                raise OSError(secret)
        providers.registerLLMProvider('private-transport-test', BrokenTransport())
        providers.llmProviderStart('private-transport-test')
        record = providers.response_record(providers.llmProviderRespond('new input'))
        self.assertNotIn(secret, record)
        self.assertIn('OSError', record)

    def test_replay_failure_retains_complete_results_until_new_input(self):
        class LostReceipt(providers.LLMProvider):
            failed = True
            received = None
            requested = False
            def tool_results(self, results):
                if self.failed:
                    raise OSError('PRIVATE_SENTINEL replay credentials')
                self.received = results
            def respond(self, *args):
                self.requested = True
                return complete([message('recovered')])
        provider = LostReceipt()
        providers.registerLLMProvider('replay-test', provider)
        providers.llmProviderStart('replay-test')
        outcomes = [['send', '(one two)'], ['c2', '(singleton)']]
        self.assertEqual(providers.llmProviderToolResults(outcomes), 'replay_error')
        self.assertEqual(providers.llmProviderEndTurn('replay_error'), 'replay_error')
        self.assertEqual(providers.llmProviderRespond('p').status, 'replay_error')
        self.assertFalse(provider.requested)
        provider.failed = False
        self.assertEqual(providers.llmProviderRespond('new input').status, 'completed')
        self.assertEqual(provider.received, outcomes)
        self.assertTrue(provider.requested)

    def test_legacy_quoted_multiline_and_partial_outer_parenthesis(self):
        examples = [
            ('(- Found a bug', ['pin', ['Found a bug']]),
            ('(send _quote_literal_quote_)', ['send', ['literal']]),
            ('send "first\nsecond"', ['send', ['"first\nsecond"']]),
            ('send ""', ['send', ['']]),
            ('write-file "file name" ""', ['write-file', ['file name', '']]),
            ('metta (+ 1 2)', ['metta', ['(+ 1 2)']]),
            ('send "hello" world', ['send', ['"hello" world']]),
        ]
        for text, expected in examples:
            with self.subTest(text=text):
                self.assertEqual(providers._legacy_calls(text)[0][1:], expected)

    def test_openai_complete_transport_and_replay(self):
        # No SDK or network required: test the producer's exact model_dump use,
        # request arguments and mechanical replay, not SDK compatibility.
        fake_sdk = types.ModuleType('openai')
        fake_sdk.OpenAI = object
        with patch.dict(sys.modules, {'openai': fake_sdk}):
            sys.path.insert(0, str(ROOT / 'providers'))
            try:
                spec = importlib.util.spec_from_file_location('candidate_openai', ROOT / 'providers/openai.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            finally:
                sys.path.pop(0)
        raw = complete([{'type':'reasoning', 'encrypted_content':'retained'}, call()])
        requests = []
        class Result:
            def model_dump(self, mode):
                self.mode = mode
                return copy.deepcopy(raw)
        class Responses:
            def create(self, **kwargs):
                requests.append(copy.deepcopy(kwargs))
                return Result()
        provider = module.OpenAIProviderImpl('OpenAI', 'UNSET_TEST_KEY', 'test-model', 'unused')
        provider._client = types.SimpleNamespace(responses=Responses())
        self.assertEqual(provider.respond('context :-:-:-: input', tools=providers.tool_schemas()), raw)
        provider.tool_results([['c1', ['result', 'full']]])
        raw['output'] = [message('done')]
        provider.respond('context :-:-:-: next')
        replay = requests[1]['input']
        self.assertEqual(replay[1]['encrypted_content'], 'retained')
        self.assertEqual(replay[3]['call_id'], 'c1')
        self.assertEqual(replay[3]['output'], '["result", "full"]')
        self.assertIn('reasoning.encrypted_content', requests[0]['include'])
        provider.tool_results([['c1', ['result', 'full']]])
        outputs = [item for item in provider._replay if item.get('type') == 'function_call_output']
        self.assertEqual(len(outputs), 1)
        provider._replay.append(call(id='unobserved'))
        provider.end_turn('interrupted')
        unresolved = provider._replay[-1]['output']
        self.assertIn('unknown', unresolved)
        self.assertNotIn('not_executed', unresolved)


if __name__ == '__main__':
    unittest.main()
