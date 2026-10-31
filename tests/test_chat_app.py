import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest

from test_chat_store import CharacterTokenizer
from transformerx.chat_store import ChatStore
from transformerx.chat import ChatService, make_server


class TestEngine:
    """Replace only expensive pretrained generation; exercise real storage/API."""
    __test__ = False
    tokenizer = CharacterTokenizer()
    identity = 'trained-checkpoint'
    model_name = 'test-qwen'
    device = 'cpu'
    max_positions = 4096
    fail = False

    def reply(self, messages, **kwargs):
        if self.fail:
            raise RuntimeError('Generation failed')
        return dict(answer='Reply to ' + messages[-1]['content'],
                    neural_state=[.25, -.25] if kwargs['loop_enabled'] else kwargs['neural_state'],
                    neural_identity=self.identity if kwargs['loop_enabled'] else kwargs['neural_identity'],
                    generated_tokens=3, notice='')


def test_successful_turn_is_atomic_and_restart_preserves_memory(tmp_path):
    service = ChatService(ChatStore(tmp_path), TestEngine(), context_tokens=1000)
    session = service.store.create()
    result = service.send(session['id'], 0, 'Remember Vega', loop_enabled=True)
    saved = ChatStore(tmp_path).get(session['id'])
    assert len(saved['messages']) == 2
    assert saved['messages'][-1]['content'] == 'Reply to Remember Vega'
    assert saved['neural_state'] == [.25, -.25]
    assert result['session']['revision'] == 1
    with pytest.raises(ValueError, match='changed'):
        service.send(session['id'], 0, 'Stale tab')
    service.engine.fail = True
    with pytest.raises(RuntimeError, match='failed'):
        service.send(session['id'], 1, 'Should not be appended')
    assert service.store.get(session['id']) == saved


def test_reset_keeps_transcript_and_disabling_loop_preserves_state(tmp_path):
    service = ChatService(ChatStore(tmp_path), TestEngine(), context_tokens=1000)
    s = service.store.create()
    s = service.send(s['id'], 0, 'Hello', loop_enabled=True)['session']
    s = service.send(s['id'], 1, 'Again', text_memory=False, loop_enabled=False)['session']
    assert s['neural_state'] == [.25, -.25]
    s = service.change(s['id'], 2, notes='A useful note')
    s = service.change(s['id'], 3, reset_loop=True)
    assert s['neural_state'] is None
    assert len(s['messages']) == 4
    assert s['notes'] == 'A useful note'


def test_http_requires_same_origin_and_rejects_bad_inputs(tmp_path):
    service = ChatService(ChatStore(tmp_path), TestEngine(), context_tokens=1000)
    server = make_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def post(path, value, origin=base):
        return urlopen(Request(base + path, data=json.dumps(value).encode(),
            headers={'Content-Type': 'application/json', 'X-TransformerX': '1', 'Origin': origin}), timeout=10)
    try:
        with urlopen(base) as response:
            assert response.status == 200
            assert b'TransformerX' in response.read()
        with pytest.raises(HTTPError) as rejected:
            post('/api/new', {}, origin='https://unrelated.example')
        assert rejected.value.code == 403
        with post('/api/new', {}) as response:
            session = json.load(response)
        with post('/api/chat', dict(id=session['id'], revision=0, message='Hello', loop_enabled=False)) as response:
            assert json.load(response)['session']['messages'][-1]['content'] == 'Reply to Hello'
        for bad in [dict(id=session['id'], revision=1, message='Hi', temperature='bad'),
                    dict(id='../escape', revision=0, message='Hi'), []]:
            with pytest.raises(HTTPError) as rejected:
                post('/api/chat', bad)
            assert rejected.value.code == 400
        with urlopen(base + '/api/export?id=' + session['id']) as response:
            assert len(json.load(response)['messages']) == 2
        with post('/api/delete', dict(id=session['id'], revision=1)) as response:
            assert json.load(response)['deleted']
        assert service.store.list() == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
