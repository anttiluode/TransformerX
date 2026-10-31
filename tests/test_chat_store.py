import json
import pytest
from transformerx.chat_store import ChatStore
from transformerx.chat_context import build_context


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(''.join(f"<{m['role']}>{m['content']}" for m in messages) + '<assistant>')


def test_session_survives_restart_and_rejects_stale_update(tmp_path):
    store = ChatStore(tmp_path)
    session = store.create()
    stale = dict(session)
    session['notes'] = 'Antti prefers Finnish.'
    session['messages'] = [{'role': 'user', 'content': 'Hei'}, {'role': 'assistant', 'content': 'Moi'}]
    session['neural_state'] = [.1, -.2]
    session['neural_identity'] = 'checkpoint-a'
    saved = store.save(session, expected_revision=0)
    loaded = ChatStore(tmp_path).get(session['id'])
    assert loaded == saved
    assert loaded['revision'] == 1
    assert loaded['notes'] == 'Antti prefers Finnish.'
    assert loaded['neural_state'] == [.1, -.2]
    with pytest.raises(ValueError, match='changed'):
        store.save(stale, expected_revision=0)
    assert store.get(session['id']) == loaded


def test_invalid_session_path_and_failed_serialization_preserve_data(tmp_path):
    store = ChatStore(tmp_path)
    session = store.create()
    with pytest.raises(ValueError):
        store.get('../outside')
    session['neural_state'] = [float('nan')]
    with pytest.raises(ValueError):
        store.save(session, expected_revision=0)
    assert store.get(session['id'])['neural_state'] is None
    assert len(store.list()) == 1
    store.delete(session['id'], expected_revision=0)
    assert store.list() == []


def test_recall_old_exchange_and_memory_off_has_no_history(tmp_path):
    session = ChatStore(tmp_path).create()
    session['notes'] = 'Use Finnish.'
    session['messages'] = [
        {'role': 'user', 'content': 'The telescope project is called Vega.'},
        {'role': 'assistant', 'content': 'Understood.'},
        {'role': 'user', 'content': 'What is two plus two?'},
        {'role': 'assistant', 'content': 'Four.'}]
    messages, info = build_context(CharacterTokenizer(), session, 'What was the telescope called?',
                                   max_tokens=3000, recent_turns=1)
    assert 'Vega' in messages[0]['content']
    assert info['recalled_turns'] == [0]
    assert messages[-1]['content'] == 'What was the telescope called?'
    messages, info = build_context(CharacterTokenizer(), session, 'Hello',
                                   max_tokens=3000, text_memory=False)
    assert len(messages) == 2
    assert 'Finnish' not in messages[0]['content']
    assert info['recalled_turns'] == []


def test_budget_drops_history_without_cutting_current_request(tmp_path):
    session = ChatStore(tmp_path).create()
    session['notes'] = 'A' * 8000
    session['messages'] = [{'role': 'user', 'content': 'x' * 3000},
                           {'role': 'assistant', 'content': 'y' * 3000}]
    messages, info = build_context(CharacterTokenizer(), session, 'Latest request', max_tokens=700)
    assert len(CharacterTokenizer().apply_chat_template(messages)) <= 700
    assert messages[-1] == {'role': 'user', 'content': 'Latest request'}
    assert info['notes_truncated']
    with pytest.raises(ValueError, match='too long'):
        build_context(CharacterTokenizer(), session, 'z' * 1000, max_tokens=700)
