"""Private, atomic JSON conversations. No model downloads or tensor imports."""
import copy
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from datetime import datetime, timezone
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def _path(self, session_id):
        if not isinstance(session_id, str) or not re.fullmatch(r'[a-f0-9]{32}', session_id):
            raise ValueError('Invalid conversation ID')
        return self.root / f'{session_id}.json'

    def _write(self, session):
        payload = json.dumps(session, ensure_ascii=False, allow_nan=False, indent=2)
        path = self._path(session['id'])
        fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=self.root)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def create(self):
        with self.lock:
            session = dict(schema=1, id=uuid.uuid4().hex, revision=0, title='New conversation',
                           created_at=now(), updated_at=now(), notes='', messages=[],
                           neural_state=None, neural_identity=None)
            self._write(session)
            return session

    def get(self, session_id):
        with self.lock:
            with self._path(session_id).open(encoding='utf-8') as stream:
                value = json.load(stream)
            if value.get('schema') != 1 or value.get('id') != session_id:
                raise ValueError('Unsupported or damaged conversation file')
            return value

    def list(self):
        with self.lock:
            sessions = [self.get(p.stem) for p in self.root.glob('*.json')
                        if re.fullmatch(r'[a-f0-9]{32}', p.stem)]
            return sorted([dict(id=s['id'], title=s['title'], updated_at=s['updated_at'],
                                turns=len(s['messages']) // 2) for s in sessions],
                          key=lambda s: s['updated_at'], reverse=True)

    def check_revision(self, session_id, expected_revision):
        session = self.get(session_id)
        if type(expected_revision) is not int or session['revision'] != expected_revision:
            raise ValueError('Conversation changed in another tab. Reload it before trying again.')
        return session

    def save(self, session, *, expected_revision):
        with self.lock:
            self.check_revision(session['id'], expected_revision)
            saved = copy.deepcopy(session)
            saved['revision'] = expected_revision + 1
            saved['updated_at'] = now()
            self._write(saved)
            return saved

    def delete(self, session_id, *, expected_revision):
        with self.lock:
            self.check_revision(session_id, expected_revision)
            self._path(session_id).unlink()
