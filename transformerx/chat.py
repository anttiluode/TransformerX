"""Run the local persistent Qwen chatbot: python -m transformerx.chat."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .chat_context import build_context
from .chat_store import ChatStore


class ChatService:
    def __init__(self, store, engine, *, context_tokens=1536):
        if context_tokens < 128 or context_tokens >= engine.max_positions:
            raise ValueError('Prompt budget must be at least 128 and below the model context limit.')
        self.store, self.engine = store, engine
        self.context_tokens = context_tokens
        self.lock = threading.RLock()

    def status(self):
        return dict(model=self.engine.model_name, device=self.engine.device,
                    loop_available=self.engine.identity is not None,
                    controller_identity=self.engine.identity, context_tokens=self.context_tokens)

    def send(self, session_id, revision, message, *, text_memory=True, loop_enabled=False,
             max_new_tokens=256, temperature=.7):
        if not isinstance(message, str) or not message.strip() or len(message) > 32000:
            raise ValueError('Message must contain 1–32000 characters.')
        if type(text_memory) is not bool or type(loop_enabled) is not bool:
            raise ValueError('Memory toggles must be booleans.')
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2048:
            raise ValueError('Response length must be between 1 and 2048 tokens.')
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError('Temperature must be between 0 and 2.')
        with self.lock:
            session = self.store.check_revision(session_id, revision)
            messages, diagnostics = build_context(self.engine.tokenizer, session, message,
                max_tokens=min(self.context_tokens, self.engine.max_positions - max_new_tokens),
                text_memory=text_memory)
            started = time.monotonic()
            result = self.engine.reply(messages, neural_state=session['neural_state'],
                neural_identity=session['neural_identity'], loop_enabled=loop_enabled,
                max_new_tokens=max_new_tokens, temperature=temperature)
            diagnostics.update(elapsed_seconds=round(time.monotonic() - started, 2),
                               generated_tokens=result['generated_tokens'], notice=result['notice'],
                               text_memory=text_memory, loop_enabled=loop_enabled)
            session['messages'].extend([dict(role='user', content=message.strip()),
                                        dict(role='assistant', content=result['answer'], diagnostics=diagnostics)])
            session['neural_state'] = result['neural_state']
            session['neural_identity'] = result['neural_identity']
            if session['title'] == 'New conversation':
                session['title'] = message.strip().replace('\n', ' ')[:60]
            saved = self.store.save(session, expected_revision=revision)
            return dict(session=saved, diagnostics=diagnostics)

    def change(self, session_id, revision, *, notes=None, reset_loop=False):
        with self.lock:
            session = self.store.check_revision(session_id, revision)
            if notes is not None:
                if not isinstance(notes, str) or len(notes) > 8000:
                    raise ValueError('Memory notes must be text of at most 8000 characters.')
                session['notes'] = notes
            if reset_loop:
                session['neural_state'] = None
                session['neural_identity'] = None
            return self.store.save(session, expected_revision=revision)


def make_server(service, port=7860):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Do not log private conversation paths/content.

        def respond(self, status, value, content_type='application/json; charset=utf-8'):
            payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                             "style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; "
                             "base-uri 'none'; form-action 'self'")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def allowed(self, mutation=False):
            hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in hosts:
                return False  # Also blocks DNS-rebinding hostnames.
            if mutation:
                return (self.headers.get('Origin') in {'http://' + h for h in hosts}
                        and self.headers.get('X-TransformerX') == '1'
                        and self.headers.get_content_type() == 'application/json')
            return True

        def do_GET(self):
            if not self.allowed():
                self.respond(403, {'error': 'Localhost access only.'})
                return
            url = urlsplit(self.path)
            try:
                if url.path == '/':
                    self.respond(200, files('transformerx').joinpath('chat_ui.html').read_bytes(),
                                 'text/html; charset=utf-8')
                elif url.path == '/api/status':
                    self.respond(200, service.status())
                elif url.path == '/api/sessions':
                    self.respond(200, service.store.list())
                elif url.path in ('/api/session', '/api/export'):
                    session_id = parse_qs(url.query).get('id', [''])[0]
                    self.respond(200, service.store.get(session_id))
                else:
                    self.respond(404, {'error': 'Not found'})
            except FileNotFoundError:
                self.respond(404, {'error': 'Conversation not found.'})
            except ValueError as exc:
                self.respond(400, {'error': str(exc)})

        def do_POST(self):
            if not self.allowed(mutation=True):
                self.respond(403, {'error': 'Same-origin JSON requests only.'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 100000:
                    raise ValueError('Request body is empty or too large.')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Request must be a JSON object.')
                path = urlsplit(self.path).path
                with service.lock:
                    if path == '/api/new':
                        result = service.store.create()
                    elif path == '/api/chat':
                        result = service.send(body.get('id'), body.get('revision'), body.get('message'),
                            text_memory=body.get('text_memory', True), loop_enabled=body.get('loop_enabled', False),
                            max_new_tokens=body.get('max_new_tokens', 256), temperature=body.get('temperature', .7))
                    elif path in ('/api/notes', '/api/reset-loop'):
                        if path == '/api/notes' and 'notes' not in body:
                            raise ValueError('Missing notes field')
                        result = service.change(body.get('id'), body.get('revision'),
                            notes=body.get('notes') if path == '/api/notes' else None,
                            reset_loop=path == '/api/reset-loop')
                    elif path == '/api/delete':
                        service.store.delete(body.get('id'), expected_revision=body.get('revision'))
                        result = {'deleted': True}
                    else:
                        self.respond(404, {'error': 'Not found'})
                        return
                self.respond(200, result)
            except FileNotFoundError:
                self.respond(404, {'error': 'Conversation not found. Reload the conversation list.'})
            except (ValueError, TypeError, KeyError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception as exc:
                self.respond(500, {'error': f'{type(exc).__name__}: {exc}. '
                    'Reload this conversation to check its saved state before retrying. '
                    'For GPU memory errors, lower --context-tokens and response length.'})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3-0.6B')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--controller', help='Trained Qwen chat controller.pt (not a classifier checkpoint)')
    parser.add_argument('--data-dir', default='chat-data')
    parser.add_argument('--context-tokens', type=int, default=1536)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535.')
    try:
        import torch
        from .chat_model import QwenChatEngine
        torch.set_num_threads(4)
        print(f'Loading {args.model}. The first launch downloads weights from Hugging Face.', flush=True)
        engine = QwenChatEngine(args.model, device=args.device, checkpoint=args.controller,
                                local_files_only=args.local_files_only)
        service = ChatService(ChatStore(Path(args.data_dir).resolve()), engine,
                               context_tokens=args.context_tokens)
        server = make_server(service, args.port)
    except (ImportError, ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\nInstall chat dependencies with: python -m pip install -e ".[chat]"\n')
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Open {url}\nDevice: {engine.device}. Conversations: {Path(args.data_dir).resolve()}\n'
          f'Neural loop: {"trained controller available" if engine.identity else "unavailable; text memory only"}\n'
          'Press Ctrl+C to stop. Saved conversations remain on disk.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
