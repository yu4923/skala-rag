"""대용량 payload는 불변 파일, 결정/오류 이력은 외부 JSONL로 저장한다."""
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'runs'
_LOG_LOCK = Lock()


class ArtifactStore:
    def __init__(self, run_id: str, root: Path = ROOT):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id):
            raise ValueError('Invalid run_id')
        self.directory = Path(root) / run_id
        self.directory.mkdir(parents=True, exist_ok=True)

    def put(self, value) -> str:
        content = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        name = hashlib.sha256(content.encode()).hexdigest() + '.json'
        target = self.directory / name
        if not target.exists():
            fd, temp = tempfile.mkstemp(dir=self.directory, suffix='.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    stream.write(content)
                os.replace(temp, target)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        return name

    def get(self, ref: str):
        if not re.fullmatch(r'[0-9a-f]{64}\.json', ref):
            raise ValueError('Invalid artifact reference')
        return json.loads((self.directory / ref).read_text(encoding='utf-8'))

    def log(self, context, event, **details):
        record = {'run_id': context['run_id'], 'trace_id': context['trace_id'], 'event': event, 'ts': datetime.now(timezone.utc).isoformat(), **details}
        with _LOG_LOCK, (self.directory / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
