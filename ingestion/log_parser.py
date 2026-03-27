import re
from datetime import datetime

# Regex for Django format:
# 2026-01-20 15:55:02,475 | INFO | base.add_job:507 | Adding job...
DJANGO_RE = re.compile(
    r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)'
    r'\s*\|\s*(?P<level>\w+)'
    r'\s*\|\s*(?P<module>[^:]+):(?P<line>\d+)'
    r'\s*\|\s*(?P<message>.+)'
)

# Regex for Celery format:
# [2026-01-20 15:11:44,923: INFO/ForkPoolWorker-5] [ENTITY: users] Starting...
CELERY_RE = re.compile(
    r'\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)'
    r':\s*(?P<level>\w+)/(?P<module>[^\]]+)\]'
    r'\s*(?P<message>.+)'
)

ENTITY_RE   = re.compile(r'\[ENTITY:\s*(\w+)\]')
WORKER_RE   = re.compile(r'ForkPoolWorker-(\d+)')
DBNAME_RE   = re.compile(r'(bridgesec_[\w\-T]+)')
DURATION_RE = re.compile(r'Time:\s*([\d.]+)s')
RECORDS_RE  = re.compile(r'(\d+)\s+records')
SUCCESS_RE  = re.compile(r'Successful:\s*(\d+)/(\d+)')

ERROR_PATTERNS = {
    'OktaTokenError'  : r'Error validating Okta token',
    'Forbidden'       : r'Forbidden:',
    'InternalError'   : r'Internal Server Error',
    'OktaAccessDenied': r'Okta API access denied',
    'CeleryMisuse'    : r'Never call result\.get\(\) within a task',
    'TypeError'       : r"TypeError:",
}

def parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None

    m = DJANGO_RE.match(line) or CELERY_RE.match(line)
    if not m:
        return None

    raw_ts  = m.group('timestamp').replace(',', '.')
    level   = m.group('level').upper()
    module  = m.group('module').strip()
    message = m.group('message').strip()
    line_no = int(m.groupdict().get('line', 0))

    try:
        timestamp = datetime.strptime(raw_ts, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        return None

    # Optional fields
    entity_m   = ENTITY_RE.search(message)
    worker_m   = WORKER_RE.search(module)
    dbname_m   = DBNAME_RE.search(message)
    duration_m = DURATION_RE.search(message)
    records_m  = RECORDS_RE.search(message)
    success_m  = SUCCESS_RE.search(message)

    error_type = None
    for name, pattern in ERROR_PATTERNS.items():
        if re.search(pattern, message):
            error_type = name
            break

    return {
        'timestamp'    : timestamp,
        'level'        : level,
        'module'       : module,
        'function'     : module,
        'line'         : line_no,
        'entity'       : entity_m.group(1)   if entity_m   else None,
        'worker_id'    : int(worker_m.group(1)) if worker_m else None,
        'message'      : message,
        'error_type'   : error_type,
        'db_name'      : dbname_m.group(1)   if dbname_m   else None,
        'duration_sec' : float(duration_m.group(1)) if duration_m else None,
        'record_count' : int(records_m.group(1))    if records_m  else None,
        'success_num'  : int(success_m.group(1))    if success_m  else None,
        'success_total': int(success_m.group(2))    if success_m  else None,
    }

def parse_file(path: str) -> list[dict]:
    results = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                results.append(parsed)
    return results