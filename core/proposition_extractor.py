import re

FRAME_PATTERNS = [
    (re.compile(r'\bif\b.*\bthen\b', re.I), 'Conditional', 'IF_THEN'),
    (re.compile(r'\bwhere\b.*\bthere\b', re.I), 'Evidence', 'EVIDENCE_TO_CAUSE'),
    (re.compile(r'\bno\b.+\bwithout\b', re.I), 'Requirement', 'REQUIRES'),
    (re.compile(r'\b(\w+)\s+is\s+(\w+)', re.I), 'Attribution', 'IS_A'),
    (re.compile(r'\b(never|always|don\'t|do not)\b', re.I), 'Norm', 'NORM')
]

def normalize(t: str) -> str:
    return re.sub(r'\s+', ' ', str(t)).strip()

def extract_proposition(text: str, nlp_provider=None):
    t = normalize(text)
    frame, ftype = 'Generic', 'UNKNOWN'
    for rx, fr, ft in FRAME_PATTERNS:
        if rx.search(t):
            frame, ftype = fr, ft
            break

    if ftype == 'IF_THEN':
        formula = 'IF(X) THEN Y'
    elif ftype == 'EVIDENCE_TO_CAUSE':
        formula = 'EVIDENCE_SIGN(X) -> CAUSE(Y)'
    elif ftype == 'REQUIRES':
        formula = 'REQUIRES(X, Y)'
    elif ftype == 'IS_A':
        formula = 'IS_A(X, Y)'
    elif ftype == 'NORM':
        formula = 'SHOULD(Agent, Action)'
    else:
        formula = 'REL(X, Y)'

    return {'idea_formula': formula, 'frame': frame, 'roles': {'X': None, 'Y': None}}
