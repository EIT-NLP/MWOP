import os
import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent
BUNDLE = WORKFLOW.parents[1]
for path in (BUNDLE / 'LLaVA-NeXT', BUNDLE / 'lmms-eval'):
    sys.path.insert(0, str(path))
os.environ.setdefault('PROJECT_ROOT', str(WORKFLOW))
os.environ['PYTHONPATH'] = os.pathsep.join([str(BUNDLE / 'LLaVA-NeXT'), str(BUNDLE / 'lmms-eval'), os.environ.get('PYTHONPATH', '')])
