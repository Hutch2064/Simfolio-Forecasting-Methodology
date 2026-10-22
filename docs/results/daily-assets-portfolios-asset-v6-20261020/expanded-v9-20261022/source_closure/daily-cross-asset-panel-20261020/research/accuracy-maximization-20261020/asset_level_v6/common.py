"""Versioned research paths; financial factories remain the sealed originals."""
from pathlib import Path
import sys

OWN = Path(__file__).resolve().parent
CAMPAIGN = next(p for p in OWN.parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = CAMPAIGN.parent/'french551-full-20261019/code'
for path in (CAMPAIGN, CODE/'src', CODE/'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
