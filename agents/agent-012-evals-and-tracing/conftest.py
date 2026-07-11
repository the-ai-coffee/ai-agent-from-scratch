import os
import sys

print('CONFTEST agents/agent-012-evals-and-tracing/conftest.py pop, sys.path[0:2]=', sys.path[0:2], 'agent in sys.modules:', 'agent' in sys.modules)
sys.modules.pop("agent", None)
sys.path.insert(0, os.path.dirname(__file__))
