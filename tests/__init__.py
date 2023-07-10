import sys
from pathlib import Path
import os
testdir = os.path.dirname(__file__)
ku_srcdir = testdir+'/..'
sys.path.insert(0, str(Path(os.path.abspath(ku_srcdir))))
srcdir = ku_srcdir+'/..'
print(srcdir)
sys.path.insert(0, str(Path(os.path.abspath(srcdir))))