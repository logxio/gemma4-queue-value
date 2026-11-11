"""Word count of the paper as a reader sees it (link targets removed, tables included)."""
import re
from pathlib import Path

text = (Path(__file__).resolve().parents[1] / "paper" / "writeup.md").read_text()
visible = re.sub(r"\]\([^)]*\)", "]", text)
words = re.findall(r"[A-Za-z0-9][^\s|*#()\[\]]*", visible)
print(len(words))
