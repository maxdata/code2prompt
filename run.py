from pathlib import Path
from github.main import generate
from github.config import Configuration

if __name__ == "__main__":
    path = Path('code2prompt')
    output = Path('output.md')
    exclude = ['*/tests/*']
    config = Configuration(path=[path], output=output, exclude=exclude)
    generate({'config': config}, path=[path])
