from pathlib import Path
from code2prompt.main import generate
from code2prompt.config import Configuration

if __name__ == "__main__":
    path = Path('code2prompt')
    output = Path('output.md')
    exclude = ['*/tests/*']
    config = Configuration(path=[path], output=output, exclude=exclude)
    generate({'config': config}, path=[path])
