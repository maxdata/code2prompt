from pathlib import Path
from code2prompt.main import generate
from code2prompt.config import Configuration

if __name__ == "__main__":
    path = Path('code2prompt/main.py')
    config = Configuration(path=[path])
    generate({'config': config}, path=[path])
