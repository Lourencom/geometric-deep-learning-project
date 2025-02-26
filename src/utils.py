import subprocess
import os

def get_git_root():
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
        return git_root
    except subprocess.CalledProcessError:
        return None  # Not inside a Git repository
    
def relative_to_absolute_path(path):
    return os.path.join(get_git_root(), path)
