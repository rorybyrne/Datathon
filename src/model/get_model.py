from .keras import Kegression
from .random_forest import RandomForest

def from_name(name):
    if(name == "kegression"):
        return Kegression
    if(name == "random_forest"):
        return RandomForest