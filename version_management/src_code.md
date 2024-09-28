## src/alg.py
```python
import numpy 
from src.base import *
def vec_add(a, b):
    return a.v + b.v
```

## src/base.py
```python
import numpy as np
class vol:
    def __init__(self, n=3):
        self.v = np.zeros((n,n))
```

