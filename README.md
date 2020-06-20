```python
%matplotlib inline
```


```python
import numpy as np
import matplotlib.pyplot as plt

import gif
```


```python
from pynobody import Body, Orbit
```


```python
body_1 = Body(name='body_1', mass=1., position=np.array([0.5, 0, 0]),
              velocity=np.array([0, 7.071067811865475e-01, 0]))

body_2 = Body(name='body_2', mass=1., position=np.array([-0.5, 0, 0]),
              velocity=np.array([0, -7.071067811865475e-01, 0]))

body_3 = Body(name='body_3', mass=1., position=np.array([0, 0.5, 0]),
              velocity=np.array([0, 0, 7.071067811865475e-01]))

body_4 = Body(name='body_4', mass=1., position=np.array([0, -0.5, 0]),
              velocity=np.array([0, 0, -7.071067811865475e-01]))
```


```python
orb = Orbit(dt_param=0.01, dt_out=0.01, dt_tot=4)

bodies = [body_1, body_2, body_3, body_4]
orb.add_body(bodies)

orb.run()
```

    100% (400 of 400) |######################| Elapsed Time: 0:00:03 Time:  0:00:03
    


```python
b1 = orb.get_state(body_1)
b2 = orb.get_state(body_2)
b3 = orb.get_state(body_3)
b4 = orb.get_state(body_4)

c = ['r', 'g', 'b', 'm']

bs = [b1, b2, b3, b4]

@gif.frame
def plot(i):
    plt.figure(figsize=(10, 10), dpi=72)

    for j, b in enumerate(bs):
        plt.plot(b['x'][:i], b['y'][:i], c[j], lw=1)
        plt.plot(b['x'][i], b['y'][i], 'ko', ms=3)

    plt.xlim((-0.75, 0.75))
    plt.ylim((-0.75, 0.75))


frames = list()
for i in range(len(b1['x'])):
    frame = plot(i)
    frames.append(frame)

gif.save(frames, "chaos.gif", duration=100)
```


```python
from IPython.display import Image
```


```python
Image("chaos.gif")
```


<img src="./misc/chaos.gif" width="500" height="500" />


```python

```
