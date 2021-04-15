#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def H(z):
    num = z**5 - z**4 + z**3 - z**2
    denom = z**5 + 0.54048*z**4 - 0.62519*z**3 - 0.66354*z**2 + 0.60317*z + 0.69341
    return num/denom

w_range = np.linspace(0, 2*np.pi, 1000)
plt.plot(w_range, np.abs(H(np.exp(1j*w_range))))
plt.show()

plt.plot(w_range, np.abs(signal.freqz(np.exp(1j*w_range))))
plt.show()