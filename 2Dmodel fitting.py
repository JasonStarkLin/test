import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

# Generate fake data
np.random.seed(0)

x, y = np.meshgrid(np.linspace(-1,1,128), np.linspace(-1,1,128))
d = np.sqrt(x*x+y*y)
sigma, mu = 1.0, 0.0
y, x = np.mgrid[:128, :128]
z = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
#z = 2. * x ** 2 - 0.5 * x ** 2 + 1.5 * x * y - 1.
#z += np.random.normal(0., 0.1, z.shape) * 50000.

# Fit the data using astropy.modeling
#p_init = models.Polynomial2D(degree=2)
p_init = models.Gaussian2D()

fit_p = fitting.LevMarLSQFitter()

with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
    warnings.simplefilter('ignore')
    p = fit_p(p_init, x, y, z)

# Plot the data with the best-fit model
plt.figure(figsize=(8, 2.5))
plt.subplot(1, 3, 1)
plt.imshow(z, origin='lower', interpolation='nearest')
plt.title("Data")
plt.subplot(1, 3, 2)
plt.imshow(p(x, y), origin='lower', interpolation='nearest')
plt.title("Model")
plt.subplot(1, 3, 3)
plt.imshow(z - p(x, y), origin='lower', interpolation='nearest')
plt.title("Residual")
plt.show()