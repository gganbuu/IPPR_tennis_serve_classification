import numpy as np
import main_model
from main_model import X_test, y_test, X_train, y_train

unique, counts = np.unique(y_train, return_counts=True)
print("Training set class distribution:", dict(zip(unique, counts)))

unique, counts = np.unique(y_test, return_counts=True)
print("Test set class distribution:", dict(zip(unique, counts)))
