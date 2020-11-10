"""
Provides a class for model accuracy evaluation.
"""

from __future__ import absolute_import

import numpy as np

from evaldnn.utils import common


class all_probs_logger:
    """ Class for model accuracy evaluation.

    Compare the predictions and the labels, update and report the model
    prediction accuracy accordingly.

    Parameters
    ----------
    ks : list of integers
        For each k in ks, top-k accuracy will be computed separately.

    """

    def __init__(self):
        self.log_array_y_true = None
        self.log_array_y_pred = None
        self.size = 0
        # TODO: save all probs

    def update(self, y_true, y_pred):
        """Update model accuracy accordingly.

        For each k in ks, the correctness and accuracy will be re-calculated
        and updated accordingly.

        Parameters
        ----------
        y_true : array
            Labels for data.
        y_pred : array
            Predictions from model.

        Notes
        -------
        This method can be invoked for many times in one instance which means
        that once a batch prediction is made this method can be invoked to update
        the status. The accuracy will be updated for every invocation.

        """
        y_true = common.to_numpy(y_true)
        y_pred = common.to_numpy(y_pred)

        size = len(y_true)

        if self.size == 0:
            self.log_array_y_true = np.copy(y_true)
            self.log_array_y_pred = np.copy(y_pred)
        else:
            # print(y_true)
            # print(self.log_array_y_true)
            self.log_array_y_true = np.concatenate([self.log_array_y_true, y_true])
            self.log_array_y_pred = np.vstack([self.log_array_y_pred, y_pred])

        self.size += size

        if self.size % 4000 == 0:
            print("Processed: {}/50000".format(self.size))
            # print(self.log_array_y_true.shape)
            # print(self.log_array_y_pred.shape)

    def save(self, filename):
        print("saving to {}".format(filename))
        np.savez(filename, y_ture = self.log_array_y_true, y_pred = self.log_array_y_pred)
