import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import numpy as np

from library.data.tokenizer import Tokenizer
from library.evaluation.evaluation import evaluate


class CallbackSettings:
    def __init__(self, log_path, hdf_name, lr_pat, lr_fac, es_pat):
        self.log_path = log_path
        self.hdf_path = hdf_name
        self.lr_patience = lr_pat
        self.lr_factor = lr_fac
        self.es_patience = es_pat


class PerformanceCallback(Callback):
    """A callback that computes the model's performance after every epoch."""

    def __init__(self, model, val_sequence, charset, verbose=True):
        """
        :param val_sequence: The sequence containing validation data.
        :param labels: The ground truth labels for the validation data.
        """
        self.ctc_model = model
        self.val_sequence = val_sequence
        self.labels = val_sequence.get_ground_truths()
        self.tokenizer = Tokenizer(charset)
        self.verbose = verbose
        self.history = list()

    def on_epoch_end(self, epoch, logs=None):
        """
        :param epoch: The current epoch.
        :param logs: The logs for the current epoch.
        """

        # Save the original sequence state to reset it after prediction.
        seq_state = self.val_sequence.predict
        self.val_sequence.predict = True

        if self.verbose:
            print('\nCalling Performance Callback')

        # Predict the validation sequence.
        predictions = self.ctc_model.predict_generator(
            generator=self.val_sequence,
            steps=len(self.val_sequence),
            verbose=1 if self.verbose else 0
        )

        # Create a result list consisting of triples of shape (label, prediction, probability)
        results = [(label, self.tokenizer.decode(pred), prob) for label, (pred, prob) in zip(self.labels, predictions)]

        # Evaluate the predictions by printing the word accuracy and character error rate.
        wer, cer = evaluate(results)

        # Add the current WER and CER to the history
        self.history.append((wer, cer))

        # Reset the sequence to its original state to prevent conflicts.
        self.val_sequence.predict = seq_state


class CyclicLR(Callback):
    """
    Source: https://github.com/bckenstler/CLR

    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments

    """

    def __init__(self, test_seq, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.test_seq = test_seq
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    """
    def on_epoch_end(self, epoch, logs=None):
        self.history.setdefault('val_loss', []).append(logs['val_loss'])
    """

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        """
        val_loss = self.model.evaluate(
            x=self.test_seq,
            steps=len(self.test_seq),
            verbose=0
        )

        print()
        print(val_loss)
        """

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        #self.history.setdefault('val_loss', []).append(val_loss)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
