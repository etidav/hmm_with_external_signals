import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List, Tuple, Type, Union


class Hidden_Markov_Model(ABC, tf.keras.Model):
    """
    Abstract class for HMM models.
    """

    def __init__(
        self, nb_hidden_states: int = 2, past_dependency: int = 1, season: int = 1
    ) -> None:
        """
        Instantiate a HMM model with gaussian emission laws and discrete hidden states.
        
        Arguments:
    
        - *nb_hidden_states*: number of hidden states of the HMM.
        
        - *past_dependency*: In case of a ARHMM, define the past dependency length.
        
        - *season*: In case of a SHMM, define the seasonality length.
        """
        super(Hidden_Markov_Model, self).__init__()
        self.K = nb_hidden_states
        self.past_dependency = past_dependency
        self.season = season
        self.auto_regressive = False
        self.define_param()
        self.init_param()

    @abstractmethod
    def define_param(self) -> None:
        """
        Define parameters of the HMM model. Have to be define : 
            - pi_var parametrizing the emission law of the first hidden state X_0
            - delta_var parametrizing the mean of the gaussian emission laws.
            - sigma_var parametrizing the standard deviation of the gaussian emission laws.
            - omega_var parametrizing the transition matrices.
        """
        pass

    @abstractmethod
    def pi(self) -> tf.Tensor:
        """
        Compute the emission probabilities of the first hidden variable values X_0
         
        Returns:
        
        - *pi*: a tf.Tensor(nb_hidden_stat,) containing the nb_hidden_states emission probabilities
        """
        pass

    @abstractmethod
    def mu(self, t: tf.Tensor, w: tf.Tensor, y_past: tf.Tensor) -> tf.Tensor:
        """
        Compute the mean of the emission densities
        
         Arguments:
        
        - *t*: tf.Tensor(nb_time_step,) corresponding to the time step in a Seasonal Hidden Markov Model. 
            component
        
        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the values of the external signal in Hidden Markov Models with 
            external variable.
            
        - *y_past*: tf.Tensor(nb_time_step, past_dependency) containing the past values of the main signal Y_{t-m} in AutoRegressive 
            Hidden Markov Models.
         
        Returns:
        
        - *mu*: a tf.Tensor(nb_hidden_states,nb_time_step) containing the mean of the emission densities.
        """
        pass

    @abstractmethod
    def sigma(self, t: tf.Tensor, w: tf.Tensor, y_past: tf.Tensor) -> tf.Tensor:
        """
        Compute the standard deviation of the emission densities
        
         Arguments:
        
        - *t*: tf.Tensor(nb_time_step,) corresponding to the time step in a Seasonal Hidden Markov Model. 
            component
        
        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the values of the external signal in Hidden Markov Models with 
            external variable.
        
        - *y_past*: tf.Tensor(nb_time_step, past_dependency) containing the past values of the main signal Y_{t-m} in AutoRegressive 
            Hidden Markov Models.
         
        Returns:
        
        - *sigma*: a tf.Tensor(nb_hidden_states,) containing the standard deviation of the emission densities
        """
        pass

    @abstractmethod
    def tp(self, t: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
        """
        Compute the transition matrix
        
         Arguments:
        
        - *t*: tf.Tensor(nb_time_step,) corresponding to the time step in a Seasonal Hidden Markov Model. 
            component
        
        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the values of the external signal in Hidden Markov Models with 
            external variable.
         
        Returns:
        
        - *tp*: a tf.Tensor(nb_time_step,nb_hidden_states,nb_hidden_states) containing the transition probabilities
        """
        pass

    def init_param(self) -> None:
        """
        Initialize parameters of the HMM model 
        """
        pi = np.random.uniform(-1, 1, self.K - 1)
        delta = np.random.uniform(-1, 1, self.K * self.delta_var.shape[1])
        sigma = np.random.uniform(-1, 1, self.K)
        omega = np.random.uniform(
            -1, 1, self.K * (self.K - 1) * self.omega_var.shape[2]
        )
        self.pi_var.assign(pi)
        self.delta_var.assign(tf.reshape(delta, [self.K, self.delta_var.shape[1]]))
        self.sigma_var.assign(sigma)
        self.omega_var.assign(tf.reshape(omega, self.omega_var.shape))

    def assign_param(
        self, pi: np.ndarray, delta: np.ndarray, sigma: np.ndarray, omega: np.ndarray
    ) -> None:
        """
        Assign pre-define parameters of the HMM model 
        
        Arguments:
        
        - *pi*: np.ndarray containing the emission law of the hidden process X
        
        - *delta*: np.nbdrray containing the mean values of the emission distributions
        
        - *sigma*: np.ndarray containing the standard deviation values of the emission distributions
        
        - *omega*: np.ndarray containing the transition matrix values
        """

        self.pi_var.assign(tf.math.log(pi[:-1] / (1 - pi[:-1])))
        self.delta_var.assign(delta)
        self.sigma_var.assign(tf.math.log(sigma))
        self.omega_var.assign(omega)

    def get_param(self, return_numpy=True) -> Tuple:
        """
        get the HMM model parameters
        
        Returns:
        
        - *param*: a Tuple containing the HMM model parameters
        """
        if return_numpy:
            return (
                self.pi().numpy(),
                self.delta_var.numpy(),
                self.sigma().numpy(),
                self.omega_var.numpy(),
            )
        else:
            return (
                self.pi(),
                self.delta_var,
                self.sigma(),
                self.omega_var,
            )

    def emission(
        self, k: int, t: tf.Tensor, w: tf.Tensor, y_past: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute a realisation of one of the emission densities
        
        Arguments:
        
        - *k*: int corresponding to the hidden state current value
        
        - *t*: tf.Tensor(nb_time_step,) corresponding to the time step in a Seasonal Hidden Markov Model. 
            component
        
        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the values of the external signal in Hidden Markov Models with 
            external variable.
            
        - *y_past*: tf.Tensor(nb_time_step, past_dependency) containing the past values of the main signal Y_{t-m} in AutoRegressive 
            Hidden Markov Models.
         
        Returns:
        
        - *realisation_value*: a tf.Tensor(nb_time_step,) containing a realisation of one of the emission densities
        """
        return np.random.normal(
            loc=self.mu(
                tf.cast(t, dtype=tf.float64),
                tf.cast(w, tf.float64),
                tf.cast(y_past, tf.float64),
            )[k],
            scale=self.sigma(
                tf.cast(t, dtype=tf.float64),
                tf.cast(w, tf.float64),
                tf.cast(y_past, tf.float64),
            )[k],
        )

    def emission_prob(
        self,
        k: int,
        y: tf.Tensor,
        t: tf.Tensor,
        w: tf.Tensor,
        y_past: tf.Tensor,
        apply_log: bool = False,
    ) -> tf.Tensor:
        """
        Compute the probability that a value have been realised by one of the emission densities
        
        Arguments:
        
        - *k*: int corresponding to the hidden state value.
        
        - *y*: tf.Tensor(nb_time_step,) containing values for which we want to know the emission probability.
        
        - *t*: tf.Tensor(nb_time_step,) corresponding to the time step in a Seasonal Hidden Markov Model. 
            component.
        
        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the values of the external signal in Hidden Markov Models with 
            external variable.
            
        - *y_past*: tf.Tensor(nb_time_step, past_dependency) containing the past values of the main signal Y_{t-m} in AutoRegressive 
            Hidden Markov Models.
        
        - *apply_log*: boolean indicating if the probability or the log probability has to be return.
        
        Returns:
        
        - *probability_value*: a tf.Tensor containing the probability that a value has been realised by one of the emission densities.
        """

        if not apply_log:
            return (
                1
                / tf.math.sqrt(
                    2
                    * np.pi
                    * tf.math.pow(
                        self.sigma(tf.cast(t, dtype=tf.float64), w, y_past)[k], 2
                    )
                )
            ) * tf.math.exp(
                -((y - self.mu(tf.cast(t, dtype=tf.float64), w, y_past)[k]) ** 2)
                / (
                    2
                    * tf.math.pow(
                        self.sigma(tf.cast(t, dtype=tf.float64), w, y_past)[k], 2
                    )
                )
            )
        else:
            return -(1 / 2) * (
                tf.math.log(
                    2
                    * np.pi
                    * self.sigma(tf.cast(t, dtype=tf.float64), w, y_past)[k] ** 2
                )
                + ((y - self.mu(tf.cast(t, dtype=tf.float64), w, y_past)[k]) ** 2)
                / tf.math.pow(self.sigma(tf.cast(t, dtype=tf.float64), w, y_past)[k], 2)
            )

    def forward_probs(self, y: tf.Tensor, w: tf.Tensor, y_past: tf.Tensor) -> tf.Tensor:
        """
        Compute the forward probabilities, ie: P(Y_1,....,Y_t,X_t=i)
        
        Arguments:
    
        - *y*: tf.Tensor(nb_time_step,) containing the main time series Y_t.
        
        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the values of the external signal in Hidden Markov Models with 
            external variable.
        
        - *y_past*: tf.Tensor(nb_time_step, past_dependency) containing the past values of the main signal Y_{t-m} in AutoRegressive 
            Hidden Markov Models.
         
        Returns:
        
        - *alpha*: a tf.Tensor containing the forward probabilities
        """
        T = len(y)
        alpha_t = []
        transition_probs = self.tp(
            tf.range(self.past_dependency, self.past_dependency + T, dtype=tf.float64),
            w,
        )
        emission_probs = [
            self.emission_prob(
                i,
                y,
                tf.range(self.past_dependency, self.past_dependency + T),
                w,
                y_past,
                apply_log=False,
            )
            for i in range(self.K)
        ]
        for i in range(self.K):
            alpha_t.append(self.pi()[i] * tf.expand_dims(emission_probs[i][0], axis=0))
        alpha = tf.transpose(alpha_t / tf.math.reduce_sum(alpha_t))
        for t in tf.range(1, T):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(alpha, tf.TensorShape([None, self.K]))]
            )
            alpha_t = []
            for i in range(self.K):
                values = [
                    alpha[t - 1][j] * emission_probs[i][t] * transition_probs[t, j, i]
                    for j in range(self.K)
                ]
                alpha_t.append(tf.math.reduce_sum(values))
            alpha = tf.concat(
                [alpha, tf.expand_dims(alpha_t / tf.math.reduce_sum(alpha_t), axis=0)],
                axis=0,
            )
        return alpha

    def backward_probs(
        self, y: tf.Tensor, w: tf.Tensor, y_past: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the backward probabilities, ie: P(Y_{t+1},....,Y_T|X_t=i)
        
        Arguments:
    
        - *y*: tf.Tensor(nb_time_step,) containing the main time series Y_t.
        
        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the values of the external signal in Hidden Markov Models with 
            external variables.
        
        - *y_past*: tf.Tensor(nb_time_step, past_dependency) containing the past values of the main signal Y_{t-m} in AutoRegressive 
            Hidden Markov Models.
         
        Returns:
        
        - *beta*: a tf.Tensor containing the backward probabilities
        """
        T = len(y)
        beta_t = []
        transition_probs = self.tp(
            tf.range(self.past_dependency, self.past_dependency + T, dtype=tf.float64),
            w,
        )
        emission_probs = [
            self.emission_prob(
                i,
                y,
                tf.range(self.past_dependency, self.past_dependency + T),
                w,
                y_past,
                apply_log=False,
            )
            for i in range(self.K)
        ]
        for i in range(self.K):
            beta_t.append(1)
        beta = tf.expand_dims(beta_t / tf.math.reduce_sum(beta_t), axis=0)
        for t in tf.range(0, T - 1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(beta, tf.TensorShape([None, self.K]))]
            )
            beta_t = []
            for i in range(self.K):
                values = [
                    beta[t][j]
                    * emission_probs[j][-t - 1]
                    * transition_probs[-t - 1, i, j]
                    for j in range(self.K)
                ]
                beta_t.append(tf.math.reduce_sum(values))
            beta = tf.concat(
                [beta, tf.expand_dims(beta_t / tf.math.reduce_sum(beta_t), axis=0)],
                axis=0,
            )
        return beta[::-1]

    def gamma_probs(self, alpha: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
        """
        Compute the gamma quantities, ie: P(X_t=i|Y_1,....,Y_T)
        
        Arguments:
    
        - *alpha*: a tf.Tensor containing the forward probabilities
        
        - *beta*: a tf.Tensor containing the backward probabilities
         
        Returns:
        
        - *gamma*: a tf.Tensor containing the gamma quantities
        """

        return (
            alpha
            * beta
            / tf.expand_dims(tf.math.reduce_sum(alpha * beta, axis=1), axis=1)
        )

    def xi_probs(
        self,
        alpha: tf.Tensor,
        beta: tf.Tensor,
        y: tf.Tensor,
        w: tf.Tensor,
        y_past: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the xi quantities, ie: P(X_t=i,X_{t+1}=j|Y_1,....,Y_T)
        
        Arguments:
        
        - *alpha*: tf.Tensor containing the forward probabilities.
        
        - *beta*: tf.Tensor containing the backward probabilities.
    
        - *y*: tf.Tensor(nb_time_step,) containing the main time series Y_t.
        
        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the value of the external signal in Hidden Markov Models with 
            external variables.
        
        - *y_past*: tf.Tensor(nb_time_step, past_dependency) containing the past value of the main signal Y_{t-m} in AutoRegressive 
            Hidden Markov Models.
         
        Returns:
        
        - *xi*: a tf.Tensor containing the xi quantities
        """
        T = len(y)
        xi = tf.reshape(alpha[:-1, :], (T - 1, self.K, 1)) * tf.concat(
            [tf.reshape(beta[1:, i], (T - 1, 1, 1)) for i in range(self.K)], axis=2
        )
        xi = (
            xi
            * self.tp(
                tf.range(
                    self.past_dependency, self.past_dependency + T - 1, dtype=tf.float64
                ),
                w[: T - 1],
            )
            * tf.concat(
                [
                    tf.reshape(
                        self.emission_prob(
                            i,
                            y[1:],
                            tf.range(
                                self.past_dependency + 1, self.past_dependency + T
                            ),
                            w[1:],
                            y_past[1:],
                            apply_log=False,
                        ),
                        (T - 1, 1, 1),
                    )
                    for i in range(self.K)
                ],
                axis=2,
            )
        )
        xi = xi / tf.reshape(
            tf.math.reduce_sum(tf.math.reduce_sum(xi, axis=2), axis=1), (T - 1, 1, 1)
        )
        return xi

    @tf.function
    def compute_Q(
        self,
        y: tf.Tensor,
        w: tf.Tensor,
        y_past: tf.Tensor,
        previous_param: List = None,
        new_param: List = None,
    ) -> tf.Tensor:
        """
        Compute the Q quantity of the EM algorithm
        
        Arguments:
        
        - *y*: tf.Tensor(nb_time_step,) containing the main time series Y_t.
        
        - *gamma*: tf.Tensor containing the gamma quantities.
    
        - *xi*: tf.Tensor containing the xi probabilities.
        
        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the value of the external signal in Hidden Markov Models with 
            external variables.
        
        - *y_past*: tf.Tensor(nb_time_step, past_dependency) containing the past value of the main signal Y_{t-m} in AutoRegressive 
            Hidden Markov Models.
         
        Returns:
        
        - *Q*: a tf.Tensor containing the Q quantity
        """
        if (previous_param is not None) or (new_param is not None):
            model_param = self.get_param(return_numpy=False)

        if previous_param is not None:
            self.assign_param(*previous_param)
        alpha = self.forward_probs(y, w, y_past)
        beta = self.backward_probs(y, w, y_past)
        gamma = tf.stop_gradient(self.gamma_probs(alpha, beta))
        xi = tf.stop_gradient(self.xi_probs(alpha, beta, y, w, y_past))
        if new_param is not None:
            self.assign_param(*new_param)
        T = len(y)
        if w is None:
            w = tf.zeros_like(y)
        if y_past is None:
            y_past = tf.zeros_like(y)

        Q_pi = tf.math.reduce_sum(gamma[0, :] * self.pi())
        Q_mu_sigma = tf.math.reduce_sum(
            gamma
            * tf.transpose(
                [
                    self.emission_prob(
                        i,
                        y,
                        tf.range(self.past_dependency, self.past_dependency + T),
                        w,
                        y_past,
                        apply_log=True,
                    )
                    for i in range(self.K)
                ]
            )
        )
        Q_tp = tf.math.reduce_sum(
            xi
            * tf.math.log(
                self.tp(
                    tf.range(
                        self.past_dependency,
                        self.past_dependency + T - 1,
                        dtype=tf.float64,
                    ),
                    w[: T - 1],
                )
            )
        )
        Q = -Q_pi - Q_mu_sigma - Q_tp

        if (previous_param is not None) or (new_param is not None):
            self.assign_param(*model_param)

        return Q

    @tf.function
    def compute_grad(
        self, y: tf.Tensor, w: tf.Tensor, y_past: tf.Tensor
    ) -> Tuple[tf.Tensor, List]:
        """
        Compute the tf.gradient link the the Q quantity of the EM algorithm
        
        Arguments:
        
        - *y*: tf.Tensor(nb_time_step,) containing the main time series Y_t.
        
        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the value of the external signal in Hidden Markov Models with 
            external variables.
        
        - *y_past*: tf.Tensor(nb_time_step, past_dependency) containing the past value of the main signal Y_{t-m} in AutoRegressive 
            Hidden Markov Models.
         
        Returns:
        
        - *EM_loss*: a tf.Tensor containing the Q quantity
        
        - *grad*: a  containing the gradient of the model parameters
        """

        with tf.GradientTape() as tape:
            EM_loss = self.compute_Q(y, w, y_past)
            grad = tape.gradient(EM_loss, self.trainable_variables)

        return grad

    def eval_model(
        self, y: tf.Tensor, w: tf.Tensor, y_past: tf.Tensor, eval_size: int
    ) -> tf.Tensor:
        """
        Compute the MSE of the model over a eval set.
        
        Arguments:

        - *y*: tf.Tensor(nb_time_step,) containing the main time series Y_t.
        
        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the values of the external signal in Hidden Markov Models with 
            external variables.
        
        - *y_past*: tf.Tensor(nb_time_step, past_dependency) containing the past values of the main signal Y_{t-m} in AutoRegressive 
            Hidden Markov Models.
            
        - *eval_size*: int defining the size of the eval set.
         
        Returns:
        
        - *mse*: the mse loss
        """
        w_test = np.repeat(w[-eval_size].numpy()[-1], eval_size - 1) 
        w_test = np.concatenate([w[-eval_size].numpy(), w_test], axis=0)
        alpha = self.forward_probs(y[:-eval_size], w[:-eval_size], y_past[:-eval_size])
        beta = self.backward_probs(y[:-eval_size], w[:-eval_size], y_past[:-eval_size])
        gamma = self.gamma_probs(alpha, beta)
        x_init = gamma[-1].numpy().argmax()
        mse_result = []
        for i in range(10):
            x_pred, y_pred, y_pred_past = self.simulate_xy(
                horizon=eval_size,
                start_t=len(y[:-eval_size]),
                x_init=x_init,
                w=w_test,
                y_init=y[-self.past_dependency - eval_size : -eval_size],
            )
            mse = tf.math.reduce_mean(
                (tf.math.squared_difference(y[-eval_size:], y_pred))
            )
            mse_result.append(mse)

        return np.mean(mse_result)

    def viterbi(self, y: tf.Tensor, w: tf.Tensor, y_past: tf.Tensor) -> tf.Tensor:
        """
        Compute the most probable hidden states sequence using the viterbi algorithm
        
        Arguments:
        
        - *y*: tf.Tensor(nb_time_step,) containing the main time series Y_t.

        - *w*: tf.Tensor(nb_time_step, past_dependency) containing the value of the external signal in Hidden Markov Models with 
            external variables.
        
        - *y_past*: tf.Tensor(nb_time_step, past_dependency) containing the past value of the main signal Y_{t-m} in AutoRegressive 
            Hidden Markov Models.
            
        Returns:
        
        - *x*: a tf.Tensor(nb_time_step,) containing the hidden states sequence
        """
        T = len(y)
        transition_probs = tf.math.log(self.tp(tf.range(T, dtype=tf.float64), w))
        emission_probs = [
            self.emission_prob(i, y, tf.range(T), w, y_past, apply_log=True)
            for i in range(self.K)
        ]

        eta = []
        for t in range(T):
            eta_t = []
            if t == 0:
                for i in range(self.K):
                    eta_t.append(emission_probs[i][0] + tf.math.log(self.pi()[i]))
                eta.append(tf.expand_dims(eta_t, axis=1))
            else:
                for i in range(self.K):
                    eta_t.append(
                        tf.math.reduce_max(
                            [
                                emission_probs[i][t]
                                + transition_probs[t, i, k]
                                + eta[t - 1][k]
                                for k in range(self.K)
                            ]
                        )
                    )
                eta.append(tf.expand_dims(eta_t, axis=1))
        eta = tf.stack(eta)
        x = tf.math.argmax(eta, axis=1)

        return tf.squeeze(x)

    def simulate_y(
        self, x: tf.Tensor, w: tf.Tensor, y_init: tf.Tensor, start_t: int
    ) -> tf.Tensor:
        """
        simulate a Y_t process according the a provided hidden states sequence.
        
        Arguments:
        
        - *x*: a tf.Tensor(nb_time_step,) containing the hidden states sequence
        
        - *w*: tf.Tensor(nb_time_step,past_dependency) containing the value of the external signal in the Hidden Markov Model with 
            external variable.
            
        - *y_init*: tf.Tensor(past_dependency,) containing the past value of the main signal Y_{start_t-past_dependency} in 
            AutoRegressive Hidden Markov Models in order to compute the first Y_{start_t}
            
        - *start_t*: time step where the Hidden sequence will start.
         
        Returns:
        
        - *y*: a tf.Tensor containing the observed sequence Y
        """
        T = len(x)
        if w is None:
            w = np.zeros(T)
        if y_init is None:
            y_init = np.zeros(1)

        y_past = [y_init]
        y = np.zeros(T)
        for t in range(T):
            y[t] = self.emission(
                x[t],
                tf.cast([start_t + t], tf.float64),
                tf.expand_dims(w[t], axis=0),
                tf.expand_dims(y_past[t], axis=0),
            )
            y_past.append(
                tf.concat([y_past[-1][1:], tf.expand_dims(y[-1], axis=0)], axis=0)
            )

        return y

    def simulate_xy(
        self,
        horizon: int,
        start_t: int = 0,
        x_init: int = None,
        w: tf.Tensor = None,
        y_init: tf.Tensor = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        simulate a (X_t,Y_t) process of length T.
        
        Arguments:
        
        - *horizon*: int corresponding to the length of the simulated process
        
        - *start_t*: time step where the sequences will start.
        
        - *x_init*: int corresponding to the first value of the hidden state X_{start_t-1}
        
        - *w*: tf.Tensor(nb_time_step,) containing the value of the external signal in the Hidden Markov Model with 
            external variable.
            
        - *y_init*: tf.Tensor(past_dependency,) containing the past value of the main signal Y_{start_t-past_dependency} in 
            AutoRegressive Hidden Markov Models in order to compute the first Y_{start_t}
         
        Returns:
        
        - *x*: a tf.Tensor(horizon,) containing the hidden states sequence X
        
        - *y*: a tf.Tensor(horizon,) containing the observed sequence Y
        
        - *y_past*: a tf.Tensor(hozieon, past_dependency) containing a sliding window of the sequence Y in the case of a AR HMM
        """
        x = []
        y_past = []
        y = []
        if w is None:
            w = tf.zeros(horizon)
        else:
            w = tf.stack(
                [
                    w[i : i + self.past_dependency]
                    for i in range(w.shape[0] - self.past_dependency + 1)
                ],
                axis=0,
            )
        if y_init is None:
            y_init = tf.zeros(self.past_dependency)
        y_past.append(y_init)
        
        transition_probs = self.tp(
            tf.range(start_t, horizon + start_t, dtype=tf.float64), w
        )
        if not self.auto_regressive:
            emission = [
                self.emission(
                    i, tf.range(start_t, horizon + start_t, dtype=tf.float64), w, y_past
                )
                for i in range(self.K)
            ]
        for t in range(horizon):
            if t == 0:
                if x_init is None:
                    x.append(np.random.binomial(1, self.pi()[1]))
                else:
                    x.append(x_init)
            else:
                x.append(np.random.binomial(1, transition_probs[t - 1][x[t - 1], 1],))
            if self.auto_regressive:
                y.append(
                    self.emission(
                        x[t],
                        tf.cast([start_t + t], tf.float64),
                        tf.expand_dims(w[t], axis=0),
                        tf.expand_dims(y_past[t], axis=0),
                    )
                )
                y_past.append(tf.concat([y_past[-1][1:], y[-1]], axis=0))
            else:
                y.append(emission[x[t]][t])

        x = tf.squeeze(tf.cast(tf.stack(x), tf.int32))
        y = tf.squeeze(tf.cast(tf.stack(y), tf.float64))
        y_past = tf.stack(y_past)

        return x, y, y_past

    def fit(
        self,
        y: tf.Tensor,
        nb_em_epoch: int,
        nb_iteration_per_epoch: int,
        nb_execution: int,
        optimizer_name: str,
        learning_rate: int,
        w: tf.Tensor = None,
        eval_size: int = 52,
        init_param: bool = True,
        return_param_evolution: bool = False,
    ) -> None:
        """
        Run the Baum and Welch algorithm to fit the HMM model.

        Arguments:

        - *y*: tf.Tensor(nb_time_step,) containing the main time series Y_t.
        
        - *w*: tf.Tensor(nb_time_step,) containing the value of the external signal in the Hidden Markov Model with 
            external variable.
        
        - *eval_size*: int defining the size of the eval set.

        - *nb_em_epoch*: how many epoch will be run in the EM algorithm

        - *nb_execution*: How many time the EM algorithm will be run with different starting points.

        - *optimizer*: a tensorflow optimizer used to fit the HMM parameters
            
        - *init_param*: if set to false, the initialization of the set of parameter will not be done and the same set of parameter will 
            be use at each EM exectution.
            
        - *return_param_evolution*: if set to True, return the evolution of all parameters during the EM algorithm.
        """

        if w is not None:
            w = tf.stack(
                [
                    w[i : i + self.past_dependency]
                    for i in range(w.shape[0] - self.past_dependency + 1)
                ],
                axis=0,
            )
        else:
            w = tf.zeros_like(y)
        y_past = tf.stack(
            [
                y[i : i + self.past_dependency]
                for i in range(y.shape[0] - self.past_dependency)
            ],
            axis=0,
        )
        y = y[self.past_dependency :]

        final_mse = np.inf
        run_mse = np.inf
        best_exec = -1
        final_param = None
        all_param = {}
        original_param = self.get_param()
        for i in range(nb_execution):
            optimizer = self.build_optimizer(
                optimizer_name=optimizer_name, learning_rate=learning_rate
            )
            if init_param:
                self.init_param()
            else:
                self.assign_param(*original_param)
            exec_param = []
            exec_param.append(self.get_param())
            wrong_initialisation = 0
            for epoch in range(nb_em_epoch):
                _Q = self.compute_Q(y, w, y_past)
                Q_diff = []
                updated_param = []
                for j in tf.range(nb_iteration_per_epoch):
                    grad = self.compute_grad(y, w, y_past)
                    optimizer.apply_gradients(zip(grad, self.trainable_variables))
                    updated_param.append(self.get_param())
                    Q_diff.append(
                        self.compute_Q(y, w, y_past, exec_param[-1], updated_param[-1])
                        - _Q
                    )
                Q_diff = Q_diff[np.argmin(Q_diff)]
                updated_param = updated_param[np.argmin(Q_diff)]
                print("epoch : ", epoch)
                print("Q diff : ", Q_diff)
                if Q_diff > 0:
                    print(
                        f"Q_diff > 0 --> optimizer learning rate reduced to {optimizer.lr/2}"
                    )
                    self.assign_param(*exec_param[-1])
                    optimizer.lr = optimizer.lr / 2
                else:
                    self.assign_param(*updated_param)
                exec_param.append(self.get_param())

                if not (np.abs(Q_diff.numpy()) <= 0 or np.abs(Q_diff.numpy()) > 0):
                    wrong_initialisation = 1
                    print("wrong initialisation")
                    break

                if np.abs(Q_diff) < 10 ** (-10):
                    print("stop of the EM algorithm")
                    break

            if not wrong_initialisation:
                run_mse = self.eval_model(y, w, y_past, eval_size)
                print(f"run mse : {run_mse}")
                all_param[i] = exec_param

                if run_mse < final_mse:
                    final_param = self.get_param()
                    final_mse = run_mse
                    best_exec = i
                    print("checkpoint")

        if final_param is not None:
            self.assign_param(*final_param)
        else:
            self.assign_param(*original_param)
        if return_param_evolution:
            return all_param, run_mse, best_exec

    def build_optimizer(
        self, optimizer_name: str, learning_rate: int
    ) -> tf.keras.optimizers.Optimizer:
        """
        Instantiate a return a tf.keras.optimizer

        Arguments:

        - *optimizer_name*: name of the chosen optimizer.

        - *learning_rate*: learning rate of the optimizer.
        
        Returns:

        - *optimizer*: a tensorflow optimizer.
        """
        if optimizer_name.lower() == "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def evaluate(
        self,
        y: tf.Tensor,
        y_test: tf.Tensor,
        w: tf.Tensor,
        w_test: int,
        start_t: tf.Tensor,
        horizon: int,
        nb_simulation: int,
        norm=None,
    ):
        """
        Evaluate a trained model by simulating a large number of trajectories and computing accuracy metrics.

        Arguments:

        - *y*: tf.Tensor(nb_time_step,) containing the train part of the main time series Y_t.
        
        - *y_test*: tf.Tensor(nb_time_step,) containing the test part of the main time series Y_t.

        - *w*: tf.Tensor(nb_time_step,) containing the train part values of the external signal in the Hidden Markov Model with 
            external variable.
            
        - *w_test*: tf.Tensor(nb_time_step,) containing the test part values of the external signal in the Hidden Markov Model with 
            external variable.

        - *start_t*: time step where the simulated sequences will start.
        
        - *horizon*: int corresponding to the length of the simulated process.
        
        - *nb_simulation*: int corresponding to number of simulation that will be generated.
        
        - *norm*: provide a normalization factor the model predictions have to be rescale. 

        Returns:

        - *result*: a dict containing the MASE, MAE, MSE accuracy metrics, the main prediction and Q1,Q9 prediction
        
        - *all_simulation*: dict containing the nb_simulation simulation of Y_t and X_t.
        """

        if w is not None:
            w_test = np.concatenate(
                [w[-self.past_dependency :], w_test[: -1]], axis=0
            )
            w = tf.stack(
                [
                    w[i : i + self.past_dependency]
                    for i in range(w.shape[0] - self.past_dependency)
                ],
                axis=0,
            )
        else:
            w = tf.zeros_like(y)
        y_past = tf.stack(
            [
                y[i : i + self.past_dependency]
                for i in range(y.shape[0] - self.past_dependency)
            ],
            axis=0,
        )
        y_init = y[-self.past_dependency :]
        y = y[self.past_dependency :]

        alpha = self.forward_probs(y, w, y_past)
        beta = self.backward_probs(y, w, y_past)
        gamma = self.gamma_probs(alpha, beta)
        x_init = gamma[-1].numpy().argmax()

        all_x_pred = []
        all_y_pred = []
        for i in range(nb_simulation):
            x_pred, y_pred, y_pred_past = self.simulate_xy(
                horizon=horizon, start_t=start_t, x_init=x_init, w=w_test, y_init=y_init
            )
            all_x_pred.append(x_pred)
            all_y_pred.append(y_pred)

        if norm is not None:
            for i in range(len(all_y_pred)):
                all_y_pred[i] *= norm
        else:
            norm = 1

        result = {}
        result["y_pred_mean"] = np.mean(all_y_pred, axis=0)
        result["y_pred_q9"] = np.quantile(all_y_pred, 0.95, axis=0)
        result["y_pred_q1"] = np.quantile(all_y_pred, 0.05, axis=0)
        mase_denom = np.mean(np.abs(y[52:] - y[:-52]) * norm)
        result["mse_y_pred_mean"] = np.mean(np.square(y_test - result["y_pred_mean"]))
        result["mae_y_pred_mean"] = np.mean(np.abs(y_test - result["y_pred_mean"]))
        result["mase_y_pred_mean"] = (
            np.mean(np.abs(y_test - result["y_pred_mean"])) / mase_denom
        )

        all_simulation = {"x": all_x_pred, "y": all_y_pred}

        return result, all_simulation
