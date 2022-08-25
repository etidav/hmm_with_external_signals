import numpy as np
import tensorflow as tf
from model.abstract_HMM import Hidden_Markov_Model


class ARSHMM(Hidden_Markov_Model):
    """
    Class defining a autoregressive seasonal HMM.
    """

    def __init__(
        self, nb_hidden_states: int = 2, past_dependency: int = 1, season: int = 1
    ) -> None:
        """
        Instantiate a ARHMM model with gaussian emission laws and discrete hidden states.
        
        Arguments:
    
        - *nb_hidden_states*: number of hidden states of the HMM.
        
        - *past_dependency*: In case of a ARHMM, define the past dependency length.
        
        - *season*: In case of a SHMM, define the seasonality length.
        """
        super().__init__(nb_hidden_states, past_dependency, season)
        self.auto_regressive = True

    def define_param(self) -> None:
        """
        Define parameters of the HMM model 
        """
        self.pi_var = tf.Variable([0.5 for i in range(self.K - 1)], dtype=tf.float64)
        self.delta_var = tf.Variable(
            [[0.5 for i in range(3 + self.past_dependency)] for j in range(self.K)],
            dtype=tf.float64,
        )
        self.sigma_var = tf.Variable([0.5 for i in range(self.K)], dtype=tf.float64)
        self.omega_var = tf.Variable(
            [
                [[0.5 for i in range(3)] for j in range(self.K - 1)]
                for k in range(self.K)
            ],
            dtype=tf.float64,
        )

    def pi(self) -> tf.Tensor:
        """
        Compute the emission probabilities of the first hidden variable values X_0
         
        Returns:
        
        - *pi*: a tf.Tensor(nb_hidden_stat,) containing the nb_hidden_states emission probabilities
        """
        return tf.concat(
            [
                tf.math.exp(self.pi_var)
                / (1 + tf.math.reduce_sum(tf.math.exp(self.pi_var))),
                tf.expand_dims(
                    1
                    - tf.math.reduce_sum(
                        tf.math.exp(self.pi_var)
                        / (1 + tf.math.reduce_sum(tf.math.exp(self.pi_var)))
                    ),
                    axis=0,
                ),
            ],
            axis=0,
        )

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
        t = tf.expand_dims(t, axis=1)
        return tf.transpose(
            tf.math.reduce_sum(
                tf.expand_dims(self.delta_var, axis=0)
                * tf.expand_dims(
                    tf.concat(
                        [
                            tf.expand_dims(
                                tf.ones(y_past.shape[0], dtype=tf.float64), axis=1
                            ),
                            y_past,
                            tf.math.cos((2 * np.pi * t) / self.season),
                            tf.math.sin((2 * np.pi * t) / self.season),
                        ],
                        axis=1,
                    ),
                    axis=1,
                ),
                axis=2,
            )
        )

    def sigma(
        self, t: tf.Tensor = None, w: tf.Tensor = None, y_past: tf.Tensor = None
    ) -> tf.Tensor:
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
        return tf.math.exp(self.sigma_var)

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
        t = tf.expand_dims(t, axis=1)
        y = tf.expand_dims(
            tf.concat(
                [
                    tf.expand_dims(tf.ones(y_past.shape[0], dtype=tf.float64), axis=1),
                    tf.math.cos((2 * np.pi * t) / self.season),
                    tf.math.sin((2 * np.pi * t) / self.season),
                ],
                axis=1,
            ),
            axis=1,
        )
        y = tf.concat([y for i in range(self.K - 1)], axis=1)
        y = tf.expand_dims(y, axis=1)
        y = tf.concat([y for i in range(self.K)], axis=1)
        omega_var = tf.expand_dims(self.omega_var, axis=0)
        P = tf.math.exp(
            tf.reshape(
                tf.math.reduce_sum(y * omega_var, axis=3),
                (t.shape[0], self.K, self.K - 1),
            )
        )
        normalized_P = P / tf.expand_dims(1 + tf.math.reduce_sum(P, axis=2), axis=2)

        return tf.concat(
            [
                normalized_P,
                tf.expand_dims(1 - tf.math.reduce_sum(normalized_P, axis=2,), axis=2,),
            ],
            axis=2,
        )


class ARSHMM_window(Hidden_Markov_Model):
    """
    Class defining a autoregressive seasonal HMM with external signals.
    """

    def __init__(
        self,
        nb_hidden_states: int = 2,
        past_dependency: int = 1,
        past_dependency0: int = 1,
        season: int = 1,
    ) -> None:
        """
        Instantiate a ARHMM model with gaussian emission laws and discrete hidden states.
        
        Arguments:
    
        - *nb_hidden_states*: number of hidden states of the HMM.
        
        - *past_dependency*: In case of a ARHMM, define the past dependency length.
        
        - *past_dependency0*: In case of a ARHMM with window, define le length of window of past dependency.
        
        - *season*: In case of a SHMM, define the seasonality length.
        """
        self.past_dependency0 = past_dependency0
        super().__init__(nb_hidden_states, past_dependency, season)
        self.auto_regressive = True

    def define_param(self) -> None:
        """
        Define parameters of the HMM model 
        """
        self.pi_var = tf.Variable([0.5 for i in range(self.K - 1)], dtype=tf.float64)
        self.delta_var = tf.Variable(
            [[0.5 for i in range(3 + self.past_dependency0)] for j in range(self.K)],
            dtype=tf.float64,
        )
        self.sigma_var = tf.Variable([0.5 for i in range(self.K)], dtype=tf.float64)
        self.omega_var = tf.Variable(
            [
                [[0.5 for i in range(3)] for j in range(self.K - 1)]
                for k in range(self.K)
            ],
            dtype=tf.float64,
        )

    def pi(self) -> tf.Tensor:
        """
        Compute the emission probabilities of the first hidden variable values X_0
         
        Returns:
        
        - *pi*: a tf.Tensor(nb_hidden_stat,) containing the nb_hidden_states emission probabilities
        """
        return tf.concat(
            [
                tf.math.exp(self.pi_var)
                / (1 + tf.math.reduce_sum(tf.math.exp(self.pi_var))),
                tf.expand_dims(
                    1
                    - tf.math.reduce_sum(
                        tf.math.exp(self.pi_var)
                        / (1 + tf.math.reduce_sum(tf.math.exp(self.pi_var)))
                    ),
                    axis=0,
                ),
            ],
            axis=0,
        )

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
        t = tf.expand_dims(t, axis=1)
        return tf.transpose(
            tf.math.reduce_sum(
                tf.expand_dims(self.delta_var, axis=0)
                * tf.expand_dims(
                    tf.concat(
                        [
                            tf.expand_dims(
                                tf.ones(w.shape[0], dtype=tf.float64), axis=1
                            ),
                            tf.gather(
                                y_past,
                                [*[i for i in range(self.past_dependency0)],],
                                axis=1,
                            ),
                            tf.math.cos((2 * np.pi * t) / self.season),
                            tf.math.sin((2 * np.pi * t) / self.season),
                        ],
                        axis=1,
                    ),
                    axis=1,
                ),
                axis=2,
            )
        )

    def sigma(
        self, t: tf.Tensor = None, w: tf.Tensor = None, y_past: tf.Tensor = None
    ) -> tf.Tensor:
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
        return tf.math.exp(self.sigma_var)

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
        t = tf.expand_dims(t, axis=1)
        y = tf.expand_dims(
            tf.concat(
                [
                    tf.expand_dims(tf.ones(t.shape[0], dtype=tf.float64), axis=1),
                    tf.math.cos((2 * np.pi * t) / self.season),
                    tf.math.sin((2 * np.pi * t) / self.season),
                ],
                axis=1,
            ),
            axis=1,
        )
        y = tf.concat([y for i in range(self.K - 1)], axis=1)
        y = tf.expand_dims(y, axis=1)
        y = tf.concat([y for i in range(self.K)], axis=1)
        omega_var = tf.expand_dims(self.omega_var, axis=0)
        P = tf.math.exp(
            tf.reshape(
                tf.math.reduce_sum(y * omega_var, axis=3),
                (t.shape[0], self.K, self.K - 1),
            )
        )
        normalized_P = P / tf.expand_dims(1 + tf.math.reduce_sum(P, axis=2), axis=2)

        return tf.concat(
            [
                normalized_P,
                tf.expand_dims(1 - tf.math.reduce_sum(normalized_P, axis=2,), axis=2,),
            ],
            axis=2,
        )
