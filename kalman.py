import numpy as np
import math

def rssi2meters(rssi, tx):
    N = 2
    return math.pow(10, -(rssi - tx) / (10 * N))

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

    @staticmethod
    def one_d(processNoise, sensorNoise, estimatedError, initialValue):
        """
        Creates a 1D Kalman Filter instance from simple scalar parameters.
        
        Args:
            processNoise (float): System model uncertainty
            sensorNoise (float): Measurement noise
            estimatedError (float): Initial error estimate
            initialValue (float): Initial state estimate
        
        Returns:
            KalmanFilter: Configured for 1D tracking
        """
        F = np.array([[1]])  # State transition matrix (1x1)
        H = np.array([[1]])  # Measurement matrix (1x1)
        Q = np.array([[processNoise]])  # Process noise matrix (1x1)
        R = np.array([[sensorNoise]])  # Measurement noise matrix (1x1)
        P = np.array([[estimatedError]])  # Initial error covariance (1x1)
        x0 = np.array([[initialValue]])  # Initial state (1x1)

        return KalmanFilter(F=F, H=H, Q=Q, R=R, P=P, x0=x0)

class KalmanFilter2:
    def __init__(self, 
                 dt,           # time step (e.g., 0.1 seconds between measurements)
                 acc_noise,    # acceleration noise magnitude
                 meas_noise):  # measurement noise magnitude
        """
        A Kalman filter for tracking distance and velocity
        
        Args:
            dt: Time step between measurements (seconds)
            acc_noise: Process noise (acceleration variation)
            meas_noise: Measurement noise (sensor/RSSI noise)
        """
        # System dynamics
        self.dt = dt
        self.A = np.array([[1, dt],    # state transition matrix
                          [0, 1]])
        self.H = np.array([[1, 0]])    # measurement matrix (we only measure position)
        
        # Noise covariances
        self.Q = np.array([            # process noise covariance
            [(dt**4)/4, (dt**3)/2],
            [(dt**3)/2, dt**2]
        ]) * acc_noise**2
        
        self.R = meas_noise**2         # measurement noise covariance
        
        # Initial state
        self.P = np.eye(2)             # initial state covariance
        self.x = np.array([[0],        # initial state estimate
                          [0]])         # [position, velocity]

    def predict(self):
        """
        Predict next state (prior) using the Kalman filter state propagation equations.
        """
        # Update state estimate
        self.x = np.dot(self.A, self.x)
        
        # Update error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.x[0].item()  # return predicted position

    def update(self, measurement):
        """
        Update the Kalman filter state based on the measurement.
        
        Args:
            measurement: The distance measurement from RSSI
        """
        # Calculate Kalman gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), 1/S)

        # Update state estimate
        y = measurement - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        # Update error covariance
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        
        return self.x[0].item()  # return updated position

class KalmanFilter3:

    cov = float('nan')
    x = float('nan')

    def __init__(self, R, Q):
        """
        Constructor
        :param R: Process Noise
        :param Q: Measurement Noise
        """
        self.A = 1
        self.B = 0
        self.C = 1

        self.R = R
        self.Q = Q

    def filter(self, measurement):
        """
        Filters a measurement
        :param measurement: The measurement value to be filtered
        :return: The filtered value
        """
        u = 0
        if math.isnan(self.x):
            self.x = (1 / self.C) * measurement
            self.cov = (1 / self.C) * self.Q * (1 / self.C)
        else:
            predX = (self.A * self.x) + (self.B * u)
            predCov = ((self.A * self.cov) * self.A) + self.R

            # Kalman Gain
            K = predCov * self.C * (1 / ((self.C * predCov * self.C) + self.Q));

            # Correction
            self.x = predX + K * (measurement - (self.C * predX));
            self.cov = predCov - (K * self.C * predCov);

        return self.x

    def last_measurement(self):
        """
        Returns the last measurement fed into the filter
        :return: The last measurement fed into the filter
        """
        return self.x

    def set_measurement_noise(self, noise):
        """
        Sets measurement noise
        :param noise: The new measurement noise
        """
        self.Q = noise

    def set_process_noise(self, noise):
        """
        Sets process noise
        :param noise: The new process noise
        """
        self.R = noise