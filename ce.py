import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from matplotlib.widgets import Slider, Button
import logging

# ======================================
#          CONFIGURATION SETUP
# ======================================
# Configure logging to monitor the application's behavior and debug if necessary.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define initial parameters for data generation and anomaly detection.
INITIAL_WINDOW_SIZE = 200
INITIAL_K_NEIGHBORS = 25
INITIAL_THRESHOLD_LOF = 2.0
INITIAL_SEASONALITY_AMP = 5
INITIAL_NOISE_LEVEL = 1
INITIAL_ANOMALY_PROB = 0.02

# Define initial weights for combining anomaly detection scores (optional).
INITIAL_WEIGHT_ILOF = 1.0
INITIAL_WEIGHT_EWMA = 1.0
INITIAL_WEIGHT_SKLOF = 1.0
INITIAL_WEIGHT_ISOFOREST = 1.0
INITIAL_WEIGHT_OCSVM = 1.0
FINAL_THRESHOLD = 3  # Voting threshold: number of detectors needed to flag anomaly

# ======================================
#        DATA STREAM SIMULATION
# ======================================
def data_stream():
    """
    Simulates a real-time data stream with dynamic seasonality, trend shifts,
    varying noise levels, and injected anomalies.
    """
    time_step = 0
    season_length = 50
    trend = 0.0  # Initialize trend
    global seasonality_amp, noise_level, anomaly_prob
    while True:
        # Introduce concept drift by changing seasonality parameters periodically.
        if time_step % 1000 == 0 and time_step > 0:
            season_length = random.randint(30, 70)  # Dynamically changing season length
            seasonality_amp = random.uniform(3, 7)
            logging.info(f'Concept drift occurred at time_step {time_step}: New seasonality_amp = {seasonality_amp:.2f}, New season_length = {season_length}')

        # Introduce trend shifts periodically.
        if time_step % 500 == 0 and time_step > 0:
            trend_shift = random.uniform(-5, 5)
            trend += trend_shift
            logging.info(f'Trend shift at time_step {time_step}: New trend = {trend:.2f}')

        # Calculate seasonality with the current amplitude and season length.
        seasonality = seasonality_amp * np.sin(2 * np.pi * (time_step % season_length) / season_length)

        # Introduce varying noise levels.
        dynamic_noise = np.random.uniform(0.5, 3) * noise_level
        noise = np.random.normal(0, dynamic_noise)

        # Combine trend, seasonality, and noise to form the data point.
        value = trend + seasonality + noise

        # Inject anomalies based on the current anomaly probability.
        if random.random() < anomaly_prob:
            if random.random() < 0.5:
                anomaly = random.choice([15, -15])
                value += anomaly
                logging.debug(f'Injected global anomaly at time_step {time_step}: {anomaly}')
            else:
                anomaly = np.random.normal(0, 3)
                value += anomaly
                logging.debug(f'Injected local anomaly at time_step {time_step}: {anomaly:.2f}')

        yield [value]
        time_step += 1

# ======================================
#        ANOMALY DETECTOR CLASSES
# ======================================

class ILOF:
    """
    Incremental Local Outlier Factor (ILOF) detector that updates in real-time.
    """
    def __init__(self, k=25, window_size=200, threshold=2.0):
        self.k = k
        self.window_size = window_size
        self.threshold = threshold  # Threshold for binary decision
        self.window = deque(maxlen=window_size)
        self.nbrs = None
        self.lrd = {}
        self.lof = {}
        logging.info(f'ILOF initialized with k={self.k}, window_size={self.window_size}, threshold={self.threshold}')

    def fit_new_point(self, point):
        self.window.append(point)
        data = np.array(self.window)
        logging.debug(f'ILOF - Added new point: {point[0]:.2f}')
        if len(data) > self.k:
            self.nbrs = NearestNeighbors(n_neighbors=self.k)
            self.nbrs.fit(data)
            distances, indices = self.nbrs.kneighbors(data)
            logging.debug('ILOF - Nearest neighbors updated.')
            reach_dist = np.maximum(distances, distances[:, [0]])
            lrd = 1 / (np.sum(reach_dist, axis=1) / self.k)
            lof = []
            for i in range(len(data)):
                lrd_ratios = lrd[indices[i]] / lrd[i]
                lof_score = np.sum(lrd_ratios) / self.k
                lof.append(lof_score)
            self.lrd = dict(zip(range(len(data)), lrd))
            self.lof = dict(zip(range(len(data)), lof))
            current_lof = lof[-1]
            logging.debug(f'ILOF - Computed LOF score for new point: {current_lof:.2f}')
            # Binary decision based on threshold
            is_anomaly = 1 if current_lof > self.threshold else 0
            return is_anomaly
        else:
            logging.debug('ILOF - Not enough data to compute LOF. Returning normal.')
            return 0  # Normal if insufficient data

    def update_parameters(self, k=None, window_size=None, threshold=None):
        reset = False
        if k is not None and k != self.k:
            logging.info(f'ILOF - Updating k from {self.k} to {k}')
            self.k = k
            reset = True
        if window_size is not None and window_size != self.window_size:
            logging.info(f'ILOF - Updating window_size from {self.window_size} to {window_size}')
            self.window_size = window_size
            self.window = deque(maxlen=window_size)
            reset = True
        if threshold is not None and threshold != self.threshold:
            logging.info(f'ILOF - Updating threshold from {self.threshold} to {threshold}')
            self.threshold = threshold
        if reset:
            self.nbrs = None
            self.lrd = {}
            self.lof = {}
            logging.info('ILOF - Detector reset due to parameter changes.')

class EWMADetector:
    """
    Exponentially Weighted Moving Average (EWMA) detector for anomaly detection.
    """
    def __init__(self, window_size=200, alpha=0.3, threshold=3.0):
        self.window_size = window_size
        self.alpha = alpha
        self.threshold = threshold  # Threshold for binary decision
        self.ewma = None
        self.ewma_var = None
        self.history = deque(maxlen=window_size)
        logging.info(f'EWMADetector initialized with window_size={self.window_size}, alpha={self.alpha}, threshold={self.threshold}')

    def fit_new_point(self, point):
        if self.ewma is None:
            self.ewma = point[0]
            self.ewma_var = 0.0
        else:
            self.ewma = self.alpha * point[0] + (1 - self.alpha) * self.ewma
            self.ewma_var = self.alpha * (point[0] - self.ewma) ** 2 + (1 - self.alpha) * self.ewma_var
        self.history.append(point[0])
        std = np.sqrt(self.ewma_var) if self.ewma_var > 0 else 1.0
        deviation = abs(point[0] - self.ewma) / std
        logging.debug(f'EWMADetector - Point: {point[0]:.2f}, EWMA: {self.ewma:.2f}, Deviation: {deviation:.2f}')
        # Binary decision based on threshold
        is_anomaly = 1 if deviation > self.threshold else 0
        return is_anomaly

    def update_parameters(self, alpha=None, threshold=None, window_size=None):
        reset = False
        if alpha is not None and alpha != self.alpha:
            logging.info(f'EWMADetector - Updating alpha from {self.alpha} to {alpha}')
            self.alpha = alpha
            reset = True
        if threshold is not None and threshold != self.threshold:
            logging.info(f'EWMADetector - Updating threshold from {self.threshold} to {threshold}')
            self.threshold = threshold
        if window_size is not None and window_size != self.window_size:
            logging.info(f'EWMADetector - Updating window_size from {self.window_size} to {window_size}')
            self.window_size = window_size
            self.history = deque(maxlen=window_size)
            reset = True
        if reset:
            self.ewma = None
            self.ewma_var = None
            logging.info('EWMADetector - Detector reset due to parameter changes.')

class SKLearnLOF:
    """
    Scikit-Learn's Local Outlier Factor (LOF) detector.
    """
    def __init__(self, window_size=200, n_neighbors=25, contamination=0.02, threshold=1.5):
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.threshold = threshold  # Threshold for binary decision
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination, novelty=False)
        self.history = deque(maxlen=self.window_size)
        logging.info(f'SKLearnLOF initialized with window_size={self.window_size}, n_neighbors={self.n_neighbors}, contamination={self.contamination}, threshold={self.threshold}')

    def fit_new_point(self, point):
        self.history.append(point[0])
        if len(self.history) >= self.n_neighbors + 1:
            data = np.array(self.history).reshape(-1, 1)
            self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination, novelty=False)
            y_pred = self.model.fit_predict(data)
            anomaly_score = -self.model.negative_outlier_factor_[-1]
            logging.debug(f'SKLearnLOF - Point: {point[0]:.2f}, LOF Score: {anomaly_score:.2f}')
            # Binary decision based on threshold
            is_anomaly = 1 if anomaly_score > self.threshold else 0
            return is_anomaly
        else:
            logging.debug('SKLearnLOF - Not enough data to compute LOF. Returning normal.')
            return 0  # Normal if insufficient data

    def update_parameters(self, n_neighbors=None, contamination=None, window_size=None, threshold=None):
        reset = False
        if n_neighbors is not None and n_neighbors != self.n_neighbors:
            logging.info(f'SKLearnLOF - Updating n_neighbors from {self.n_neighbors} to {n_neighbors}')
            self.n_neighbors = n_neighbors
            reset = True
        if contamination is not None and contamination != self.contamination:
            logging.info(f'SKLearnLOF - Updating contamination from {self.contamination} to {contamination}')
            self.contamination = contamination
            reset = True
        if window_size is not None and window_size != self.window_size:
            logging.info(f'SKLearnLOF - Updating window_size from {self.window_size} to {window_size}')
            self.window_size = window_size
            self.history = deque(maxlen=window_size)
            reset = True
        if threshold is not None and threshold != self.threshold:
            logging.info(f'SKLearnLOF - Updating threshold from {self.threshold} to {threshold}')
            self.threshold = threshold
        if reset:
            self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination, novelty=False)
            logging.info('SKLearnLOF - Detector reset due to parameter changes.')

class IsolationForestDetector:
    """
    Isolation Forest detector for anomaly detection.
    """
    def __init__(self, window_size=200, contamination=0.02, n_estimators=100, threshold=0.0):
        self.window_size = window_size
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.threshold = threshold  # Threshold for binary decision
        self.model = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination, random_state=42)
        self.history = deque(maxlen=self.window_size)
        logging.info(f'IsolationForestDetector initialized with window_size={self.window_size}, contamination={self.contamination}, n_estimators={self.n_estimators}, threshold={self.threshold}')

    def fit_new_point(self, point):
        self.history.append(point[0])
        if len(self.history) >= self.window_size:
            data = np.array(self.history).reshape(-1, 1)
            self.model.fit(data)
            score = self.model.decision_function([[point[0]]])[0]
            anomaly = self.model.predict([[point[0]]])[0]
            logging.debug(f'IsolationForestDetector - Point: {point[0]:.2f}, Score: {score:.2f}, Anomaly: {anomaly}')
            # Binary decision based on threshold
            is_anomaly = 1 if score < self.threshold else 0  # Lower scores indicate anomalies
            return is_anomaly
        else:
            logging.debug('IsolationForestDetector - Not enough data to compute Isolation Forest. Returning normal.')
            return 0  # Normal if insufficient data

    def update_parameters(self, contamination=None, window_size=None, n_estimators=None, threshold=None):
        reset = False
        if contamination is not None and contamination != self.contamination:
            logging.info(f'IsolationForestDetector - Updating contamination from {self.contamination} to {contamination}')
            self.contamination = contamination
            reset = True
        if window_size is not None and window_size != self.window_size:
            logging.info(f'IsolationForestDetector - Updating window_size from {self.window_size} to {window_size}')
            self.window_size = window_size
            self.history = deque(maxlen=window_size)
            reset = True
        if n_estimators is not None and n_estimators != self.n_estimators:
            logging.info(f'IsolationForestDetector - Updating n_estimators from {self.n_estimators} to {n_estimators}')
            self.n_estimators = n_estimators
            reset = True
        if threshold is not None and threshold != self.threshold:
            logging.info(f'IsolationForestDetector - Updating threshold from {self.threshold} to {threshold}')
            self.threshold = threshold
        if reset:
            self.model = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination, random_state=42)
            logging.info('IsolationForestDetector - Detector reset due to parameter changes.')

class OneClassSVMDetector:
    """
    One-Class SVM detector for anomaly detection.
    """
    def __init__(self, window_size=200, kernel='rbf', gamma='scale', nu=0.02, threshold=0.0):
        self.window_size = window_size
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.threshold = threshold  # Threshold for binary decision
        self.model = OneClassSVM(kernel=self.kernel, gamma=self.gamma, nu=self.nu)
        self.history = deque(maxlen=self.window_size)
        logging.info(f'OneClassSVMDetector initialized with window_size={self.window_size}, kernel={self.kernel}, gamma={self.gamma}, nu={self.nu}, threshold={self.threshold}')

    def fit_new_point(self, point):
        self.history.append(point[0])
        if len(self.history) >= self.window_size:
            data = np.array(self.history).reshape(-1, 1)
            self.model.fit(data)
            score = self.model.decision_function([[point[0]]])[0]
            anomaly = self.model.predict([[point[0]]])[0]
            logging.debug(f'OneClassSVMDetector - Point: {point[0]:.2f}, Score: {score:.2f}, Anomaly: {anomaly}')
            # Binary decision based on threshold
            is_anomaly = 1 if score < self.threshold else 0  # Lower scores indicate anomalies
            return is_anomaly
        else:
            logging.debug('OneClassSVMDetector - Not enough data to compute One-Class SVM. Returning normal.')
            return 0  # Normal if insufficient data

    def update_parameters(self, kernel=None, gamma=None, nu=None, window_size=None, threshold=None):
        reset = False
        if kernel is not None and kernel != self.kernel:
            logging.info(f'OneClassSVMDetector - Updating kernel from {self.kernel} to {kernel}')
            self.kernel = kernel
            reset = True
        if gamma is not None and gamma != self.gamma:
            logging.info(f'OneClassSVMDetector - Updating gamma from {self.gamma} to {gamma}')
            self.gamma = gamma
            reset = True
        if nu is not None and nu != self.nu:
            logging.info(f'OneClassSVMDetector - Updating nu from {self.nu} to {nu}')
            self.nu = nu
            reset = True
        if window_size is not None and window_size != self.window_size:
            logging.info(f'OneClassSVMDetector - Updating window_size from {self.window_size} to {window_size}')
            self.window_size = window_size
            self.history = deque(maxlen=window_size)
            reset = True
        if threshold is not None and threshold != self.threshold:
            logging.info(f'OneClassSVMDetector - Updating threshold from {self.threshold} to {threshold}')
            self.threshold = threshold
        if reset:
            self.model = OneClassSVM(kernel=self.kernel, gamma=self.gamma, nu=self.nu)
            logging.info('OneClassSVMDetector - Detector reset due to parameter changes.')

# ======================================
#          VISUALIZATION SETUP
# ======================================
# Hide the default matplotlib toolbar for a cleaner interface.
plt.rcParams['toolbar'] = 'None'

# Create the main figure with a specified size and layout.
fig = plt.figure(figsize=(18, 12), constrained_layout=True)
manager = plt.get_current_fig_manager()
try:
    manager.full_screen_toggle()  # Attempt to make the plot full screen.
except AttributeError:
    pass  # If full_screen_toggle is not available, proceed without it.

# Define a GridSpec layout to organize multiple plots and controls.
gs = fig.add_gridspec(6, 3)

# --------------------------------------
#       MAIN PLOT FOR COMBINED ANOMALY DETECTION
# --------------------------------------
ax_main = fig.add_subplot(gs[0:2, :])
ax_main.set_title('Real-Time Data Stream Anomaly Detection', fontsize=18)
ax_main.set_xlabel('Time Steps', fontsize=14)
ax_main.set_ylabel('Value', fontsize=14)
ax_main.grid(True)

# Initialize data lists for the main plot.
xdata, ydata = [], []
anomalies_main_x, anomalies_main_y = [], []
line_normal, = ax_main.plot([], [], color='blue', label='Data Stream')
scatter_anomalies_main = ax_main.scatter([], [], c='darkred', marker='o', s=50, label='Combined Anomalies')
ax_main.legend(loc='upper right')

# --------------------------------------
#        INDIVIDUAL DETECTORS' PLOTS
# --------------------------------------
# Define individual subplots for each anomaly detector with distinct colors and markers.
ax_ilof = fig.add_subplot(gs[2, 0])
ax_ilof.set_title('ILOF Anomalies', fontsize=14)
ax_ilof.set_xlabel('Time Steps', fontsize=12)
ax_ilof.set_ylabel('Value', fontsize=12)
ax_ilof.grid(True)
line_ilof, = ax_ilof.plot([], [], color='green', label='ILOF')
scatter_ilof = ax_ilof.scatter([], [], c='orange', marker='x', label='Anomalies')
ax_ilof.legend(loc='upper right')

ax_ewma = fig.add_subplot(gs[2, 1])
ax_ewma.set_title('EWMA Anomalies', fontsize=14)
ax_ewma.set_xlabel('Time Steps', fontsize=12)
ax_ewma.set_ylabel('Value', fontsize=12)
ax_ewma.grid(True)
line_ewma, = ax_ewma.plot([], [], color='purple', label='EWMA')
scatter_ewma = ax_ewma.scatter([], [], c='lime', marker='D', label='Anomalies')
ax_ewma.legend(loc='upper right')

ax_sklof = fig.add_subplot(gs[2, 2])
ax_sklof.set_title('SKLearn LOF Anomalies', fontsize=14)
ax_sklof.set_xlabel('Time Steps', fontsize=12)
ax_sklof.set_ylabel('Value', fontsize=12)
ax_sklof.grid(True)
line_sklof, = ax_sklof.plot([], [], color='orange', label='SKLearn LOF')
scatter_sklof = ax_sklof.scatter([], [], c='cyan', marker='s', label='Anomalies')
ax_sklof.legend(loc='upper right')

ax_iso = fig.add_subplot(gs[3, 0])
ax_iso.set_title('Isolation Forest Anomalies', fontsize=14)
ax_iso.set_xlabel('Time Steps', fontsize=12)
ax_iso.set_ylabel('Value', fontsize=12)
ax_iso.grid(True)
line_iso, = ax_iso.plot([], [], color='cyan', label='Isolation Forest')
scatter_iso = ax_iso.scatter([], [], c='magenta', marker='^', label='Anomalies')
ax_iso.legend(loc='upper right')

ax_ocsvm = fig.add_subplot(gs[3, 1])
ax_ocsvm.set_title('One-Class SVM Anomalies', fontsize=14)
ax_ocsvm.set_xlabel('Time Steps', fontsize=12)
ax_ocsvm.set_ylabel('Value', fontsize=12)
ax_ocsvm.grid(True)
line_ocsvm, = ax_ocsvm.plot([], [], color='magenta', label='One-Class SVM')
scatter_ocsvm = ax_ocsvm.scatter([], [], c='yellow', marker='v', label='Anomalies')
ax_ocsvm.legend(loc='upper right')

# Hide unused subplots to maintain a clean interface.
for ax in [fig.add_subplot(gs[3, 2]),
           fig.add_subplot(gs[4, :]),
           fig.add_subplot(gs[5, :])]:
    ax.axis('off')

# --------------------------------------
#            SLIDER SETUP
# --------------------------------------
# Define GridSpec for arranging sliders and controls neatly.
gs_weight_sliders = gs[4, 0:3].subgridspec(3, 3, hspace=0.5, wspace=0.4)

# Create axes for weight sliders and final threshold.
ax_weight_ilof = fig.add_subplot(gs_weight_sliders[0, 0])
ax_weight_ewma = fig.add_subplot(gs_weight_sliders[0, 1])
ax_weight_sklof = fig.add_subplot(gs_weight_sliders[0, 2])
ax_weight_iso = fig.add_subplot(gs_weight_sliders[1, 0])
ax_weight_ocsvm = fig.add_subplot(gs_weight_sliders[1, 1])
ax_final_threshold = fig.add_subplot(gs_weight_sliders[1, 2])

# Initialize sliders with skyblue color for better readability.
# Note: Weight sliders are optional in majority voting. You can comment them out if not needed.
slider_weight_ilof = Slider(ax=ax_weight_ilof, label='ILOF Weight', valmin=0.0, valmax=5.0, valinit=INITIAL_WEIGHT_ILOF, valstep=0.1, color='skyblue')
slider_weight_ewma = Slider(ax=ax_weight_ewma, label='EWMA Weight', valmin=0.0, valmax=5.0, valinit=INITIAL_WEIGHT_EWMA, valstep=0.1, color='skyblue')
slider_weight_sklof = Slider(ax=ax_weight_sklof, label='SKLearn LOF Weight', valmin=0.0, valmax=5.0, valinit=INITIAL_WEIGHT_SKLOF, valstep=0.1, color='skyblue')
slider_weight_iso = Slider(ax=ax_weight_iso, label='Isolation Forest Weight', valmin=0.0, valmax=5.0, valinit=INITIAL_WEIGHT_ISOFOREST, valstep=0.1, color='skyblue')
slider_weight_ocsvm = Slider(ax=ax_weight_ocsvm, label='One-Class SVM Weight', valmin=0.0, valmax=5.0, valinit=INITIAL_WEIGHT_OCSVM, valstep=0.1, color='skyblue')
slider_final_threshold = Slider(ax=ax_final_threshold, label='Voting Threshold', valmin=1, valmax=5, valinit=FINAL_THRESHOLD, valstep=1, color='skyblue')

# Create GridSpec for data generation parameter sliders.
gs_data_sliders = gs[5, 0:3].subgridspec(2, 3, hspace=0.6, wspace=0.4)

# Create axes for data generation parameter sliders.
ax_slider_season_amp = fig.add_subplot(gs_data_sliders[0, 0])
ax_slider_noise = fig.add_subplot(gs_data_sliders[0, 1])
ax_slider_anomaly = fig.add_subplot(gs_data_sliders[0, 2])
ax_slider_window = fig.add_subplot(gs_data_sliders[1, 0])
ax_slider_k = fig.add_subplot(gs_data_sliders[1, 1])
ax_slider_threshold_lof = fig.add_subplot(gs_data_sliders[1, 2])

# Initialize data generation sliders with skyblue color.
slider_season_amp = Slider(ax=ax_slider_season_amp, label='Seasonality Amp', valmin=1, valmax=10, valinit=INITIAL_SEASONALITY_AMP, valstep=0.1, color='skyblue')
slider_noise = Slider(ax=ax_slider_noise, label='Noise Level', valmin=0.1, valmax=5.0, valinit=INITIAL_NOISE_LEVEL, valstep=0.1, color='skyblue')
slider_anomaly = Slider(ax=ax_slider_anomaly, label='Anomaly Prob', valmin=0.0, valmax=0.1, valinit=INITIAL_ANOMALY_PROB, valstep=0.01, color='skyblue')
slider_window = Slider(ax=ax_slider_window, label='Window Size', valmin=50, valmax=1000, valinit=INITIAL_WINDOW_SIZE, valstep=10, color='skyblue')
slider_k = Slider(ax=ax_slider_k, label='K Neighbors', valmin=5, valmax=50, valinit=INITIAL_K_NEIGHBORS, valstep=1, color='skyblue')
slider_threshold_lof = Slider(ax=ax_slider_threshold_lof, label='LOF Threshold', valmin=1.0, valmax=3.0, valinit=INITIAL_THRESHOLD_LOF, valstep=0.1, color='skyblue')

# --------------------------------------
#            BUTTON SETUP
# --------------------------------------
# Create a reset button to allow users to reset all parameters and data.
ax_reset = fig.add_subplot(gs[5, 2])
button_reset = Button(ax_reset, 'Reset All', color='lightgrey', hovercolor='0.975')
ax_reset.axis('off')  # Hide the axis for the button.

# Create a Play/Pause button to control the animation.
button_play_ax = fig.add_axes([0.85, 0.01, 0.07, 0.04])  # Position manually.
button_play = Button(button_play_ax, 'Pause', color='lightgrey', hovercolor='0.975')

# Create a Save Plot button to save the current state of the plots.
button_save_ax = fig.add_axes([0.75, 0.01, 0.07, 0.04])  # Position manually.
button_save = Button(button_save_ax, 'Save Plot', color='lightgrey', hovercolor='0.975')

# ======================================
#       INITIALIZE DATA GENERATION PARAMETERS
# ======================================
seasonality_amp = INITIAL_SEASONALITY_AMP  # Corrected variable name
noise_level = INITIAL_NOISE_LEVEL
anomaly_prob = INITIAL_ANOMALY_PROB
THRESHOLD_FINAL = FINAL_THRESHOLD  # Voting threshold

# ======================================
#          INITIALIZE DETECTORS
# ======================================
detector_ilof = ILOF(k=INITIAL_K_NEIGHBORS, window_size=INITIAL_WINDOW_SIZE, threshold=INITIAL_THRESHOLD_LOF)
detector_ewma = EWMADetector(window_size=INITIAL_WINDOW_SIZE, alpha=0.3, threshold=3.0)
detector_sklof = SKLearnLOF(window_size=INITIAL_WINDOW_SIZE, n_neighbors=INITIAL_K_NEIGHBORS, contamination=INITIAL_ANOMALY_PROB, threshold=1.5)
detector_iso = IsolationForestDetector(window_size=INITIAL_WINDOW_SIZE, contamination=INITIAL_ANOMALY_PROB, n_estimators=100, threshold=0.0)
detector_ocsvm = OneClassSVMDetector(window_size=INITIAL_WINDOW_SIZE, kernel='rbf', gamma='scale', nu=INITIAL_ANOMALY_PROB, threshold=0.0)

# ======================================
#           INITIALIZE DATA STREAM
# ======================================
stream = data_stream()

# Initialize lists to store anomalies detected by individual detectors.
anomalies_ilof_x, anomalies_ilof_y = [], []
anomalies_ewma_x, anomalies_ewma_y = [], []
anomalies_sklof_x, anomalies_sklof_y = [], []
anomalies_iso_x, anomalies_iso_y = [], []
anomalies_ocsvm_x, anomalies_ocsvm_y = [], []

# ======================================
#            PLOT INITIALIZATION
# ======================================
def init_plot():
    """
    Initializes the plots with default axes limits and empty data.
    """
    ax_main.set_xlim(0, 200)
    ax_main.set_ylim(-20, 20)
    ax_ilof.set_xlim(0, 200)
    ax_ilof.set_ylim(-20, 20)
    ax_ewma.set_xlim(0, 200)
    ax_ewma.set_ylim(-20, 20)
    ax_sklof.set_xlim(0, 200)
    ax_sklof.set_ylim(-20, 20)
    ax_iso.set_xlim(0, 200)
    ax_iso.set_ylim(-20, 20)
    ax_ocsvm.set_xlim(0, 200)
    ax_ocsvm.set_ylim(-20, 20)
    logging.info('Plot initialized.')
    return (line_normal, scatter_anomalies_main,
            line_ilof, scatter_ilof,
            line_ewma, scatter_ewma,
            line_sklof, scatter_sklof,
            line_iso, scatter_iso,
            line_ocsvm, scatter_ocsvm)

# ======================================
#         UPDATE PARAMETERS CALLBACK
# ======================================
def update_parameters(val):
    """
    Callback function to update parameters based on slider values.
    """
    global seasonality_amp, noise_level, anomaly_prob, THRESHOLD_FINAL
    # Update data generation parameters from sliders.
    seasonality_amp = slider_season_amp.val
    noise_level = slider_noise.val
    anomaly_prob = slider_anomaly.val
    THRESHOLD_FINAL = slider_final_threshold.val  # Voting threshold

    # Update detectors' weights from sliders (optional in majority voting).
    # If weights are no longer used, you can comment them out or keep for flexibility.
    weight_ilof = slider_weight_ilof.val
    weight_ewma = slider_weight_ewma.val
    weight_sklof = slider_weight_sklof.val
    weight_iso = slider_weight_iso.val
    weight_ocsvm = slider_weight_ocsvm.val
    lof_threshold = slider_threshold_lof.val

    # Update detectors' parameters based on sliders.
    detector_ilof.update_parameters(k=int(slider_k.val), window_size=int(slider_window.val), threshold=slider_threshold_lof.val)
    detector_sklof.update_parameters(n_neighbors=int(slider_k.val), contamination=anomaly_prob, window_size=int(slider_window.val), threshold=slider_threshold_lof.val)
    detector_ewma.update_parameters(window_size=int(slider_window.val))
    detector_iso.update_parameters(window_size=int(slider_window.val))
    detector_ocsvm.update_parameters(window_size=int(slider_window.val))

    # Log updated weights for debugging (optional).
    logging.debug(f'Weights updated: ILOF={weight_ilof}, EWMA={weight_ewma}, SKLOF={weight_sklof}, IsolationForest={weight_iso}, OneClassSVM={weight_ocsvm}')

# ======================================
#                RESET CALLBACK
# ======================================
def reset(event):
    """
    Callback function to reset all sliders to their initial values and clear all data.
    """
    global detector_ilof, detector_ewma, detector_sklof, detector_iso, detector_ocsvm
    logging.info('Reset button clicked. Resetting sliders and clearing data.')
    # Reset sliders to their initial values.
    slider_window.reset()
    slider_k.reset()
    slider_threshold_lof.reset()
    slider_season_amp.reset()
    slider_noise.reset()
    slider_anomaly.reset()
    slider_weight_ilof.reset()
    slider_weight_ewma.reset()
    slider_weight_sklof.reset()
    slider_weight_iso.reset()
    slider_weight_ocsvm.reset()
    slider_final_threshold.reset()

    # Clear all data lists.
    xdata.clear()
    ydata.clear()
    anomalies_main_x.clear()
    anomalies_main_y.clear()
    anomalies_ilof_x.clear()
    anomalies_ilof_y.clear()
    anomalies_ewma_x.clear()
    anomalies_ewma_y.clear()
    anomalies_sklof_x.clear()
    anomalies_sklof_y.clear()
    anomalies_iso_x.clear()
    anomalies_iso_y.clear()
    anomalies_ocsvm_x.clear()
    anomalies_ocsvm_y.clear()

    # Update all plots with cleared data.
    line_normal.set_data(xdata, ydata)
    scatter_anomalies_main.set_offsets(np.c_[anomalies_main_x, anomalies_main_y])

    line_ilof.set_data(xdata, ydata)
    scatter_ilof.set_offsets(np.c_[anomalies_ilof_x, anomalies_ilof_y])

    line_ewma.set_data(xdata, ydata)
    scatter_ewma.set_offsets(np.c_[anomalies_ewma_x, anomalies_ewma_y])

    line_sklof.set_data(xdata, ydata)
    scatter_sklof.set_offsets(np.c_[anomalies_sklof_x, anomalies_sklof_y])

    line_iso.set_data(xdata, ydata)
    scatter_iso.set_offsets(np.c_[anomalies_iso_x, anomalies_iso_y])

    line_ocsvm.set_data(xdata, ydata)
    scatter_ocsvm.set_offsets(np.c_[anomalies_ocsvm_x, anomalies_ocsvm_y])

    # Reinitialize detectors to their initial states.
    detector_ilof = ILOF(k=INITIAL_K_NEIGHBORS, window_size=INITIAL_WINDOW_SIZE, threshold=INITIAL_THRESHOLD_LOF)
    detector_ewma = EWMADetector(window_size=INITIAL_WINDOW_SIZE, alpha=0.3, threshold=3.0)
    detector_sklof = SKLearnLOF(window_size=INITIAL_WINDOW_SIZE, n_neighbors=INITIAL_K_NEIGHBORS, contamination=INITIAL_ANOMALY_PROB, threshold=1.5)
    detector_iso = IsolationForestDetector(window_size=INITIAL_WINDOW_SIZE, contamination=INITIAL_ANOMALY_PROB, n_estimators=100, threshold=0.0)
    detector_ocsvm = OneClassSVMDetector(window_size=INITIAL_WINDOW_SIZE, kernel='rbf', gamma='scale', nu=INITIAL_ANOMALY_PROB, threshold=0.0)

    # Reset axes limits to initial values.
    ax_main.set_xlim(0, 200)
    ax_main.set_ylim(-20, 20)
    ax_ilof.set_xlim(0, 200)
    ax_ilof.set_ylim(-20, 20)
    ax_ewma.set_xlim(0, 200)
    ax_ewma.set_ylim(-20, 20)
    ax_sklof.set_xlim(0, 200)
    ax_sklof.set_ylim(-20, 20)
    ax_iso.set_xlim(0, 200)
    ax_iso.set_ylim(-20, 20)
    ax_ocsvm.set_xlim(0, 200)
    ax_ocsvm.set_ylim(-20, 20)

    logging.info('Reset completed.')
    fig.canvas.draw()

# ======================================
#           PLAY/PAUSE CALLBACK
# ======================================
is_paused = False  # Global flag to control animation state.

def toggle_pause(event):
    """
    Callback function to toggle the animation between play and pause states.
    """
    global is_paused
    if is_paused:
        ani.event_source.start()
        button_play.label.set_text('Pause')
        logging.info('Animation resumed.')
    else:
        ani.event_source.stop()
        button_play.label.set_text('Play')
        logging.info('Animation paused.')
    is_paused = not is_paused

# ======================================
#           SAVE PLOT CALLBACK
# ======================================
def save_plot(event):
    """
    Callback function to save the current state of all plots as an image.
    """
    filename = f'anomaly_detection_plot_{np.random.randint(1000)}.png'
    fig.savefig(filename)
    logging.info(f'Plot saved as {filename}')

# ======================================
#       REGISTER CALLBACK FUNCTIONS
# ======================================
# Connect the update function to all sliders.
slider_weight_ilof.on_changed(update_parameters)
slider_weight_ewma.on_changed(update_parameters)
slider_weight_sklof.on_changed(update_parameters)
slider_weight_iso.on_changed(update_parameters)
slider_weight_ocsvm.on_changed(update_parameters)
slider_final_threshold.on_changed(update_parameters)

slider_season_amp.on_changed(update_parameters)
slider_noise.on_changed(update_parameters)
slider_anomaly.on_changed(update_parameters)
slider_window.on_changed(update_parameters)
slider_k.on_changed(update_parameters)
slider_threshold_lof.on_changed(update_parameters)

# Connect the reset button to its callback.
button_reset.on_clicked(reset)

# Connect the play/pause button to its callback.
button_play.on_clicked(toggle_pause)

# Connect the save plot button to its callback.
button_save.on_clicked(save_plot)

# ======================================
#            UPDATE PLOT FUNCTION
# ======================================
def update_plot(frame):
    """
    Updates the plots with new data points from the data stream using a majority voting approach.
    """
    global detector_ilof, detector_ewma, detector_sklof, detector_iso, detector_ocsvm
    value = next(stream)
    xdata.append(frame)
    ydata.append(value[0])

    # Update the main data stream line.
    line_normal.set_data(xdata, ydata)

    # Update individual plots' lines.
    line_ilof.set_data(xdata, ydata)
    line_ewma.set_data(xdata, ydata)
    line_sklof.set_data(xdata, ydata)
    line_iso.set_data(xdata, ydata)
    line_ocsvm.set_data(xdata, ydata)

    # Retrieve current weights and thresholds from sliders.
    # Note: Weights are no longer needed for majority voting but retained if needed elsewhere.
    weight_ilof = slider_weight_ilof.val
    weight_ewma = slider_weight_ewma.val
    weight_sklof = slider_weight_sklof.val
    weight_iso = slider_weight_iso.val
    weight_ocsvm = slider_weight_ocsvm.val
    threshold_final = slider_final_threshold.val  # Voting threshold
    lof_threshold = slider_threshold_lof.val  # Not needed in majority voting unless individual detector thresholds are dynamic

    # Collect binary anomaly decisions from each detector.
    decision_ilof = detector_ilof.fit_new_point(value)
    decision_ewma = detector_ewma.fit_new_point(value)
    decision_sklof = detector_sklof.fit_new_point(value)
    decision_iso = detector_iso.fit_new_point(value)
    decision_ocsvm = detector_ocsvm.fit_new_point(value)

    # Aggregate the decisions.
    anomaly_votes = decision_ilof + decision_ewma + decision_sklof + decision_iso + decision_ocsvm
    logging.debug(f'Time Step {frame}: Anomaly Votes = {anomaly_votes}, Voting Threshold = {threshold_final}')

    # Determine if the data point is anomalous based on majority voting.
    if anomaly_votes >= threshold_final:
        anomalies_main_x.append(frame)
        anomalies_main_y.append(value[0])

    scatter_anomalies_main.set_offsets(np.c_[anomalies_main_x, anomalies_main_y])

    # Update individual anomalies with distinct colors and markers based on binary decisions.
    if decision_ilof:
        anomalies_ilof_x.append(frame)
        anomalies_ilof_y.append(value[0])
    scatter_ilof.set_offsets(np.c_[anomalies_ilof_x, anomalies_ilof_y])

    if decision_ewma:
        anomalies_ewma_x.append(frame)
        anomalies_ewma_y.append(value[0])
    scatter_ewma.set_offsets(np.c_[anomalies_ewma_x, anomalies_ewma_y])

    if decision_sklof:
        anomalies_sklof_x.append(frame)
        anomalies_sklof_y.append(value[0])
    scatter_sklof.set_offsets(np.c_[anomalies_sklof_x, anomalies_sklof_y])

    if decision_iso:
        anomalies_iso_x.append(frame)
        anomalies_iso_y.append(value[0])
    scatter_iso.set_offsets(np.c_[anomalies_iso_x, anomalies_iso_y])

    if decision_ocsvm:
        anomalies_ocsvm_x.append(frame)
        anomalies_ocsvm_y.append(value[0])
    scatter_ocsvm.set_offsets(np.c_[anomalies_ocsvm_x, anomalies_ocsvm_y])

    # Smoothly adjust x-axis limits to accommodate new data points.
    xmin, xmax = ax_main.get_xlim()
    if frame >= xmax - 100:  # Start extending earlier for smoother transition.
        ax_main.set_xlim(xmin, xmax + 100)
        ax_ilof.set_xlim(xmin, xmax + 100)
        ax_ewma.set_xlim(xmin, xmax + 100)
        ax_sklof.set_xlim(xmin, xmax + 100)
        ax_iso.set_xlim(xmin, xmax + 100)
        ax_ocsvm.set_xlim(xmin, xmax + 100)
        logging.debug(f'Extended x-axis to {xmax + 100}')

    # Smoothly adjust y-axis limits based on incoming data.
    current_y = value[0]
    ymin, ymax = ax_main.get_ylim()
    buffer = 5
    if current_y >= ymax - buffer:
        ax_main.set_ylim(ymin, current_y + buffer)
        ax_ilof.set_ylim(ymin, current_y + buffer)
        ax_ewma.set_ylim(ymin, current_y + buffer)
        ax_sklof.set_ylim(ymin, current_y + buffer)
        ax_iso.set_ylim(ymin, current_y + buffer)
        ax_ocsvm.set_ylim(ymin, current_y + buffer)
        logging.debug(f'Extended y-axis upper limit to {current_y + buffer}')
    elif current_y <= ymin + buffer:
        ax_main.set_ylim(current_y - buffer, ymax)
        ax_ilof.set_ylim(current_y - buffer, ymax)
        ax_ewma.set_ylim(current_y - buffer, ymax)
        ax_sklof.set_ylim(current_y - buffer, ymax)
        ax_iso.set_ylim(current_y - buffer, ymax)
        ax_ocsvm.set_ylim(current_y - buffer, ymax)
        logging.debug(f'Extended y-axis lower limit to {current_y - buffer}')

    # Redraw the canvas to reflect updates.
    fig.canvas.draw()
    return (line_normal, scatter_anomalies_main,
            line_ilof, scatter_ilof,
            line_ewma, scatter_ewma,
            line_sklof, scatter_sklof,
            line_iso, scatter_iso,
            line_ocsvm, scatter_ocsvm)

# ======================================
#         CREATE ANIMATION OBJECT
# ======================================
# Create the animation using FuncAnimation with blitting disabled for compatibility.
ani = animation.FuncAnimation(fig, update_plot, init_func=init_plot, blit=False, interval=50)

# ======================================
#                DISPLAY
# ======================================
# Display the interactive plots.
plt.show()
