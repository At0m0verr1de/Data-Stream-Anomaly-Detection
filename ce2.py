import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque
from sklearn.neighbors import NearestNeighbors
from matplotlib.widgets import Button
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

        # Varying noise levels.
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
#        ANOMALY DETECTOR CLASS
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

# ======================================
#          VISUALIZATION SETUP
# ======================================
# Hide the default matplotlib toolbar for a cleaner interface.
plt.rcParams['toolbar'] = 'None'

# Create the main figure with a specified size and layout.
fig = plt.figure(figsize=(12, 8), constrained_layout=True)
manager = plt.get_current_fig_manager()
try:
    manager.full_screen_toggle()  # Attempt to make the plot full screen.
except AttributeError:
    pass  # If full_screen_toggle is not available, proceed without it.

# Define a GridSpec layout to organize the plot and controls.
gs = fig.add_gridspec(3, 2)

# --------------------------------------
#       MAIN PLOT FOR ILOF ANOMALY DETECTION
# --------------------------------------
ax_main = fig.add_subplot(gs[0:2, :])
ax_main.set_title('Real-Time Data Stream Anomaly Detection (ILOF)', fontsize=16)
ax_main.set_xlabel('Time Steps', fontsize=12)
ax_main.set_ylabel('Value', fontsize=12)
ax_main.grid(True)

# Initialize data lists for the main plot.
xdata, ydata = [], []
anomalies_main_x, anomalies_main_y = [], []
line_normal, = ax_main.plot([], [], color='blue', label='Data Stream')
scatter_anomalies_main = ax_main.scatter([], [], c='darkred', marker='o', s=50, label='ILOF Anomalies')
ax_main.legend(loc='upper right')

# --------------------------------------
#            BUTTON SETUP
# --------------------------------------
# Create a reset button to allow users to reset all parameters and data.
ax_reset = fig.add_subplot(gs[2, 0])
button_reset = Button(ax_reset, 'Reset All', color='lightgrey', hovercolor='0.975')
ax_reset.axis('off')  # Hide the axis for the button.

# Create a Play/Pause button to control the animation.
button_play_ax = fig.add_axes([0.7, 0.05, 0.1, 0.04])  # Position manually.
button_play = Button(button_play_ax, 'Pause', color='lightgrey', hovercolor='0.975')

# Create a Save Plot button to save the current state of the plots.
button_save_ax = fig.add_axes([0.81, 0.05, 0.1, 0.04])  # Position manually.
button_save = Button(button_save_ax, 'Save Plot', color='lightgrey', hovercolor='0.975')

# ======================================
#       INITIALIZE DATA GENERATION PARAMETERS
# ======================================
seasonality_amp = INITIAL_SEASONALITY_AMP  # Corrected variable name
noise_level = INITIAL_NOISE_LEVEL
anomaly_prob = INITIAL_ANOMALY_PROB
THRESHOLD_FINAL = FINAL_THRESHOLD  # Voting threshold

# ======================================
#          INITIALIZE ILOF DETECTOR
# ======================================
detector_ilof = ILOF(k=INITIAL_K_NEIGHBORS, window_size=INITIAL_WINDOW_SIZE, threshold=INITIAL_THRESHOLD_LOF)

# ======================================
#           INITIALIZE DATA STREAM
# ======================================
stream = data_stream()

# Initialize lists to store anomalies detected by ILOF.
anomalies_ilof_x, anomalies_ilof_y = [], []

# ======================================
#            PLOT INITIALIZATION
# ======================================
def init_plot():
    """
    Initializes the plot with default axes limits and empty data.
    """
    ax_main.set_xlim(0, 200)
    ax_main.set_ylim(-20, 20)
    logging.info('Plot initialized.')
    return (line_normal, scatter_anomalies_main)

# ======================================
#                RESET CALLBACK
# ======================================
def reset(event):
    """
    Callback function to reset all data and reinitialize the detector.
    """
    global detector_ilof, xdata, ydata, anomalies_main_x, anomalies_main_y
    logging.info('Reset button clicked. Resetting data and reinitializing detector.')
    
    # Clear all data lists.
    xdata.clear()
    ydata.clear()
    anomalies_main_x.clear()
    anomalies_main_y.clear()
    
    # Update the plot with cleared data.
    line_normal.set_data(xdata, ydata)
    scatter_anomalies_main.set_offsets(np.c_[anomalies_main_x, anomalies_main_y])
    
    # Reinitialize the detector.
    detector_ilof = ILOF(k=INITIAL_K_NEIGHBORS, window_size=INITIAL_WINDOW_SIZE, threshold=INITIAL_THRESHOLD_LOF)
    
    # Reset axes limits to initial values.
    ax_main.set_xlim(0, 200)
    ax_main.set_ylim(-20, 20)
    
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
    Callback function to save the current state of the plot as an image.
    """
    filename = f'anomaly_detection_plot_{np.random.randint(1000)}.png'
    fig.savefig(filename)
    logging.info(f'Plot saved as {filename}')

# ======================================
#       REGISTER CALLBACK FUNCTIONS
# ======================================
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
    Updates the plot with new data points from the data stream using ILOF detection.
    """
    global detector_ilof, anomalies_main_x, anomalies_main_y
    value = next(stream)
    xdata.append(frame)
    ydata.append(value[0])

    # Update the main data stream line.
    line_normal.set_data(xdata, ydata)

    # Retrieve binary anomaly decision from ILOF detector.
    decision_ilof = detector_ilof.fit_new_point(value)

    # Determine if the data point is anomalous based on ILOF.
    if decision_ilof:
        anomalies_main_x.append(frame)
        anomalies_main_y.append(value[0])

    scatter_anomalies_main.set_offsets(np.c_[anomalies_main_x, anomalies_main_y])

    # Smoothly adjust x-axis limits to accommodate new data points.
    xmin, xmax = ax_main.get_xlim()
    if frame >= xmax - 100:  # Start extending earlier for smoother transition.
        ax_main.set_xlim(xmin, xmax + 100)
        logging.debug(f'Extended x-axis to {xmax + 100}')

    # Smoothly adjust y-axis limits based on incoming data.
    current_y = value[0]
    ymin, ymax = ax_main.get_ylim()
    buffer = 5
    if current_y >= ymax - buffer:
        ax_main.set_ylim(ymin, current_y + buffer)
        logging.debug(f'Extended y-axis upper limit to {current_y + buffer}')
    elif current_y <= ymin + buffer:
        ax_main.set_ylim(current_y - buffer, ymax)
        logging.debug(f'Extended y-axis lower limit to {current_y - buffer}')

    # Redraw the canvas to reflect updates.
    fig.canvas.draw()
    return (line_normal, scatter_anomalies_main)

# ======================================
#         CREATE ANIMATION OBJECT
# ======================================
# Create the animation using FuncAnimation with blitting disabled for compatibility.
ani = animation.FuncAnimation(fig, update_plot, init_func=init_plot, blit=False, interval=50)

# ======================================
#                DISPLAY
# ======================================
# Display the interactive plot.
plt.show()
