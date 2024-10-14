# Project Title:
# Efficient Data Stream Anomaly Detection

# Project Description:
# Your task is to develop a Python script capable of detecting anomalies in a continuous data stream.
# This stream, simulating real-time sequences of floating-point numbers, could represent various metrics
# such as financial transactions or system metrics. Your focus will be on identifying unusual patterns,
# such as exceptionally high values or deviations from the norm.

# Objectives:
# 1. Algorithm Selection: Identify and implement a suitable algorithm for anomaly detection, capable of adapting to concept drift and seasonal variations.
# 2. Data Stream Simulation: Design a function to emulate a data stream, incorporating regular patterns, seasonal elements, and random noise.
# 3. Anomaly Detection: Develop a real-time mechanism to accurately flag anomalies as the data is streamed.
# 4. Optimization: Ensure the algorithm is optimized for both speed and efficiency.
# 5. Visualization: Create a straightforward real-time visualization tool to display both the data stream and any detected anomalies.

# Requirements:
# 1. The project must be implemented using Python 3.x.
# 2. Your code should be thoroughly documented, with comments to explain key sections.
# 3. Include a concise explanation of your chosen algorithm and its effectiveness.
# 4. Ensure robust error handling and data validation.
# 5. Limit the use of external libraries. If necessary, include a requirements.txt file.


# NOTE - Open the Application in Full Screen mode only
# Play/Pause the siumulation using the Pause button
# Save the plot for future purposes using Save Button
# On closing the matplot simulation, all the anomalies are written down into Anomalies.csv


# Algorithm Selection

# Incremental Local Outlier Factor (ILOF):

# The Incremental Local Outlier Factor (ILOF) excels in environments characterized by dynamic
# data streams and concept drift, where the underlying data distribution evolves over time.
# ILOF's primary strength lies in its ability to adaptively update its anomaly detection
# model in real-time, leveraging a sliding window mechanism that continuously incorporates
# new data while discarding outdated information. This incremental learning approach ensures
# that ILOF remains responsive to changing patterns and emerging trends, making it particularly
# effective for applications such as real-time monitoring of financial transactions, network security,
# and sensor data analysis in IoT systems. However, ILOF's adaptability comes with trade-offs; its
# precision in pinpointing exact anomalies can be lower compared to more static models like LOF.
# Additionally, the computational overhead of maintaining and updating the model incrementally
# may pose challenges in scenarios with extremely high-velocity data streams, potentially impacting
# scalability and real-time performance.


# Local Outlier Factor (LOF):

# Local Outlier Factor (LOF) is renowned for its high precision in identifying outliers within
# static or relatively stable datasets. By evaluating the local density of data points and comparing
# it to their neighbors, LOF effectively distinguishes anomalies that significantly deviate from
# their immediate surroundings. This makes LOF exceptionally suitable for applications requiring
# accurate and reliable anomaly detection, such as fraud detection in financial systems, quality
# control in manufacturing processes, and intrusion detection in cybersecurity. The algorithm's
# mature and optimized implementation in libraries like Scikit-Learn ensures robust performance
# and ease of integration. However, LOF has inherent limitations when applied to dynamic environments
# with frequent concept drift. Its static nature means that LOF does not inherently adapt to evolving
# data distributions, leading to potential decreased detection accuracy as patterns shift over time.
# Additionally, LOF can be sensitive to parameter settings (e.g., the number of neighbors), which may
# require careful tuning to balance sensitivity and specificity in diverse applications.


# Reason for including the Justified/Unjustified factor to anomalies using an example -
# For example, you could get bid ask data where spreads are in the range of 1 tick to 3 ticks.
# Now, suddenly you get 20-25 tick spreads. This could be due to a drastic macro event. Or it
# could be a derivative like Options whose spreads have increased due to massive underlying move
# So if you do get abnormal variation in the data- you will need a classifier algorithm to say if
# it is justified by some macro/underlying move or it is indeed due to data discrepancy

# Although this feature might not be working 100% correctly, it has proved useful in offline testing


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random                                                       # For the Data Stream
from collections import deque                                       # Organising Data
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor  # LOF and ILOF
from matplotlib.widgets import Button
import logging                                                      # Terminal Logging Statements
import csv                                                          # Added for CSV operations
import atexit                                                       # Ensures CSV file closure on exit

# New Imports for Classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ======================================
#          CONFIGURATION SETUP
# ======================================

# Configure logging to monitor the application's behavior and debug.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters for data generation and anomaly detection.
INITIAL_WINDOW_SIZE = 200
INITIAL_K_NEIGHBORS = 25
INITIAL_THRESHOLD_LOF_ILOF = 2.0
INITIAL_THRESHOLD_LOF_SKLOF = 1.5
INITIAL_SEASONALITY_AMP = 5
INITIAL_NOISE_LEVEL = 1
INITIAL_ANOMALY_PROB = 0.02

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
        # Introduces Concept Drift by changing seasonality parameters periodically.
        if time_step % 1000 == 0 and time_step > 0:
            season_length = random.randint(30, 70)  # Dynamically changing season length
            seasonality_amp = random.uniform(3, 7)
            logging.info(f'Concept drift occurred at time_step {time_step}: New seasonality_amp = {seasonality_amp:.2f}, New season_length = {season_length}')

        # Introduces trend shifts periodically.
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
        self.current_lof_score = None  # To store the latest LOF score
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
            self.current_lof_score = lof[-1]
            logging.debug(f'ILOF - Computed LOF score for new point: {self.current_lof_score:.2f}')
            # Binary decision based on threshold
            is_anomaly = 1 if self.current_lof_score > self.threshold else 0
            return is_anomaly
        else:
            logging.debug('ILOF - Not enough data to compute LOF. Returning normal.')
            self.current_lof_score = None
            return 0  # Normal if insufficient data

    def get_current_lof_score(self):
        """
        Returns the latest LOF score.
        """
        return self.current_lof_score

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
            self.current_lof_score = None
            logging.info('ILOF - Detector reset due to parameter changes.')

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
        self.current_lof_score = None  # To store the latest LOF score
        logging.info(f'SKLearnLOF initialized with window_size={self.window_size}, n_neighbors={self.n_neighbors}, contamination={self.contamination}, threshold={self.threshold}')

    def fit_new_point(self, point):
        self.history.append(point[0])
        if len(self.history) >= self.n_neighbors + 1:
            data = np.array(self.history).reshape(-1, 1)
            self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination, novelty=False)
            y_pred = self.model.fit_predict(data)
            anomaly_score = -self.model.negative_outlier_factor_[-1]
            self.current_lof_score = anomaly_score
            logging.debug(f'SKLearnLOF - Point: {point[0]:.2f}, LOF Score: {anomaly_score:.2f}')
            # Binary decision based on threshold
            is_anomaly = 1 if anomaly_score > self.threshold else 0
            return is_anomaly
        else:
            logging.debug('SKLearnLOF - Not enough data to compute LOF. Returning normal.')
            self.current_lof_score = None
            return 0  # Normal if insufficient data

    def get_current_lof_score(self):
        """
        Returns the latest LOF score.
        """
        return self.current_lof_score

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
            self.current_lof_score = None
            logging.info('SKLearnLOF - Detector reset due to parameter changes.')

# ======================================
#          VISUALIZATION SETUP
# ======================================

# Hide the default matplotlib toolbar for a cleaner interface.
plt.rcParams['toolbar'] = 'None'

# Create the main figure with a specified size and layout.
fig = plt.figure(figsize=(18, 14), constrained_layout=True)
manager = plt.get_current_fig_manager()

try:
    manager.full_screen_toggle()  # Attempt to make the plot full screen.
except AttributeError:
    pass  # If full_screen_toggle is not available, proceed without it.

# Define a GridSpec layout to organize multiple plots and controls.
gs = fig.add_gridspec(4, 2, height_ratios=[3, 3, 2, 1])

# ------------------------------------------------------
#       MAIN PLOT FOR COMBINED ANOMALY DETECTION
# ------------------------------------------------------
ax_main = fig.add_subplot(gs[0:2, 0:2])
ax_main.set_title('Real-Time Data Stream Combined Anomaly Detection (ILOF | SKLearn LOF)', fontsize=18)
ax_main.set_xlabel('Time Steps', fontsize=14)
ax_main.set_ylabel('Value', fontsize=14)
ax_main.grid(True)

# Initialize data lists for the main plot.
xdata, ydata = [], []
anomalies_main_x, anomalies_main_y = [], []
line_normal, = ax_main.plot([], [], color='blue', label='Data Stream')
scatter_anomalies_main = ax_main.scatter([], [], c='darkred', marker='o', s=50, label='Detected Anomalies')
ax_main.legend(loc='upper right')

# --------------------------------------
#        INDIVIDUAL DETECTORS' PLOTS
# --------------------------------------

# ILOF Plot
ax_ilof = fig.add_subplot(gs[2, 0])
ax_ilof.set_title('ILOF Anomalies', fontsize=16)
ax_ilof.set_xlabel('Time Steps', fontsize=12)
ax_ilof.set_ylabel('Value', fontsize=12)
ax_ilof.grid(True)
line_ilof, = ax_ilof.plot([], [], color='green', label='ILOF Data Stream')
scatter_ilof = ax_ilof.scatter([], [], c='orange', marker='x', label='ILOF Anomalies')
ax_ilof.legend(loc='upper right')

# SKLearn LOF Plot
ax_sklof = fig.add_subplot(gs[2, 1])
ax_sklof.set_title('SKLearn LOF Anomalies', fontsize=16)
ax_sklof.set_xlabel('Time Steps', fontsize=12)
ax_sklof.set_ylabel('Value', fontsize=12)
ax_sklof.grid(True)
line_sklof, = ax_sklof.plot([], [], color='purple', label='SKLearn LOF Data Stream')
scatter_sklof = ax_sklof.scatter([], [], c='cyan', marker='s', label='SKLearn LOF Anomalies')
ax_sklof.legend(loc='upper right')

# ------------------------------------------------------------
#            BUTTON SETUP (The colors do not work on Mac)
# ------------------------------------------------------------
# Create a dedicated row for buttons.

# Play/Pause Button
button_play_ax = fig.add_subplot(gs[3, 0])
button_play = Button(button_play_ax, 'Pause', color='#4CAF50', hovercolor='#45a049')  # Green button
button_play.ax.patch.set_edgecolor('black')  # Black border
button_play.ax.patch.set_linewidth(2)        # Border width
button_play.ax.patch.set_facecolor('#4CAF50')  # Initial facecolor
button_play_ax.axis('off')  # Hide the axis for the button.

# Save Plot Button
button_save_ax = fig.add_subplot(gs[3, 1])
button_save = Button(button_save_ax, 'Save Plot', color='#2196F3', hovercolor='#0b7dda')  # Blue button
button_save.ax.patch.set_edgecolor('black')  # Black border
button_save.ax.patch.set_linewidth(2)        # Border width
button_save.ax.patch.set_facecolor('#2196F3')  # Initial facecolor
button_save_ax.axis('off')  # Hide the axis for the button.

# ================================================
#       INITIALIZE DATA GENERATION PARAMETERS
# ================================================
seasonality_amp = INITIAL_SEASONALITY_AMP
noise_level = INITIAL_NOISE_LEVEL
anomaly_prob = INITIAL_ANOMALY_PROB

# ======================================
#          INITIALIZE DETECTORS
# ======================================
detector_ilof = ILOF(k=INITIAL_K_NEIGHBORS, window_size=INITIAL_WINDOW_SIZE, threshold=INITIAL_THRESHOLD_LOF_ILOF)
detector_sklof = SKLearnLOF(window_size=INITIAL_WINDOW_SIZE, n_neighbors=INITIAL_K_NEIGHBORS, contamination=INITIAL_ANOMALY_PROB, threshold=INITIAL_THRESHOLD_LOF_SKLOF)

# ======================================
#           INITIALIZE DATA STREAM
# ======================================
stream = data_stream()

# Initialize lists to store anomalies detected by individual detectors.
anomalies_ilof_x, anomalies_ilof_y = [], []
anomalies_sklof_x, anomalies_sklof_y = [], []

# ======================================
#            CSV LOGGING SETUP
# ======================================
# Open the CSV file for writing anomalies
csv_filename = 'anomalies.csv'
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
# Write the header with additional columns for classification
csv_writer.writerow(['Time Step', 'Value', 'Detector', 'Classification', 'Confidence'])
logging.info(f'CSV file {csv_filename} created and header written.')

# Ensure the file is closed when the script exits
def close_csv():
    csv_file.close()
    logging.info('CSV file closed.')

atexit.register(close_csv)

# ======================================
#            ANOMALY CLASSIFIER
# ======================================
# Initialize a simple classifier (e.g., Logistic Regression)
# For demonstration, we'll use synthetic labels. In practice, you'd need labeled data.

# Example: Placeholder for external event data
external_event = {}  # Dictionary to map time_steps to events

# Initialize classifier with a pipeline: StandardScaler followed by LogisticRegression
classifier = make_pipeline(StandardScaler(), LogisticRegression())
# Placeholder training data
# In practice, you'd train this with historical labeled anomalies
X_train = np.array([
    [1.5, 1.7],  # Feature: [ILOF_score, SKLearnLOF_score]
    [2.0, 2.1],
    [0.5, 0.4],
    [3.0, 3.2],
    [0.3, 0.2],
    [2.5, 2.6],
    [0.4, 0.5],
    [3.1, 3.3],
    [0.2, 0.1],
    [2.8, 2.9]
])
y_train = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 1=Justified, 0=Unjustified
classifier.fit(X_train, y_train)
logging.info('Anomaly classifier initialized and trained with placeholder data.')

# ======================================
#            PLOT INITIALIZATION
# ======================================
def init_plot():
    """
    Initializes the plots with default axes limits and empty data.
    """
    # Main plot limits
    ax_main.set_xlim(0, 200)
    ax_main.set_ylim(-20, 20)

    # ILOF plot limits
    ax_ilof.set_xlim(0, 200)
    ax_ilof.set_ylim(-20, 20)

    # SKLearn LOF plot limits
    ax_sklof.set_xlim(0, 200)
    ax_sklof.set_ylim(-20, 20)

    logging.info('Plot initialized.')
    return (line_normal, scatter_anomalies_main,
            line_ilof, scatter_ilof,
            line_sklof, scatter_sklof)

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
        button_play.ax.patch.set_facecolor('#4CAF50')  # Restore original green color
        logging.info('Animation resumed.')
    else:
        ani.event_source.stop()
        button_play.label.set_text('Play')
        button_play.ax.patch.set_facecolor('#f44336')  # Change to red when paused
        logging.info('Animation paused.')
    is_paused = not is_paused

# ======================================
#           SAVE PLOT CALLBACK
# ======================================
def save_plot(event):
    """
    Callback function to save the current state of the plots as an image.
    """
    filename = f'anomaly_detection_plot_{np.random.randint(1000)}.png'
    fig.savefig(filename)
    logging.info(f'Plot saved as {filename}')

# ======================================
#       REGISTER CALLBACK FUNCTIONS
# ======================================
# Connect the play/pause button to its callback.
button_play.on_clicked(toggle_pause)

# Connect the save plot button to its callback.
button_save.on_clicked(save_plot)

# ======================================
#            UPDATE PLOT FUNCTION
# ======================================
def update_plot(frame):
    """
    Updates the plots with new data points from the data stream.
    Also logs detected anomalies to a CSV file.
    """
    global detector_ilof, detector_sklof
    value = next(stream)
    xdata.append(frame)
    ydata.append(value[0])

    # Update the main data stream line.
    line_normal.set_data(xdata, ydata)

    # Update the ILOF plot's data stream line.
    line_ilof.set_data(xdata, ydata)

    # Update the SKLearn LOF plot's data stream line.
    line_sklof.set_data(xdata, ydata)

    # Retrieve anomaly decisions from both detectors.
    decision_ilof = detector_ilof.fit_new_point(value)
    decision_sklof = detector_sklof.fit_new_point(value)

    # Retrieve the latest LOF scores for classification
    lof_ilof = detector_ilof.get_current_lof_score()
    lof_sklof = detector_sklof.get_current_lof_score()

    # Determine which detector(s) identified the anomaly
    detectors = []
    features = []
    if decision_ilof:
        anomalies_ilof_x.append(frame)
        anomalies_ilof_y.append(value[0])
        detectors.append('ILOF')
        if lof_ilof is not None:
            features.append(lof_ilof)
    if decision_sklof:
        anomalies_sklof_x.append(frame)
        anomalies_sklof_y.append(value[0])
        detectors.append('SKLearn LOF')
        if lof_sklof is not None:
            features.append(lof_sklof)

    # If any detector identified an anomaly, classify it
    if detectors and len(features) == 2:
        # Use the LOF scores as features for classification
        # Reshape to match classifier's expected input
        feature_vector = np.array(features).reshape(1, -1)
        classification = classifier.predict(feature_vector)[0]
        confidence = np.max(classifier.predict_proba(feature_vector))

        # Map numerical classification to string labels
        classification_str = 'Justified' if classification == 1 else 'Unjustified'

        anomalies_main_x.append(frame)
        anomalies_main_y.append(value[0])
        scatter_anomalies_main.set_offsets(np.c_[anomalies_main_x, anomalies_main_y])

        # Update individual scatter plots
        if decision_ilof:
            scatter_ilof.set_offsets(np.c_[anomalies_ilof_x, anomalies_ilof_y])
        if decision_sklof:
            scatter_sklof.set_offsets(np.c_[anomalies_sklof_x, anomalies_sklof_y])

        # Write anomaly details to CSV
        csv_writer.writerow([frame, value[0], ', '.join(detectors), classification_str, f'{confidence:.2f}'])
        csv_file.flush()  # Ensure data is written to disk
        logging.info(f'Anomaly detected at time_step {frame}: Value={value[0]:.2f}, Detector(s)={", ".join(detectors)}, Classification={classification_str}, Confidence={confidence:.2f}')

    # Smoothly adjust x-axis limits to accommodate new data points.
    xmin, xmax = ax_main.get_xlim()
    if frame >= xmax - 100:  # Start extending earlier for smoother transition.
        ax_main.set_xlim(xmin, xmax + 100)
        ax_ilof.set_xlim(xmin, xmax + 100)
        ax_sklof.set_xlim(xmin, xmax + 100)
        logging.debug(f'Extended x-axis to {xmax + 100}')

    # Smoothly adjust y-axis limits based on incoming data.
    current_y = value[0]
    ymin, ymax = ax_main.get_ylim()
    buffer = 5
    if current_y >= ymax - buffer:
        ax_main.set_ylim(ymin, current_y + buffer)
        ax_ilof.set_ylim(ymin, current_y + buffer)
        ax_sklof.set_ylim(ymin, current_y + buffer)
        logging.debug(f'Extended y-axis upper limit to {current_y + buffer}')
    elif current_y <= ymin + buffer:
        ax_main.set_ylim(current_y - buffer, ymax)
        ax_ilof.set_ylim(current_y - buffer, ymax)
        ax_sklof.set_ylim(current_y - buffer, ymax)
        logging.debug(f'Extended y-axis lower limit to {current_y - buffer}')

    # Redraw the canvas to reflect updates.
    fig.canvas.draw()
    return (line_normal, scatter_anomalies_main,
            line_ilof, scatter_ilof,
            line_sklof, scatter_sklof)

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
