import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.io import wavfile
import librosa
import librosa.display
import pandas as pd
import os
import soundfile as sf
from pathlib import Path

# Butterworth bandpass filter and Hilbert transform to get envelope
def butterworth_hilbert_filter(data, fs, band):
    """
    Apply a Butterworth bandpass filter and Hilbert transform to the input data.
    Parameters:
    - data: 1D numpy array of the signal to be filtered.
    - fs: Sampling frequency of the signal.
    - band: Tuple (low_freq, high_freq) specifying the frequency band for the filter.
    Returns:
    - filtered: The bandpass filtered signal.
    - amplitude_envelope: The amplitude envelope of the filtered signal.
    """
    # Ensure data is a numpy array
    data = np.asarray(data)

    # Bandpass filter: Butterworth (frequencies in the pass band are minimally affected) 
    nyq = 0.5 * fs
    low, high = band[0] / nyq, band[1] / nyq
    b, a = signal.butter(3, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)

    # Hilbert transform to get amplitude envelope
    analytic_signal = signal.hilbert(filtered)
    amplitude_envelope = np.abs(analytic_signal)

    return filtered, amplitude_envelope


# Process the USV signal to get filtered signal, envelope, smoothed envelope, downsampled envelope, and its derivative
def get_env(y, fs, freq_range):
    """
    Process the USV signal to extract the filtered signal, envelope, smoothed envelope,
    downsampled envelope, and its derivative.

    Parameters:
    - y: 1D numpy array of the audio signal.
    - fs: Sampling frequency of the audio signal.
    - freq_range: Tuple (low_freq, high_freq) specifying the frequency band for filtering.

    Returns:
    - filtered_usv: The bandpass filtered signal.
    - envelope: The amplitude envelope of the filtered signal.
    - time: Time array corresponding to the original signal.
    - downsampled_env: The smoothed and downsampled envelope (1 kHz).
    - env_derivative: Derivative of the downsampled envelope.
    """

    # 1. Get parameters
    y = (y * 32768).astype('int16')
    fs = fs

    # 2. Filter in USV band and get envelope
    filtered_usv, envelope = butterworth_hilbert_filter(y, fs, freq_range)
    time = np.arange(len(y)) / fs

    # 3. Smooth envelope with lowpass (500 Hz)
    b_lp, a_lp = signal.butter(3, 500 / (fs / 2), btype='low')
    smoothed_env = signal.filtfilt(b_lp, a_lp, envelope)

    # 4. Downsample to 1 kHz
    target_fs = 1000
    downsample_factor = int(fs / target_fs)
    downsampled_env = smoothed_env[::downsample_factor]

    env_derivative = np.gradient(downsampled_env)

    # Return results
    return (filtered_usv, envelope, time, downsampled_env, env_derivative)


# Calculate threshold statistics for onset/offset of a chunk
def get_thresholds_helper(downsampled_env, env_derivative):
    """
    Calculate onset and offset thresholds based on the envelope derivative and amplitude.

    Parameters:
    - env_derivative: Derivative of the downsampled envelope.
    - downsampled_env: Downsampled amplitude envelope.

    Returns:
    - env_dx_avg: Mean of the envelope derivative (onset threshold).
    - env_dx_std: Standard deviation of the envelope derivative.
    - mean_amp: Mean of the downsampled envelope (offset threshold).
    - amp_std: Standard deviation of the downsampled envelope.
    """

    # Onset: rate of change of noise amplitude
    env_dx_avg, env_dx_std = env_derivative.mean(), np.std(env_derivative)

    # Offset: mean amplitude of noise
    amp_avg, amp_std = downsampled_env.mean(), np.std(downsampled_env)

    return env_dx_avg, env_dx_std, amp_avg, amp_std
    

# Calculate thresholds for entire audio file by processing in chunks
def get_thresholds(fs, wavfile, trial_starts, trial_duration, chunk_duration, freq_range):
    """
    Calculate statistics for onset and offset thresholds for the whole audio file
    by proccessing it in chunks.

    Parameters:
    - fs: Sampling frequency of the audio file.
    - wavfile: Path to the audio file.
    - trial_starts: List of start times (in seconds) for each chunk to be tested.
    - chunk_duration: Duration (in seconds) of each chunk to be tested.
    - freq_range: Tuple (low_freq, high_freq) specifying the frequency band for filtering

    Returns:
    - on_avg: Average onset threshold.
    - on_std: Standard deviation of onset threshold.
    - off_avg: Average offset threshold.
    - off_std: Standard deviation of offset threshold.
    """

    # Initialize lists to store threshold statistics from each chunk
    env_dx_avg_list = []
    env_dx_std_list = []
    amp_avg_list = []
    amp_std_list = []

    # Store lengths for weighted averaging
    lengths = []

    # Loop through each start time and process each trial in chunks
    for start in trial_starts:
        stream = librosa.stream(wavfile, 
                                block_length= 1, 
                                frame_length= int(fs*chunk_duration), 
                                hop_length= int(fs*chunk_duration),
                                offset= start,
                                duration= trial_duration)
        for y in stream:    # y is a chunk of audio data
            # Process chunk to get downsampled envelope and its derivative
            _, _, _, downsampled_env, env_derivative = get_env(y, fs, freq_range)

            # Calculate and store threshold statistics for the chunk
            env_dx_avg, env_dx_std, amp_avg, amp_std = get_thresholds_helper(downsampled_env, env_derivative)
            env_dx_avg_list.append(env_dx_avg)
            env_dx_std_list.append(env_dx_std)
            amp_avg_list.append(amp_avg)
            amp_std_list.append(amp_std)

            lengths.append(len(downsampled_env))

    # Convert to numpy arrays for vectorized weighting
    lengths = np.array(lengths)
    weights = lengths / lengths.sum()

    # Average chunk statistics to summarize thresholds for the whole file
    on_avg = np.average(env_dx_avg_list, weights=weights)
    on_std = np.average(env_dx_std_list, weights=weights)
    off_avg = np.average(amp_avg_list, weights=weights)
    off_std = np.average(amp_std_list, weights=weights)

    return on_avg, on_std, off_avg, off_std


# Detects start times using derivative
def get_onsets(env_derivative, start_time, window_size, on_thresh):
    """
    Detect onsets in the USV signal based on the derivative of the downsampled envelope.

    Parameters:
    - env_derivative: Derivative of the downsampled envelope.
    - start_time: Start time of the chunk in the original audio file.
    - window_size: Number of consecutive frames (in ms) of low activity required before an onset.
    - on_thresh: Derivative threshold to consider as an onset.

    Returns:
    - onsets: Binary array indicating detected onsets.
    - onset_times: Times (in seconds) of detected onsets in the original audio file
    """
    
    # Pre-process: get int boolean array and pad with 0s
    above_thresh = (env_derivative > on_thresh).astype(int)
    padded_array = np.pad(above_thresh, 
                            pad_width=(window_size, 0), 
                            mode='constant', constant_values=0)

    # Filter array to only include True values which have window_size Falses before
    onsets = [1 if val == 1 and sum(
        padded_array[i-window_size:i]) == 0 
        else 0 for i, val in enumerate(padded_array)][window_size:]
    
    # Store times of onsets in original audio file
    onset_times = (np.where(np.array(onsets) == 1)[0] / 1000) + start_time

    return onsets, onset_times


# Detects end times using the mean of noise from wav file, plus std standard deviations
def get_offsets(downsampled_env, window_size, off_thresh, start_time):
    """
    Detect offsets in the USV signal based on the amplitude of the downsampled envelope
    and a window of low activity.

    Parameters:
    - usv: A USV object with attributes 'downsampled_env' and 'start_time'.
    - window_size: Number of consecutive frames (in ms) of low activity required after an offset.
    - offset_thresh: Amplitude threshold to consider as an offset.
    - start_time: Start time of the chunk in the original audio file.

    Returns:
    - offsets: Binary array indicating detected offsets.
    - offset_times: Times (in seconds) of detected offsets in the original audio file.
    """

    # Pre-process: get int boolean array and pad with 0s
    above_thresh = (downsampled_env > off_thresh).astype(int)
    padded_array = np.pad(above_thresh, 
                            pad_width=(0, window_size), mode='constant', 
                            constant_values=0)

    # Filter array to only include True values which have window_size Falses after
    offsets = [1 if val == 1 and sum(
        padded_array[i+1:i+window_size]) == 0 
        else 0 for i, val in enumerate(padded_array)][:-window_size]
    
    # Store times of offsets in original audio file
    offset_times = (np.where(np.array(offsets) == 1)[0] / 1000) + start_time

    return offsets, offset_times


# Make sure onsets are followed by offsets and are indicating USVs
def verify_usv(downsampled_env, onsets, offsets, amp_thresh, start_time):
    """
    Verify and pair onsets and offsets to identify valid USV events.
    Parameters:
    - usv: A USV object with attributes 'onsets', 'offsets', 'downsampled_env', and 'start_time'.
    - amp_thresh: Minimum amplitude threshold to consider a valid USV event.
    Returns:
    - chunk_pairs: List of tuples representing start and stop times of detected USV events.
    - orphans: List of unpaired onsets or offsets.
    """

    # Ensure onsets and offsets arrays are the same length
    if len(onsets) != len(offsets):
        raise IndexError('Arrays for starts and ends are not the same length')

    # 1 = onset, -1 = offset, 0 = neither
    usv_bounds = enumerate(np.subtract(onsets, offsets))    # maybe use output of np.where
    usv_bounds = [i for i in usv_bounds if i[1] != 0]

    # Pair onsets and offsets
    pairs = []
    i = 0
    while i < len(usv_bounds):
        if usv_bounds[i][1] == 1:
            # Look ahead for the next -1
            for j in range(i + 1, len(usv_bounds)):
                if usv_bounds[j][1] == -1:
                    pairs.append((usv_bounds[i][0], usv_bounds[j][0]))
                    break  # Stop at the first -1 for this 1
        i += 1

    i = 0
    while i < len(pairs):  # Stop at second-to-last index to avoid IndexError
        # If two pairs share the same end time
        if i + 1 < len(pairs) and pairs[i][1] == pairs[i + 1][1]:
            max_amp = max(downsampled_env[pairs[i][0]: pairs[i + 1][0]])
            # Verify max amplitude between start times exceeds amplitude threshold
            if max_amp < amp_thresh:
                del pairs[i]  # Remove current, do not increment i
            else:
                del pairs[i + 1]  # Remove next, do not increment i
            continue  # Check the new pair at the same index

        # If max amplitude between start and end does not exceed amplitude threshold
        elif max(downsampled_env[pairs[i][0] : pairs[i][1]]) < amp_thresh:
            del pairs[i]  # Remove current, do not increment i
            continue  # Check the new pair at the same index

        i += 1

    # Store paired and unpaired values separately
    orphans = (np.array([val for val in usv_bounds 
                    if val[0] not in [item for tuple in pairs for item in tuple]])
                    / 1000 + start_time)
    chunk_pairs = list(map(
        lambda x: (x[0]/1000 + start_time, x[1]/1000 + start_time), pairs))
    
    return chunk_pairs, orphans


# Systematically align old pairs with new pairs
def align(old_pairs, new_pairs):
    """
    Align and merge new USV pairs with existing pairs in the USV object.
    - If a new pair starts within an old pair and ends after it, extend the old pair's end time.
    - If a new pair starts exactly when an old pair starts, replace the old pair with the new pair.
    - If a new pair ends exactly when an old pair ends, skip the new pair.
    - If a new pair does not overlap with any old pair, append it to the list of old pairs.
    Parameters:
    - old_pairs: List of existing USV pairs (tuples of start and stop times).
    - new_pairs: List of new USV pairs (tuples of start and stop times).
    Returns:
    - Merged list of USV pairs."""

    # Handle edge cases
    if len(new_pairs) == 0:
        return old_pairs
    elif len(old_pairs) == 0:
        return new_pairs

    # Find index in old_pairs to start checking from
    min_val = new_pairs[0][0]
    start_index = next((i for i, (start, end) in enumerate(old_pairs) if start >= min_val or end >= min_val), 0)

    # If no such index exists, start from the end of old_pairs
    if start_index == 0:
       start_index = len(old_pairs)

    # Iterate through new pairs and align with old pairs
    iterator = iter(new_pairs)
    old_index = start_index

    for pair in iterator:
        try:
            old_pair = old_pairs[old_index]
            if old_pair[1] == pair[1]:
                old_index += 1                 
                continue
            elif old_pair[0] == pair[0]:
                old_pairs[old_index] = pair
            elif pair[0] < old_pair[1] and pair[0] > old_pair[0]:
                old_pairs[old_index] = (old_pair[0], pair[1])
            else:
                continue
            old_index += 1
        except:
            old_pairs.append(pair)
            old_index += 1

    return old_pairs


# Calculate nearest neighbor distances for each USV pair
def get_nearest_neighbors(usv_pairs):
    """
    Calculate the nearest neighbor distances for each USV pair.

    Parameters:
    - usv_pairs: List of tuples, in chronological order, representing start and stop times of detected USV events.

    Returns:
    - List of nearest neighbor distances for each USV pair.
    """
    if len(usv_pairs) < 2:
        return [float('inf')] * len(usv_pairs)

    starts = usv_pairs[1:,0]
    ends = usv_pairs[0:-1,1]

    distances = starts - ends
    nearest_neighbors = [distances[0]] + list(map(lambda dists: min(dists), zip(distances[:-1], distances[1:]))) + [distances[-1]]

    return nearest_neighbors


### Trial-related Methods ###

# Assign trial numbers to each USV based on start times and trial intervals
def get_trial_nums(start_times, trial_starts, trial_duration):
    start_times = np.array(start_times)
    trial_starts = np.array(trial_starts)
    trial_ends = np.array(trial_starts) + trial_duration

    # Find indices of start_times that fall at the bounds of each trial
    left = np.searchsorted(start_times, trial_starts, side='left')
    right = np.searchsorted(start_times, trial_ends, side='left')

    trial_nums = np.full(len(start_times), None, dtype=object)

    # Assign vectorized per-trial slices
    for i, (l, r) in enumerate(zip(left, right)):
        trial_nums[l:r] = i + 1

    return trial_nums.tolist()


# Calculate lever press (open) time and latency for each trial
def get_open(lever_press, trial_starts, trial_duration):
    trial_starts = np.asarray(trial_starts)
    lever_press = np.asarray(lever_press)
    trial_ends = trial_starts + trial_duration

    # Find candidate trial for each open_time
    idx = np.searchsorted(trial_starts, lever_press, side='right') - 1
    valid = (idx >= 0) & (lever_press < trial_ends[idx])

    # Initialize dictionary with None values
    opens = {i + 1: None for i in range(len(trial_starts))}
    open_latency = {i + 1: None for i in range(len(trial_starts))}

    filled = np.full(len(trial_starts), False)

    # Assign first open time per trial
    for trial_index, open_t in zip(idx[valid], lever_press[valid]):
        if not filled[trial_index]:
            trial_num = trial_index + 1
            opens[trial_num] = open_t
            open_latency[trial_num] = open_t - trial_starts[trial_index]
            filled[trial_index] = True

    return opens, open_latency


# Add before_open column to usv_data DataFrame
def get_before_open(usv_data, lever_press, trial_starts, trial_duration):
    """
    Add 'before_open' column to the usv_data DataFrame within the USV object.
    This column indicates whether each USV occurred before the lever press (open) time of its trial.
    
    Parameters:
    - usv_data: DataFrame containing USV data with 'trial_num' and 'start' columns.
    - lever_press: List of lever press times for each trial.
    - trial_starts: List of trial start times.
    - trial_duration: Duration of each trial.
    Returns:
    - usv_data: Updated DataFrame with 'before_open' column added.
        - 'before_open': 'True' if USV occurred before open time, 'False' if after, 'None' if no open time.
    """
    opens, open_latency = get_open(lever_press, trial_starts, trial_duration)

    usv_data['open_time'] = usv_data['trial_num'].map(opens)
    usv_data['open_latency'] = usv_data['trial_num'].map(open_latency)

    usv_data['before_open'] = (
    np.where(usv_data['open_time'].isna(), 
             'None', 
             np.where(usv_data['start'] < usv_data['open_time'], 'True', 'False'))
    )

    usv_data = usv_data.drop(columns=['open_time'])

    return usv_data


# Add trial-related attributes to usv_data DataFrame
def get_trial_attributes(usv):
    """
    Add trial-related attributes to the usv_data DataFrame within the USV object.
    
    Parameters:
    - usv: A USV object with attributes 'trial_starts' and 'usv_data'.
    Returns:
    - usv_data: Updated DataFrame with trial-related columns added.
        - 'trial_start': Start time of the trial for each USV.
        - 'lever_press': Lever press time for each USV.
        - 'prox_to_trial_start': Time difference between USV start and trial start.
        - 'prox_to_lever_press': Time difference between USV start and lever press time
    """
    trial_starts = usv.trial_starts.values
    trial_starts_dict = {i+1: val for i, val in enumerate(trial_starts)}
    usv_data = usv.usv_data

    usv_data['trial_start'] = usv_data['trial_num'].map(trial_starts_dict)
    usv_data['lever_press'] = (usv_data['trial_start'] + usv_data['open_latency']).where(usv_data['open_latency'].notna(), None)
    usv_data['prox_to_trial_start'] = usv_data['start'] - usv_data['trial_start']
    usv_data['prox_to_lever_press'] = (usv_data['start'] - usv_data['lever_press']).where(usv_data['lever_press'].notna(), None)

    return usv_data


# Manual sorting of calls from existing CSV
def validate_usv(usv_csv, parent_dir):
    if parent_dir == "25kHz":
        usv_type = "25kHz call"
    else:
        usv_type = "USV"
    call_lst = []

    data = pd.read_csv(usv_csv)
    labels = data['label'].tolist()
    i = 0

    while i < len(labels):
        label = labels[i]
        answer = input(f"Is {label} a {usv_type}? (1 = yes, 2 = no, skip = jump, exit = stop): ").strip()

        # YES - keep it
        if answer == '1':
            call_lst.append(label)
            i += 1

        # NO - just move on
        elif answer == '2':
            i += 1

        # EXIT early
        elif answer.lower() == 'exit':
            print("Exiting the sorting process.")
            return call_lst

        # SKIP ahead
        elif answer.lower() == 'skip':
            try:
                skip_to = int(input(f"Skip to which index? (current = {i}, max = {len(labels)-1}): ").strip())
                if skip_to <= i:
                    print("Skip index must be GREATER than the current index.")
                    continue
                i = skip_to
            except ValueError:
                print("Invalid index. Must be an integer.")

        else:
            print("Invalid input. Enter 1, 2, 'skip', or 'exit'.")

    return call_lst


# Extract trial start times from params.txt file
def extract_trial_starts(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    
    trial_starts = []
    grab = False

    for line in lines:
        # Detect the start of the trial starts section
        if "Trial Starts:" in line:
            grab = True

        if grab:
            stripped = line.strip()

            # Stop at pandas Series name line
            if stripped.startswith("Name:"):
                break
            
            # Skip empty lines
            if stripped:
                parts = stripped.split()
                # Ensure there are at least two columns: index + value
                if len(parts) >= 2:
                    trial_starts.append(float(parts[-1]))

    return {i+1: val for i, val in enumerate(trial_starts)}


# Add trial-related columns to existing usv_data CSV
def add_cols_to_existing(directory):
    original_csv = None
    trial_starts = None
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        if file.endswith(".csv"):
            original_csv = pd.read_csv(filepath)
        elif file.endswith("params.txt"):
            trial_starts = extract_trial_starts(filepath)

    original_csv['trial_start'] = original_csv['trial_num'].map(trial_starts)
    original_csv['lever_press'] = (original_csv['trial_start'] + original_csv['open_latency']).where(original_csv['open_latency'].notna(), None)
    original_csv['prox_to_trial_start'] = original_csv['start'] - original_csv['trial_start']
    original_csv['prox_to_lever_press'] = (original_csv['start'] - original_csv['lever_press']).where(original_csv['lever_press'].notna(), None)

    return original_csv


# ==== NOT IN USE ==== #


# Manual sorting of 25 kHz calls
def sort_25khz(usv):
    twentyfive_lst = []

    labels = list(usv.usv_data['label'])
    i = 0

    while i < len(labels):
        label = labels[i]
        answer = input(f"Is {label} a 25 kHz call? (1 = yes, 2 = no, skip = jump, exit = stop): ").strip()

        # YES - keep it
        if answer == '1':
            twentyfive_lst.append(label)
            i += 1

        # NO - just move on
        elif answer == '2':
            i += 1

        # EXIT early
        elif answer.lower() == 'exit':
            print("Exiting the sorting process.")
            return twentyfive_lst

        # SKIP ahead
        elif answer.lower() == 'skip':
            try:
                skip_to = int(input(f"Skip to which index? (current = {i}, max = {len(labels)-1}): ").strip())
                if skip_to <= i:
                    print("Skip index must be GREATER than the current index.")
                    continue
                i = skip_to
            except ValueError:
                print("Invalid index. Must be an integer.")

        else:
            print("Invalid input. Enter 1, 2, 'skip', or 'exit'.")

    return twentyfive_lst


# ===== IN PROGRESS ===== #

# Extract lever press times from params.txt file
def extract_lever_press(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        
    lever_press = []
    grab = False

    # for line in lines:
    #     # Detect the start of the trial starts section
    #     if "Trial Starts:" in line:
    #         grab = True

    #     if grab:
    #         stripped = line.strip()

    #         # Stop at pandas Series name line
    #         if stripped.startswith("Name:"):
    #             break
            
    #         # Skip empty lines
    #         if stripped:
    #             parts = stripped.split()
    #             # Ensure there are at least two columns: index + value
    #             if len(parts) >= 2:
    #                 trial_starts.append(float(parts[-1]))

    # return {i+1: val for i, val in enumerate(trial_starts)}

def usv2vec(start, stop, wavfile, fs=None, n_fft=4096, hop_length=1024, n_mfcc=13, n_mels=10, fmin=20000, fmax=30000):
    y, sr = librosa.load(wavfile, sr=fs, offset=start-0.005, duration=(stop - start)+0.005)

    cutoff = 50000
    nyq = sr / 2
    normalized_cutoff = cutoff / nyq
    order = 6

    sos = signal.butter(order, normalized_cutoff, btype='low', output='sos')
    y_filtered = signal.sosfiltfilt(sos, y)
    y = y_filtered

    # Error for bad filepath
    if not Path(wavfile).is_file():
        raise FileNotFoundError(f"Not a file: {wavfile}")
    
    # Load audio segment -- one extracted USV with 5ms padding on either side
    # signal, sr = librosa.load(wavfile, sr=fs, offset=start-0.005, duration=(stop - start)+0.005)

    # === Time domain features (2 total) === #
    
    # Avg amplitude (loudness)
    rms = np.mean(librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length))

    # Oscillation rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)) 
    
    # === Spectral shape features (11 total) === #

    # Mean freq
    centroid  = np.mean(librosa.feature.spectral_centroid (y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
    
    # Bandwidth (mean freq spread)
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
    
    # y to noise ratio across freq bins
    flatness = np.mean(librosa.feature.spectral_flatness (y=y, n_fft=n_fft, hop_length=hop_length))
    
    # Mean freq below which falls 85% of energy -- change rolloff percent by param
    rolloff = np.mean(librosa.feature.spectral_rolloff (y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
    
    # Mean contrast between freq bins (default of 7) -- to change n_bands=x
    ## n_fft=x to adjust freq resolution, n_bands=x to adjust num of features
    contrast  = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                          n_bands=6), axis=1)
    
    # === Pitch features (1 total) === #

    # Get fundamental freq at each frame then average across frames
    f0_track = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr,frame_length=n_fft, hop_length=hop_length)
    f0 = np.nanmean(f0_track)
    if np.isnan(f0): # If all frames are silent: set pitch to 0
        f0 = 0.0
    
    # Cepstral (MFCC)  features (13 total, set by param)
    #measures spectral shape of PSD
    #each coeeff describes specific qualities of power over frequencies
    #c0 totalenergy,c1,c2 =slope, c3-c6=curvature
    #Uses magic to get selected num of coefficients
    #using  n_mfcc=13 because it is standard
    # mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,n_fft=n_fft, hop_length=hop_length), axis=1)
    
    # Mel spectrogram num of mels sets num of features (default 10)
    mel_power = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                               n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = np.mean(mel_power, axis=1)
    
    return np.concatenate((
        np.array([rms, zcr, centroid, bandwidth, flatness, rolloff, f0], dtype=float),
        contrast.astype(float)))

def export_to_wav(usv_csv, usv_list, subdir, wavfile):
    """
    Export detected USV events to individual WAV files based on the provided CSV file.

    Parameters:
    - usv_csv: Path to the CSV file containing start and stop times of USV events.
    - usv_list: List of USV objects, each with attributes 'wavfile' and 'name'.
    - dir_name: Directory where the exported WAV files will be saved.
    """
    try:
        os.mkdir(f"exported_usvs")
    except FileExistsError:
        pass

    export_dir = os.path.join("exported_usvs", subdir)

    try:
        os.mkdir(export_dir)
    except FileExistsError:
        pass

    df = pd.read_csv(usv_csv)
    usv_names = ['USV_' + str(num) for num in usv_list]
    raw_data = df.loc["label" in usv_names]

    for i, row in raw_data.iterrows():
        y, fs = librosa.load(wavfile, offset= row['start'] - 0.01, duration= row['stop'] - row['start'] + 0.01, sr= None)
        sf.write(f"{row['label']}.wav", fs, y)
    
    print('program complete')

def bang_detector():
    ...

# == Plotting Methods == #

# Plot spectrogram of wav with amplitude envelope
def plot_spectrogram(self):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[3, 1])

    # - Spectrogram -
    D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
    img = librosa.display.specshow(D, sr=self.fs, x_axis='time', y_axis='hz', ax=ax1)
    ax1.set_ylim(self.freq_range[0], self.freq_range[1]+10000)
    ax1.set_title("Spectrogram")
    cax = fig.add_axes([0.92, 0.43, 0.02, 0.45]) 
    fig.colorbar(img, ax=ax1, cax=cax, format='%+2.0f dB')
    ax1.grid(False)

    # - Amplitude Envelope -
    time = np.arange(len(self.downsampled_env)) / 1000
    ax2.plot(time, self.downsampled_env, label='Envelope', color='gray')

    ax2.set_ylabel('Amplitude')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Amplitude Envelope')
    ax2.legend()
    ax2.grid()

    plt.subplots_adjust(right= 0.90)

    # Save figure
    plt.savefig(f"{self.name}_spectrogram.png")

# Plot spectrogram with amplitude envelope and detected onsets
def plot_usv(self, region, lines='on'):
    region_start, region_end = region
    y, fs = librosa.load(self.wavfile, offset= region_start, duration= region_end-region_start, sr= None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 6), sharex=True, height_ratios=[3, 1])

    # - Spectrogram -
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='hz', ax=ax1)
    ax1.set_ylim(self.freq_range[0], self.freq_range[1]+10000)
    ax1.set_title("Spectrogram")
    cax = fig.add_axes([0.92, 0.43, 0.02, 0.45]) 
    fig.colorbar(img, ax=ax1, cax=cax, format='%+2.0f dB')
    ax1.grid(False)

    # Overlay detected event regions
    for i, (start, end) in enumerate(self.pairs):
        if end < region_start:
            continue
        if start < region_start:
            start = region_start
        if end > region_end:
            end = region_end
        if start > region_end:
            break
        start = start - region_start
        end = end - region_start
        ax1.axvspan(start, end, color='cyan', alpha=0.3, label='Detected Event' if i == 0 else "")

    # - Amplitude Envelope -
    self.get_env(self.freq_range)  # Ensure envelope is calculated for the current region
    time = np.arange(len(self.downsampled_env)) / 1000
    ax2.plot(time, self.downsampled_env, label='Envelope', color='gray')

    ax2.set_ylabel('Amplitude')
    ax2.set_xlabel('Time (s)')
    ax2.set_xlim(region_start, region_end)
    ax2.set_title('Amplitude Envelope')
    ax2.legend()
    ax2.grid()

    plt.subplots_adjust(right= 0.90)

    if lines == 'on':
        # Add onset lines (from get_onsets())
        for time in self.onsets:
            if time < region_start:
                continue
            if time > region_end:
                break
            time = time - region_start
            ax1.axvline(x = time, color = 'b')
            ax2.axvline(x = time, color = 'b')

        # offsets to test code
        for time in self.offsets:
            if time < region_start:
                continue
            if time > region_end:
                break
            time = time - region_start
            ax1.axvline(x = time, color = 'r')
            ax2.axvline(x = time, color = 'r')

    # Save figure
    plt.savefig(f"{self.name}_spectrogram.png")

