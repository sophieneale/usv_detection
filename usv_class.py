import pandas as pd
import numpy as np
import librosa

import os
import time
import json

import usv_library as ul


# USV class for processing different Wav files
class USV:

    # Constructor
    def __init__(self, wavfile, name, trial_starts_file=None, offset=0):
        """
        Initialize USV object with WAV file and parameters.

        Parameters:
        - wav_filename: Path to the WAV file.
        - name: Identifier for the USV object.
        - trial_starts: Optional path to CSV file with trial start times.
        - offset: Time offset to adjust trial start times.

        Instance Variables:
        - wavfile: Path to the WAV file.
        - fs: Sampling frequency of the WAV file.
        - duration: Duration of the WAV file in seconds.
        - name: Identifier for the USV object.
        - pairs: List to store detected USV start and stop times.
        - onset_times: List to store detected onset times.
        - offset_times: List to store detected offset times.
        - start_time: Start time for processing chunks.
        - trial_starts: List of trial start times.
        - trial_duration: Duration of each trial in seconds.
        """

        self.wavfile = wavfile
        self.fs = librosa.get_samplerate(wavfile)
        self.duration = librosa.get_duration(filename=wavfile)
        self.name = name
        self.trial_starts_file = trial_starts_file

        if trial_starts_file != None:
            self.markers = pd.read_csv(self.trial_starts_file, header=None)
            self.beep_offset = offset
            self.lever_press = self.markers[self.markers[0] == 'Lever_press'][4] - self.beep_offset
            self.trial_starts = self.markers[self.markers[0] == 'Trial_start'][4] - self.beep_offset
        else:
            self.trial_starts = [0]

        self.pairs = []
        self.onset_times = []
        self.offset_times = []

    def set_default_params(self, params_file):
        """
        Set default parameters for USV detection using a JSON file.

        Parameters:
        - json: Path to the JSON file containing default parameters.
        Attributes set:
        - freq_range: lower and upper frequency bounds for filtering
        - window_size: minimum window size for onset/offset detection (ms)
        - chunk_duration: duration of chunks to process (s)
        - frame_duration: duration of frames for streaming (s)
        - on_std_multiplier: standard deviation multiplier for onset threshold
        - off_std_multiplier: standard deviation multiplier for offset threshold
        - amp_std_multiplier: standard deviation multiplier for amplitude threshold
        - twentyfive: boolean for 25 kHz call detection
        - (optional) call_duration: length bounds for 25 kHz calls (s)
        - (optional) neighbor_distance: max distance to nearest neighbor for 25 kHz calls (s)
        - has_markers: boolean for whether trial starts / lever presses are provided
        - params_source: source of parameters (json file)
        """
        with open(f"default_params/{params_file}", 'r') as f:
            args = json.load(f)
            for key, value in args.items():
                setattr(self, key, value)

        try:
            _ = self.usv_type
            _ = self.freq_range
            _ = self.window_size
            _ = self.chunk_duration
            _ = self.frame_duration
            _ = self.on_std_multiplier
            _ = self.off_std_multiplier
            _ = self.amp_std_multiplier
            _ = self.min_call_duration
            _ = self.detect_between_trials
            _ = self.twentyfive
            if self.twentyfive:
                _ = self.max_call_duration
                # add optional params check later
            _ = self.trial_starts is not None
            _ = self.params_source 
            print(f"Standard default parameters loaded from default_params/{params_file} for {self.name}.")
        except AttributeError as e:
            raise AttributeError(f"Missing parameter in {json} - {e}")
        
        if self.trial_starts is not None and not hasattr(self, 'trial_duration'):
            raise ValueError("Trial duration must be specified if trial starts are provided and is not equal to the duration of the WAV file.")
            

    # Run to Detect USVs in a large .wav file
    def detect_usv(self, frame_duration, freq_range):
        """
        Detect USVs in the WAV file by processing in chunks.

        Parameters:
        - frame_duration: Duration of each processing frame in seconds.
        - freq_range: Tuple specifying the frequency range for filtering (min_freq, max_freq).
        """
        self.freq_range = freq_range

        stream_starts = [0] if self.detect_between_trials else self.trial_starts
        stream_duration = self.duration if self.detect_between_trials else self.trial_duration

        for start in stream_starts:
        # Create a stream to load RAW data in chunks
            stream = librosa.stream(self.wavfile, 
                                    block_length= 1, 
                                    frame_length=self.fs*frame_duration, 
                                    hop_length=int(self.fs*frame_duration/2),
                                    offset = start,
                                    duration=stream_duration)
            
            # Iterate through stream and operate on each chunk
            self.start_time = start
            i = 0
            for y in stream:
                i += 1
                self.detect_usv_helper(y, self.fs, freq_range, self.start_time)
                # print(f"detected for chunk {i}, start time = {self.start_time}")
                self.start_time += frame_duration/2
        
        # Remove USVs that are less than minimum call duration
        self.pairs = list(filter(lambda pair: abs(pair[1] - pair[0]) > self.min_call_duration, self.pairs))

        # Remove USVs that aren't within max duration bounds for 25 kHz calls
        if self.twentyfive:
            self.pairs = list(filter(lambda pair: abs(pair[1] - pair[0]) <= self.max_call_duration, self.pairs))
            
            # Remove USVs that are not within neighbor distance for 25 kHz calls
            if hasattr(self, 'neighbor_distance'):
                nearest_neighbors = ul.get_nearest_neighbors(np.array(self.pairs))
                if len(nearest_neighbors) != len(self.pairs):
                    raise ValueError("Length of nearest neighbors does not match length of pairs.")
                boolean_filter = np.array(list(map(lambda nn: nn <= self.neighbor_distance, nearest_neighbors)))
                self.pairs = np.array(self.pairs)[boolean_filter].tolist()



    def detect_usv_helper(self, y, fs, freq_range, start_time):
        filtered_usv, envelope, time, downsampled_env, env_derivative = ul.get_env(y, fs, freq_range)
        onsets, onset_times = ul.get_onsets(env_derivative, start_time, self.window_size, self.on_thresh)
        offsets, offset_times = ul.get_offsets(downsampled_env, self.window_size, self.off_thresh, start_time)
        self.onset_times.extend(onset_times)
        self.offset_times.extend(offset_times)
        chunk_pairs, orphans = ul.verify_usv(downsampled_env, onsets, offsets, self.amp_thresh, start_time)
        self.pairs = ul.align(self.pairs, chunk_pairs)


    def run_thresholds(self, freq_range):
        """
        Calculate thresholds for USV detection based on specified frequency range.
        Parameters:
        - freq_range: Tuple specifying the frequency range for filtering (min_freq, max_freq)."""

        # Get thresholds for onsets and offsets from file
        self.on_avg, self.on_std, self.off_avg, self.off_std = (
            ul.get_thresholds(self.fs, self.wavfile, self.trial_starts, 
                          self.trial_duration, self.chunk_duration, 
                          freq_range)
        )

        # Set thresholds based on averages and standard deviations
        self.on_thresh = self.on_avg + self.on_std * self.on_std_multiplier
        self.off_thresh = self.off_avg + self.off_std * self.off_std_multiplier
        self.amp_thresh = self.off_avg + self.off_std * self.amp_std_multiplier


    def add_cols(self):
        """
        Store detected USVs in a DataFrame and save to CSV files.
        Each USV is saved with its start time, stop time, waveform data, label, and duration.
        """

        self.usv_data = pd.DataFrame(columns= ['start', 'stop', 'label', 'duration'])

        for i, (start, stop) in enumerate(self.pairs):
            self.usv_data.loc[i] = [start, stop, f"USV_{i+1}", stop-start]
        
        # Add trial number and before_open columns if trial_starts / lever presses are provided
        if self.trial_starts is not None:
            self.usv_data['trial_num'] = ul.get_trial_nums(self.usv_data['start'], self.trial_starts, self.trial_duration)
            self.usv_data = ul.get_before_open(self.usv_data, self.lever_press, self.trial_starts, self.trial_duration)
            self.usv_data = ul.get_trial_attributes(self)


    def get_session_data(self):
        """
        Export trial information.
        """
        session_data = pd.DataFrame(columns= ['trial_num', 'trial_start', 'open_time', 'open_latency'])
        session_data['trial_num'] = np.arange(1,6)
        session_data['trial_start'] = self.trial_starts.values
        opens, open_latency = ul.get_open(self.lever_press, self.trial_starts, self.trial_duration)
        session_data['open_time'] = session_data['trial_num'].map(opens)
        session_data['open_latency'] = session_data['trial_num'].map(open_latency)

        self.session_data = session_data
 

    def add_embeddings(self):
        """
        Add embedding features to the USV DataFrame.

        Parameters:
        - embeddings: DataFrame containing embedding features for each USV.
        """
        if 'start' not in self.usv_data.columns or 'stop' not in self.usv_data.columns:
            raise ValueError("USV data must contain 'start' and 'stop' columns to compute embeddings.")
        
        for index, row in self.usv_data.iterrows():
            wavfile = self.wavfile
            start = row['start']
            stop = row['stop']
            vec = ul.usv2vec(start, stop, wavfile, fs=None, n_fft=4096, hop_length=1024,
                        n_mfcc=13, n_mels=40,fmin=self.freq_range[0], fmax=self.freq_range[1])
            self.usv_data.at[index, 'rms'] = vec[0]
            self.usv_data.at[index, 'zcr'] = vec[1]
            self.usv_data.at[index, 'centroid'] = vec[2]
            self.usv_data.at[index, 'bandwidth'] = vec[3]
            self.usv_data.at[index, 'flatness'] = vec[4]
            self.usv_data.at[index, 'rolloff'] = vec[5]
            self.usv_data.at[index, 'f0'] = vec[6]
            for i in range(1, 8):
                self.usv_data.at[index, f'contrast{i}'] = vec[6 + i]


    def export_params(self, path):
        """
        Export the parameters used for USV detection to a json file.
        """
        with open(f"{path}/{self.name}_USV.json", 'w') as f:
            params = {
                "wavfile": self.wavfile,
                "name": self.name,
                "fs": self.fs,
                "duration": self.duration,
                "usv_type": self.usv_type,
                "freq_range": self.freq_range,
                "window_size": self.window_size,
                "chunk_duration": self.chunk_duration,
                "frame_duration": self.frame_duration,
                "on_std_multiplier": self.on_std_multiplier,
                "off_std_multiplier": self.off_std_multiplier,
                "amp_std_multiplier": self.amp_std_multiplier,
                "min_call_duration": self.min_call_duration,
                "onset_params": {
                    "on_avg": self.on_avg,
                    "on_std": self.on_std,
                    "on_thresh": self.on_thresh
                },
                "offset_params": {
                    "off_avg": self.off_avg,
                    "off_std": self.off_std,
                    "off_thresh": self.off_thresh
                },
                "amp_thresh": self.amp_thresh,
                "has_markers": self.trial_starts is not None,
                "params_source": self.params_source if hasattr(self, 'params_source') else "user_defined"
            }

            if self.trial_starts is not None:
                params["trial_starts_file"] = self.trial_starts_file.split('/')[-1]
                params["detect_between_trials"] = self.detect_between_trials
                params["trial_starts"] = self.trial_starts.tolist()
                params["trial_duration"] = self.trial_duration
                params["lever_press"] = self.lever_press.tolist()
                params["beep_offset"] = self.beep_offset
            
            if self.twentyfive:
                params["twentyfive"] = self.twentyfive
                params["max_call_duration"] = self.max_call_duration
                if hasattr(self, 'neighbor_distance'): # implement for optional parameters
                    params["neighbor_distance"] = self.neighbor_distance

            json.dump(params, f, indent=4)
        
        print(f"Parameters exported for {self.name}!")


    def get_labels(self, path):
        """
        Export the USV DataFrame to Audacity lables.
        """
        labels = self.usv_data[['start', 'stop', 'label']]
        labels.to_csv(f'{path}/{self.name}_USV_aud.txt', sep='\t', index=False, header=False)

        print(f"Audacity labels exported for {self.name}!")

    
    def store_csvs(self, path):
        """
        Export the USV DataFrame to a CSV file.
        """
        self.usv_data.to_csv(f'{path}/{self.name}_USV.csv', index=False)

        if hasattr(self, 'session_data'):
            self.session_data.to_csv(f'{path}/{self.name}_session_data.csv', index=False)

        print(f"CSV files stored for {self.name}!")

    def save_all(self, path):
        """
        Run all export functions to save parameters, USV data, session data, and labels.
        """
        self.export_params(path)
        self.store_csvs(path)
        self.get_labels(path)


    def validate_usvs(self, path):
        """
        Validate detected USVs through manual rater input.
        """
        if hasattr(self, 'usv_data'):
            if self.twentyfive:
                twentyfives = ul.sort_25khz(self)
            else:
                validated = ul.validate_usvs(f'{path}/{self.name}_USV_data.csv') # fix this


    def run_usv_detection(self, path):
        """
        Run the full USV detection pipeline: threshold calculation, USV detection, and storage.
        Parameters:
        - freq_range: Tuple specifying the frequency range for filtering (min_freq, max_freq).
        """
        freq_range = self.freq_range
        
        self.run_thresholds(freq_range)
        print(f"Thresholds in the {freq_range[0]/1000}-{freq_range[1]/1000} kHz range \
        for {self.name} have been set, beginning USV detection...")
        
        self.detect_usv(self.frame_duration, freq_range)
        print("usv detection complete... storing usvs")

        self.add_cols()
        print(f"Additional columns added for {self.name}!")

        self.add_embeddings()
        print(f"Embeddings added for {self.name}!")

        if self.trial_starts is not None:
            self.get_session_data()
            print(f"Getting session data for {self.name}!")
        
        self.save_all(path)

        print("run_usv_detection complete!")

# === NOT NEEDED FOR NOW === #

    # Sort 25 kHz calls manually and add to session data
    def sort_25kHz_calls(self):
        """
        Run to sort 25 kHz calls and add to the session data.

        Parameters:
        - usv_data: DataFrame containing detected USVs with their attributes.
        - session_data: DataFrame containing session trial information.
        Outputs:
        - Updated session_data DataFrame with 'calls_before' and 'calls_after' columns.
        """
        twentyfives = ul.sort_25khz(self)
        self.usv_data['25_kHz_call'] = self.usv_data['label'].isin(twentyfives).astype(int)

        calls_per_trial = (self.usv_data[self.usv_data['25_kHz_call'] == 1]
                                      .groupby(by=['trial_num', 'before_open'])['label']
                                      .count().reset_index())
        cpt_dict = {t: (calls_per_trial[calls_per_trial['trial_num'] == t].values 
                                if t in calls_per_trial['trial_num'].values else ['None'])
                                for t in np.arange(1, 6)}
        
        self.session_data['calls_before'] = False
        self.session_data['calls_after'] = False

        for t, arr in cpt_dict.items():
            if arr is None or len(arr) == 0:
                continue

            # Convert to a flat list of strings
            values = [str(x) for row in arr for x in row]

            # Check membership
            calls_before = 'True' in values
            calls_after = 'False' in values

            # Assign to all rows with that trial_num
            self.session_data.loc[self.session_data['trial_num'] == t, 'calls_before'] = calls_before
            self.session_data.loc[self.session_data['trial_num'] == t, 'calls_after'] = calls_after      

# === IN PROGRESS === #

    def run_optimization():
        test_file = 'optimizer_test.wav'
        fs = librosa.get_samplerate(test_file)
        trial_starts = [105.232]
        trial_duration = 240
        chunk_duration = 10
        freq_range = (20000, 90000)
        chunk_best = run_thresh()
        frame_best = None

        def run_thresh():
            start = time()
            ul.get_thresholds(fs, test_file, trial_starts, 
                          trial_duration, chunk_duration, 
                          freq_range)
            end = time()
            return end-start
        
        while abs(chunk_best-duration):
            ...
        start = time()
        ul.get_thresholds(fs, test_file, trial_starts, 
                          trial_duration, chunk_duration, 
                          freq_range)
        end = time()

        duration = end-start
        if chunk_best == None:
            chunk_best = duration
    
    def user_set_params(self):
        ...
