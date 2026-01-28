import pandas as pd
import os


def get_input(var_name, prompt):
    value = input(prompt + " ('quit' to exit)").strip()

    if value.lower() in {"quit"}:
        raise SystemExit(f"{var_name} entry was cancelled. Exiting program.")

    return value


def enter_rater_id():
    var_name = "Rater ID"
    rater_id = get_input(var_name, "Enter rater ID (first/last initials):")

    while True:
        if rater_id == "":
            rater_id = get_input(var_name, f"{var_name} cannot be empty. Try again: ")
        elif not rater_id.isalpha():
            rater_id = get_input(var_name, f"{var_name} must be your initials (first/last). Try again:")
        elif len(rater_id) != 2:
            rater_id = get_input(var_name, f"{var_name} must be your initials (first/last). Try again:")
        else:
            rater_id = rater_id.upper()
            break

    print(f"Rater ID set to: {rater_id}")
    
    return rater_id


def enter_freq_range():
    var_name = "frequency range"
    selection = get_input(var_name, "Which frequency range are you rating? (A) 20kHz or (B) 40kHz:")

    while True:
        if selection == "":
            selection = get_input(var_name, "Frequency range cannot be empty. Try again: ")
        elif selection.strip().lower() not in ['a', 'b']:
            selection = get_input(var_name, "You can only select option A or B. Try again: ")
        else:
            selection = selection.upper()
            break
    if selection == 'A':
        parent_dir = "25kHz"
        freq_range = (20000, 30000)
    else:
        parent_dir = "40kHz"
        freq_range = (40000, 90000)

    print(f"Frequency range set to: {freq_range}")

    return freq_range, parent_dir


def search_session(base_path, wavfile_name):
    data_directory = os.path.join(base_path, wavfile_name)
    if not os.path.exists(data_directory):
        print(f"Folder for {wavfile_name} not found in USV_DATA directory.")
        print("Run USV detection to create the folder and files.")
        return False, None
    else:
        print(f"Folder for {wavfile_name} found.")
        return True, data_directory


def search_labels(base_path, session_id):
    wavfolder_exists, wavfolder = search_session(base_path, session_id)
    
    if not wavfolder_exists:
        return False
    
    labels_file = os.path.join(wavfolder, f"{session_id}_USV_aud.txt")
    if os.path.exists(labels_file):
        print("Audacity labels found.")
        print(f"Labels file: {session_id}_USV_aud.txt")
        return True
    else:
        print("No Audacity labels found. Check folder contents.")
        return False


def get_trial_starts(base_path, session_id):
    data_directory = os.path.join(base_path, session_id)
    for file in os.listdir(data_directory):
        if file.endswith(".csv"):
            file_parts = file.replace(".csv", "").split("_")
            trial_starts_file = file_parts[-1]
            print("Trial starts file found.")
            return file, trial_starts_file
    else:
        print("No trial starts file found.")
        return None, None


def check_existing_rater(base_path, session_id, rater_id):
    data_directory = os.path.join(base_path, session_id)
    rater_file = os.path.join(data_directory, f"{session_id}_USV_rated_{rater_id}.csv")
    if os.path.exists(rater_file):
        overwrite = input(f"Rater ID {rater_id} has already rated session {session_id}. Overwrite existing data? (y/n): ")
        if overwrite.lower() == 'y':
            print("Existing rater data will be overwritten.")
            return True
        else:
            print("Exiting program to avoid overwriting existing data.")
            return False
    else:
        return True
    

def get_offset(trial_starts_file):
    """
    Get offset time from trial starts file.
    """
    markers = pd.read_csv(trial_starts_file, header=None)
    beeps = markers[markers[0] == 'EVT32'].reset_index(drop=True)[[0, 4]].rename(columns={4: 'Timestamp', 0: 'Event'})
    print(beeps)
    plex_beep = get_input("Plexon Event Time", "Enter plexon beep time in seconds to align trials: ")
    audacity_beep = get_input("Audacity Event Time", "Enter audacity beep time in seconds: ")
    offset = abs(float(plex_beep) - float(audacity_beep))
    return offset

errors = {}
inputs = {}

def get_required_input(name, prompt):
    value = input(prompt).strip()
    if not value:
        errors[name] = f"{name} is required."
    else:
        inputs[name] = value
        errors.pop(name, None)



