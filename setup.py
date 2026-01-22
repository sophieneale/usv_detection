import pandas as pd
import os

def enter_rater_id():
    
    rater_id = input("Enter rater ID: ")

    while True:
        if rater_id == "":
            rater_id = input("Rater ID cannot be empty. Try again: ")
        elif not rater_id.isalpha():
            rater_id = input("Rater ID must contain only letters. Try again:")
        elif len(rater_id) != 2:
            rater_id = input("Rater ID must be your initials (first/last). Try again:")
        else:
            rater_id = rater_id.upper()
            break

    print(f"Rater ID set to: {rater_id}")
    
    return rater_id

def enter_freq_range():
    selection = input("Which frequency range are you rating? (A) 20kHz or (B) 40kHz: ")
    while True:
        if selection == "":
            selection = input("Frequency range cannot be empty. Try again: ")
        elif selection.strip().lower() not in ['a', 'b']:
            selection = input("You can only select option A or B. Try again: ")
        else:
            selection = selection.strip().upper()
            break
    if selection == 'A':
        parent_dir = "25kHz"
        freq_range = (20000, 30000)
    else:
        parent_dir = "40kHz"
        freq_range = (40000, 90000)

    print(f"Frequency range set to: {freq_range}")

    return freq_range, parent_dir

def search_for_wav_files(base_path, wavfile_name):
    data_directory = os.path.join(base_path, wavfile_name)
    if not os.path.exists(data_directory):
        print(f"Folder for {wavfile_name} not found in USV_DATA directory.")
        print("Run USV detection to create the folder and files.")
        return False
    else:
        print(f"Folder for {wavfile_name} found.")
        labels_file = os.path.join(data_directory, f"{wavfile_name}_USV_aud.txt")
        if os.path.exists(labels_file):
            print("Audacity labels found.")
            print(f"File name: {wavfile_name}_USV_aud.txt'")
            return True
        else:
            print("No Audacity labels found.")
            print("Check folder contents.")
            return False
        
def get_offset(trial_starts):
    """
    Get offset time from trial starts file.
    """
    markers = pd.read_csv(trial_starts, header=None)
    beeps = markers[markers[0] == 'EVT32']
    print(beeps)
    plex_beep = input("Enter plexon beep time in seconds to align trials: ")
    audacity_beep = input("Enter audacity beep time in seconds: ")
    offset = abs(float(plex_beep) - float(audacity_beep))
    return offset
