import pandas as pd
import os
import argparse

import usv_library as ul
import usv_class as uc
from usv_class import USV
import setup

def main():

    ...


parser = argparse.ArgumentParser(description='Parser for USV Detection',)
parser.add_argument('--wavfile', type=str, help='Path to WAV file')
parser.add_argument('--name', type=str, help='Name of the session')

