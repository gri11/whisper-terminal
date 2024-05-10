#!/usr/bin/env python3

import mlx_whisper

import sounddevice as sd
import numpy as np
import wave

import os, sys
import argparse


def main(args):
    if args.file:
        print(f"file: {args.file}")
        return 0

    fs = 44100
    duration = 10  # seconds

    print("recording...")
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=1, dtype="float64")
    sd.wait()
    print("recording done")

    filename = "recording.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit encoding
        wf.setframerate(fs)
        wf.writeframes((myrecording * (2**15)).astype(np.int16))

    text = mlx_whisper.transcribe(filename)["text"]

    print(text)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Terminal")
    parser.add_argument("--file", help="audio file to be transcribed")
    args = parser.parse_args()

    main(args)
