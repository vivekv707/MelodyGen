from flask import Flask, render_template, send_file,request
import random
import collections
import numpy as np
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
import os
import random
from typing import Dict
app = Flask(__name__)



@app.route("/")
def index():
    # render the template with the audio file URL
    return render_template("index.html")

@app.route("/audio")
def get_audio_file():
    # send the audio file as a response
    return send_file("output.mid", mimetype="audio/midi")

@app.route("/loader")
def loader():

    def midi_to_notes(midi_file: str) -> Dict[int, pd.DataFrame]:
        pm = pretty_midi.PrettyMIDI(midi_file)
        instruments_notes = {}

        for instrument in pm.instruments:
            notes = collections.defaultdict(list)
            sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
            prev_start = sorted_notes[0].start

            for note in sorted_notes:
                start = note.start
                end = note.end
                notes['pitch'].append(note.pitch)
                notes['start'].append(start)
                notes['end'].append(end)
                notes['step'].append(start - prev_start)
                notes['duration'].append(end - start)
                notes['velocity'].append(note.velocity)
                prev_start = start

            instruments_notes[instrument.program] = pd.DataFrame({name: np.array(value) for name, value in notes.items()})

        return instruments_notes


    def notes_to_midi_multi(instruments_notes: dict,out_file: str) -> pretty_midi.PrettyMIDI:

        pm = pretty_midi.PrettyMIDI()

        for instrument_name, notes in instruments_notes.items():
            instruments_to_use = ['piano','organ','guitar','bass']
            if instrument_name == 'piano':
                instrument = 0
            elif instrument_name == 'organ':
                instrument = 22
            elif instrument_name == 'guitar':
                instrument = 25
            else:
                instrument = 32
            instrument = pretty_midi.Instrument(instrument)

            prev_start = 0
            for i, note in notes.iterrows():
                if note['velocity']>127:
                    note['velocity'] = 127
                if note['pitch']>127:
                    note['pitch'] = 127
                start = float(prev_start + note['step'])
                end = float(start + note['duration'])
                note = pretty_midi.Note(
                    velocity=int(note['velocity']),
                    pitch=int(note['pitch']),
                    start=start,
                    end=end,
                )
                instrument.notes.append(note)
                prev_start = start

            pm.instruments.append(instrument)

        pm.write(out_file)
        return pm

    def predict_next_note(
    notes: np.ndarray,
    model: tf.keras.Model,
    temperature: float = 1.0) -> tuple[int, float, float]:
        """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""

        assert temperature > 0

        # Add batch dimension
        inputs = tf.expand_dims(notes, 0)

        predictions = model.predict(inputs)
        pitch_logits = predictions['pitch']
        step = predictions['step']
        duration = predictions['duration']
        velocity_logits = predictions['velocity']

        pitch_logits /= temperature
        velocity_logits/=temperature
        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)
        duration = tf.squeeze(duration, axis=-1)
        step = tf.squeeze(step, axis=-1)
        velocity = tf.random.categorical(pitch_logits,num_samples=1)
        velocity = tf.squeeze(velocity,axis=-1)

        # `step` and `duration` values should be non-negative
        step = tf.maximum(0, step)
        duration = tf.maximum(0, duration)

        return int(pitch), float(step), float(duration),int(velocity)

    instruments_notes = {
    'piano': 0,
    'organ': 1,
    'guitar': 2,
    'bass': 3,
    }   
    instruments = ['piano']
    loaded_models = {}

    for instrument in instruments:
        model_dir = os.path.join('models', instrument)
        loaded_model = tf.keras.models.load_model(model_dir,compile=False)
        loaded_models[instrument] = loaded_model

    models = loaded_models
    key_order = ['pitch', 'step', 'duration','velocity']
    seq_length = 25
    vocab_size = 128
    desired_duration = 60  # duration in seconds
    duration = request.args.get('duration', default='Short')
    if duration == 'Long':
        desired_duration = 120
    elif duration == "Medium":
        desired_duration = 60
    else:   
        desired_duration = 30
    temperature = 2.0
    random_sample = random.randint(1, 6)
    sample = f'samples/sample{random_sample}.mid'
    raw_notes = midi_to_notes(sample)
    for notes in raw_notes.values():
        raw_notes = notes
        break
    generated_notes_all = {}
    instruments_to_use = ['piano']
    for instrument in instruments_to_use:

        sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

        # The initial sequence of notes; pitch is normalized similar to training sequences
        input_notes = (sample_notes[:seq_length] / np.array([vocab_size, 1, 1, 128]))

        generated_notes = []
        prev_start = 0
        while prev_start < desired_duration:
            pitch, step, duration, velocity = predict_next_note(input_notes, models[instrument], temperature)
            start = prev_start + step
            end = start + duration
            if start >= desired_duration:
                break
            input_note = (pitch, step, duration, velocity)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))
        generated_notes_all[instrument] = generated_notes
    out_pm = notes_to_midi_multi(generated_notes_all, out_file='output.mid')

    return "Done"

if __name__ == '__main__':
   app.run()