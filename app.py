# import gradio as gr
# import tensorflow as tf
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np

# # Load model and tokenizer only once at startup
# model = load_model('model.keras')
# with open('tokenizer.pkl', 'rb') as file:
#     tokenizer = pickle.load(file)

# MAXLEN = 111  # Set this according to your training

# def predict_plagiarism(source_txt, suspect_txt):
#     # Combine both sentences with a space separator to mimic training input
#     combined_text = source_txt + " " + suspect_txt
#     seq = tokenizer.texts_to_sequences([combined_text])
#     pads = pad_sequences(seq, maxlen=MAXLEN)
#     prob = model.predict(pads)[0][0]
#     if prob > 0.7:
#         return f"This Text Has Plagiarism With Similarity Score: {prob:.3f}"
#     else:
#         return "This Text Has No Plagiarism"


# iface = gr.Interface(
#     fn=predict_plagiarism,
#     inputs=[gr.Textbox(lines=2, label="Source Text"), gr.Textbox(lines=2, label="Suspect Text")],
#     outputs=gr.Textbox(label="Result"),
#     title="Plagiarism Detector",
#     description="Enter a source sentence and a suspected plagiarized sentence to check for plagiarism."
# )

# if __name__ == "__main__":
#     iface.launch()

import gradio as gr
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model('model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAXLEN = 111

def predict_plagiarism_single(text):
    # Preprocess text as in training
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=MAXLEN)
    prediction = model.predict(padded_seq)[0][0]
    if prediction > 0.7:
        return f"This Text Has Plagiarism With Similarity Score: {prediction:.3f}"
    else:
        return "This Text Has No Plagiarism"

iface = gr.Interface(
    fn=predict_plagiarism_single,
    inputs=gr.Textbox(lines=4, label="Enter Text To Check"),
    outputs="text",
    title="Plagiarism Detector",
    description="Input text and check if the model detects plagiarism."
)

if __name__ == "__main__":
    iface.launch()
