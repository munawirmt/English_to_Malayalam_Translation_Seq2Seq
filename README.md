# English to Malayalam Neural Machine Translation (NMT)

A deep learning-based **Sequence-to-Sequence (Seq2Seq)** language translator that converts English sentences into Malayalam. This project is built using **TensorFlow/Keras** and utilizes an **Encoder-Decoder LSTM architecture** with custom inference loops.

---

## Features
- **Encoder-Decoder Architecture:** Uses LSTM networks to capture sentence contexts.
- **Custom Inference Pipeline:** Reuses trained layer weights for token-by-token sentence generation.
- **Tokenization Support:** Handles text preprocessing and sequence padding automatically.
- **Save & Load Ready:** Fully optimized to export trained models and tokenizers for later use.

---

##  Project Architecture

The model follows a classic **Sequence-to-Sequence** pipeline:
1. **Encoder:** Takes the English token sequence and compresses it into internal state vectors (Context Vectors).
2. **Decoder (Training):** Learns to map the context vector along with the target shifted sequence (`<start> ...`) to predict the actual Malayalam sentence (`... <end>`).
3. **Decoder (Inference):** Uses a recursive loop where each predicted word becomes the input for predicting the next word until an `<end>` token is encountered.

---

## Tech Stack
- **Language:** Python 3
- **Framework:** TensorFlow / Keras
- **Libraries:** NumPy, Pickle (for tokenizer serialization)
- **Environment:** Google Colab / Jupyter Notebook

---

## Dataset Structure
The dataset consists of parallel sentence pairs with special markers added to the target language:
- **Encoder Input:** `"I am reading"`
- **Decoder Input:** `"<start> ഞാൻ വായിക്കുന്നു"`
- **Decoder Target:** `"ഞാൻ വായിക്കുന്നു <end>"`

---

## ⚙️ How It Works (Quick Overview)

### 1. Preprocessing & Training
The text is tokenized into word indices, padded to equal length, and fed into the multi-input Keras model.
```python
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=200)
```

### 2. Custom Inference Loop
During prediction, the states are extracted once from the encoder, and a loop iteratively feeds the predicted word back into the decoder:
```python
while not stop_condition:
    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
    # Extract next token and transition states...
```

---

##  File Outputs
After completing training, the script exports the following assets:
- `en_ml_translator.keras` - The unified training model.
- `encoder_model.keras` / `decoder_model.keras` - Separated inference models.
- `en_tokenizer.pickle` / `ml_tokenizer.pickle` - Pickled tokenizers to decode indices back into text.

---

## Future Enhancements
- Expand the parallel dataset to thousands of complex sentences.
- Implement **Attention Mechanism** (Bahdanau/Luong) to improve longer sentence translations.
- Upgrade to a **Transformer-based architecture** for state-of-the-art results.
