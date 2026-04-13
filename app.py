import streamlit as st
import torch
import spacy
import pickle
import os
from model_arch import Transformer
from deep_translator import GoogleTranslator

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Neural Machine Translator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
        border-radius: 10px;
    }
    .stButton > button {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 20px;
        padding: 10px 30px;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #0072ff, #00c6ff);
        scale: 1.05;
        transition: 0.3s;
    }
    .result-card {
        background-color: #1e1e26;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #3e3e4a;
        margin-bottom: 20px;
    }
    .model-name {
        color: #00c6ff;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .translation-text {
        font-size: 1.1rem;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD NLP MODELS ---
@st.cache_resource
def load_nlp():
    # Streamlit Cloud will have these installed via requirements.txt
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("❌ Spacy model 'en_core_web_sm' not found. Ensure it is in requirements.txt.")
        return None

en_nlp = load_nlp()

# --- LOAD METADATA ---
@st.cache_resource
def load_metadata():
    metadata_path = 'essential/transformer_metadata.pkl'
    if not os.path.exists(metadata_path):
        st.error(f"❌ Metadata file not found at {metadata_path}")
        return None
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)

metadata = load_metadata()

# --- MODEL LOADING ---
@st.cache_resource
def initialize_model(metadata, device):
    v_size = len(metadata['de_vocab']) # Following notebook's v_size = len(de_vocab)
    d_k = metadata['d_k']
    heads = metadata['heads']
    head_dim = metadata['head_dim']
    seq_len = metadata['max_length']
    
    from model_arch import (
        inputEmbedding, positionalEncoding, InitialLayer, FeedForward,
        MultiHeadAttention, CrossAttention, OutputLayer, EncoderBlock,
        DecoderBlock, Encoder, Decoder, Transformer
    )
    
    # 1. Basic Components
    emb = inputEmbedding(v_size, d_k)
    pos = positionalEncoding(seq_len, d_k)
    it = InitialLayer(emb, pos)
    fnn = FeedForward(d_k)
    att = MultiHeadAttention(d_k, heads, head_dim, device)
    cr = CrossAttention(d_k, heads, head_dim)
    out_layer = OutputLayer(d_k, v_size)

    # 2. Build Stacks (Shared weights as per notebook)
    enc_blocks = torch.nn.ModuleList([EncoderBlock(att, fnn, d_k) for _ in range(6)])
    dec_blocks = torch.nn.ModuleList([DecoderBlock(att, fnn, cr, d_k) for _ in range(6)])

    encoder_stack = Encoder(enc_blocks, d_k)
    decoder_stack = Decoder(dec_blocks, d_k)

    # 3. Initialize Transformer
    model = Transformer(encoder_stack, decoder_stack, it, out_layer, d_k).to(device)
    return model

# --- TRANSLATION FUNCTION ---
def translate_sentence(sentence, model, en_vocab, de_vocab, de_itos, device, max_len=128):
    model.eval()
    
    # 1. Tokenize and convert to indices
    tokens = [token.text.lower() for token in en_nlp.tokenizer(sentence)]
    tokens = ['<sos>'] + tokens + ['<eos>']
    
    # Handle unknown tokens
    unk_idx = en_vocab.get('<unk>', 0)
    src_indices = [en_vocab.get(token, unk_idx) for token in tokens]
    
    # 2. Convert to Tensor and move to device [Batch, Seq]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    # 3. Start the decoder with just <sos>
    sos_idx = de_vocab['<sos>']
    eos_idx = de_vocab['<eos>']
    trg_indices = [sos_idx]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
        
        # Take the last token predicted
        preds = output.argmax(2)[:, -1].item()
        trg_indices.append(preds)
        
        # Stop if model predicts <eos>
        if preds == eos_idx:
            break
            
    # Convert indices back to words (skipping <sos> and <eos>)
    translated_tokens = [de_itos[idx] for idx in trg_indices if idx not in [sos_idx, eos_idx]]
    return " ".join(translated_tokens)

# --- APP LAYOUT ---
st.title("🚀 Neural Machine Translation Comparison")
st.markdown("Compare your custom Transformer models with industry standards.")

if metadata:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sidebar for Model Selection
    st.sidebar.header("📁 Model Configuration")
    model_files = [f for f in os.listdir('models') if f.endswith('.pt')]
    
    if not model_files:
        st.sidebar.warning("No .pt models found in 'models/' folder.")
        selected_models = []
    else:
        selected_models = st.sidebar.multiselect(
            "Select Models to compare:",
            model_files,
            default=model_files[:1] if model_files else []
        )

    # Main Input Area
    input_sentence = st.text_area("Enter English Sentence:", placeholder="Type something to translate...", height=100)
    
    if st.button("Translate ✨"):
        if not input_sentence.strip():
            st.warning("Please enter a sentence.")
        else:
            cols = st.columns(2)
            
            # --- CUSTOM MODELS ---
            with cols[0]:
                st.subheader("🛠️ Your Custom Models")
                if not selected_models:
                    st.info("Select models from the sidebar to see results.")
                else:
                    # Model initialization (cached)
                    base_model = initialize_model(metadata, device)
                    
                    for model_name in selected_models:
                        with st.spinner(f"Translating with {model_name}..."):
                            model_path = os.path.join('models', model_name)
                            try:
                                # Load weights (using map_location for cpu/gpu safety)
                                base_model.load_state_dict(torch.load(model_path, map_location=device))
                                translation = translate_sentence(
                                    input_sentence, 
                                    base_model, 
                                    metadata['en_vocab'], 
                                    metadata['de_vocab'], 
                                    metadata['de_itos'], 
                                    device,
                                    max_len=metadata['max_length']
                                )
                                
                                st.markdown(f"""
                                <div class="result-card">
                                    <div class="model-name">{model_name}</div>
                                    <div class="translation-text">{translation}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error loading {model_name}: {e}")

            # --- GOOGLE TRANSLATE ---
            with cols[1]:
                st.subheader("🌍 Industry Reference (Google)")
                with st.spinner("Fetching Google Translation..."):
                    try:
                        google_translation = GoogleTranslator(source='en', target='de').translate(input_sentence)
                        st.markdown(f"""
                        <div class="result-card" style="border-color: #4285F4;">
                            <div class="model-name" style="color: #4285F4;">Google Translate</div>
                            <div class="translation-text">{google_translation}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Google Translate Error: {e}")

else:
    st.info("Waiting for configuration... Ensure 'essential/transformer_metadata.pkl' exists.")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Built with ❤️ for Machine Translation Research</p>", unsafe_allow_html=True)
