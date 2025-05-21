import streamlit as st
import torch
from gliner import GLiNER
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="GLiNER Entity Recognition",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .entity-highlight {
        padding: 2px 4px;
        border-radius: 4px;
        margin: 0 2px;
    }
    .entity-label {
        border: 1px solid #333;
        border-radius: 4px;
        padding: 1px 3px;
        background: white;
        color: black;
        font-size: 0.7em;
        margin-left: 4px;
    }
    .results-container {
        line-height: 1.8;
        font-family: sans-serif;
        padding: 15px;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🔍 GLiNER Entity Recognition App")
st.markdown("Upload text and extract named entities using the GLiNER model.")

# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")
    model_name = st.selectbox(
        "Select GLiNER Model",
        ["qtung24/gliner-biomed-bi-finetune", "other-model-option"]
    )
    
    st.subheader("Default Entity Labels")
    default_labels = ['drug group', 'drug generic name', 'drug brand name', 'chemical or toxins or pesticides']
    
    st.subheader("Custom Entity Labels")
    custom_label_input = st.text_area(
        "Enter custom labels (one per line) or use defaults",
        "\n".join(default_labels)
    )
    
    # Color selection
    st.subheader("Entity Colors")
    color_options = {
        "drug group": "#4dd0e1",
        "drug generic name": "#26a69a", 
        "drug brand name": "#fb8c00",
        "chemical or toxins or pesticides": "#8bc34a"
    }
    
    # Let user customize colors
    st.write("Customize entity colors:")
    for label in default_labels:
        color_options[label] = st.color_picker(f"Color for {label}", color_options.get(label, "#90caf9"))

# Main content area
tab1, tab2 = st.tabs(["Text Input", "Results"])

# Text input tab
with tab1:
    # Text input area
    text_input = st.text_area(
        "Enter text for entity recognition",
        height=300,
        placeholder="Paste your text here... For example: 'The patient was prescribed amoxicillin 500mg daily along with acetaminophen for pain relief.'"
    )
    
    # Process button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        process_button = st.button("Process Text")

# Results tab
with tab2:
    # Initialize model on first run
    @st.cache_resource
    def load_model(model_name):
        try:
            with st.spinner('Loading GLiNER model...'):
                model = GLiNER.from_pretrained(model_name)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    if process_button and text_input:
        # Load the model
        model = load_model(model_name)
        
        if model:
            # Get labels from textarea
            labels = [label.strip() for label in custom_label_input.split('\n') if label.strip()]
            
            try:
                # Predict entities
                with st.spinner('Processing text...'):
                    entities = model.predict_entities(text_input, labels)
                
                # Sort entities by start position
                entities.sort(key=lambda x: x['start'])
                
                # Display entities as a table
                if entities:
                    st.subheader("Extracted Entities")
                    
                    # Create a dataframe for the table view
                    entities_df = pd.DataFrame([
                        {
                            "Entity": text_input[ent["start"]:ent["end"]],
                            "Label": ent["label"],
                            "Start": ent["start"],
                            "End": ent["end"],
                            "Score": round(ent.get("score", 0), 3)
                        }
                        for ent in entities
                    ])
                    
                    st.dataframe(entities_df, use_container_width=True)
                    
                    # Generate highlighted text
                    highlighted = ""
                    last_idx = 0
                    
                    for ent in entities:
                        start, end = ent["start"], ent["end"]
                        label = ent["label"]
                        color = color_options.get(label, "#90caf9")
                        
                        # Add plain text before entity
                        highlighted += text_input[last_idx:start]
                        
                        # Add highlighted span
                        highlighted += f'''
                        <span class="entity-highlight" style="background-color:{color}">
                            {text_input[start:end]}
                            <span class="entity-label">{label}</span>
                        </span>
                        '''
                        
                        last_idx = end
                    
                    # Add remaining text
                    highlighted += text_input[last_idx:]
                    
                    # Display highlighted text
                    st.subheader("Highlighted Text")
                    st.markdown(f'<div class="results-container">{highlighted}</div>', unsafe_allow_html=True)
                else:
                    st.info("No entities were detected in the provided text.")
            except Exception as e:
                st.error(f"Error processing text: {e}")
    else:
        if process_button and not text_input:
            st.warning("Please enter some text to process.")
        else:
            st.info("Enter your text in the 'Text Input' tab and click 'Process Text' to see results here.")

# Add more information at the bottom
st.markdown("---")
st.markdown("""
### About GLiNER
GLiNER is a generative language model fine-tuned for named entity recognition tasks.
This application uses the model to identify and extract entities from text based on the labels you provide.

**Note:** Processing may take some time depending on the length of the text and the complexity of the entities.
""")

