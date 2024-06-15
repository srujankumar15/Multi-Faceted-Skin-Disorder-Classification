
import base64
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import random


labels = ["Acne and Rosacea", "Actinic Keratosis Basal Cell Carcinoma", "Atopic Dermatitis", "Eczema", "Nail Fungus", "Psoriasis pictures Lichen Planus","Monkeypox"]

@st.cache_data
def load_model():
    model = tf.keras.models.load_model('skin_model2.h5') 
    return model

model = load_model()

def predict(image):
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (224, 224))
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    return prediction

def main():
    col1, col2, col3 = st.columns([1.8, 2, 1])
    with col2:
        st.image("rash.png", width=140)
    st.write("<h2 style='text-align:center; text-shadow:-2px 0 black, 0 1px black,1px 0 black, 0 -1px black;color:yellow'>Multi-Faceted Skin Disorder Classification</h2>",unsafe_allow_html=True)
    st.write("Upload an image of a skin to detect the disease.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict!"):
            prediction = predict(image)
            predicted_class_index = np.argmax(prediction)
            predicted_class = labels[predicted_class_index]
            
            if predicted_class == 'Acne and Rosacea':
                remedies = 'For more serious rosacea with bumps and pimples, you may be prescribed an oral antibiotic pill such as doxycycline (Oracea, others). Acne medicine taken by mouth. For severe rosacea that dont respond to other medicine, you may be prescribed isotretinoin (Amnesteem, Claravis, others)'
            elif predicted_class == 'Actinic Keratosis Basal Cell Carcinoma':
                remedies = 'Basal cell carcinoma is most often treated with surgery to remove all of the cancer and some of the healthy tissue around it. Options might include: Surgical excision. In this procedure, your doctor cuts out the cancerous lesion and a surrounding margin of healthy skin.'
            elif predicted_class == 'Atopic Dermatitis':
                remedies = 'Use of topical corticosteroids is the first-line treatment for atopic dermatitis flare-ups. Pimecrolimus and tacrolimus are topical calcineurin inhibitors that can be used in conjunction with topical corticosteroids as first-line treatment.'
            elif predicted_class == 'Eczema':
                remedies = 'An effective, intensive treatment for severe eczema involves applying a corticosteroid ointment and sealing in the medication with a wrap of wet gauze topped with a layer of dry gauze.'
            elif predicted_class == 'Nail Fungus':
                remedies = 'Antifungal pills also work more quickly than medicine applied to the nails. Taking antifungal pills for two months can cure an infection under the fingernails. Usually three months of treatment cures a toenail fungal infection.'
            elif predicted_class == 'Psoriasis pictures Lichen Planus':
                remedies = 'There isnt a cure for psoriasis or lichen planus, but there are treatments to reduce discomfort for both. Psoriasis outbreaks can be treated with topical ointments, light therapy, and even systemic medications. Because psoriasis is a chronic condition, you will always be susceptible to outbreaks.'
            elif predicted_class == 'Monkeypox':
                remedies = 'If you have Monkeypox, isolate at home in a separate room from family and pets until your rash and scabs heal. There is no specific treatment approved for mpox. Health care professionals may treat mpox with some antiviral drugs used to treat smallpox, such as tecovirimat (TPOXX) or brincidofovir (Tembexa).'
            st.write("Prediction :", predicted_class)
            st.write("Remedies :")
            st.write(remedies)


if __name__ == "__main__":
    main()
