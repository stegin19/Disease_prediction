import streamlit as st
import tensorflow as tf
import time
import numpy as np
#import wikipedia
from google_images_download import google_images_download



print()


def model_prediction(test_image):
    model = tf.keras.models.load_model("skin_disease_p2.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Expand dimensions to create a batch
    predictions = model.predict(image)
    return np.argmax(predictions)

# def get_wikipedia_summary(disease_name):
#     try:
#         return wikipedia.summary(disease_name, sentences=5)
#     except wikipedia.exceptions.DisambiguationError as e:
#         # If there are multiple pages with similar names, you can handle it here
#         return f"Multiple pages found for '{disease_name}'. Please specify."



#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])


if(app_mode=="Home"):
    st.header("DERMATOLOGICAL DISEASE DETECTION WITH ML-BASED IMAGE ANALYSIS ")
    # image_path = "home_page.jpeg"
    # st.image(image_path,use_column_width=True)
    st.markdown("""
                Dermatological diseases represent a significant portion of global health issues, affecting millions of individuals worldwide.
                 Early and accurate diagnosis is crucial for effective treatment and prevention of complications. 
                This study explores the application of machine learning techniques for the prediction of dermatology diseases based on
                 clinical and image data. The results indicate the potential of machine learning in dermatology disease prediction, 
                with certain models demonstrating high accuracy in identifying specific skin conditions. The developed models can 
                aid healthcare professionals in making more informed decisions, leading to timely and accurate diagnosis.
                 The ultimate goal of disease prediction is to empower healthcare professionals and individuals with the knowledge 
                and tools necessary to take preventive measures, make informed decisions, and ultimately improve overall health and well-being.
                 A key component of our research involves the implementation of Convolutional Neural Networks (CNNs) to enhance the analysis of 
                dermatological images. CNNs have demonstrated remarkable success in image classification tasks, particularly in medical imaging.
                 In our study, CNNs are utilized to automatically learn hierarchical representations of features within dermatological images, 
                capturing intricate patterns and textures that may be indicative of specific skin conditions.
 
    """)

elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                The dataset comprises images depicting various types of skin infections caused by bacteria, fungi, parasites, and viruses.
                 Bacterial infections can range from cellulitis to impetigo, while fungal infections include athlete's foot and ringworm.
                 Parasitic infections such as scabies and cutaneous larva migrans are also represented. Additionally, viral infections like shingles and chickenpox are included.
                 Each class in the dataset corresponds to a specific type of skin infection, totaling 8 classes in all.
               
                **this dataset contains 8 classes ,They are**

                - Bacterial Infections- cellulitis
                - Bacterial Infections- impetigo
                - Fungal Infections - athlete -foot
                - Fungal Infections - nail-fungus
                - Fungal Infections - ringworm
                - Parasitic Infections - cutaneous-larva-migrans
                - Viral skin infections - chickenpox
                - Viral skin infections - shingles

                link for the [Dataset](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset)

                """)    


elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=2,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        with st.spinner('Please wait'):
            result_index = model_prediction(test_image)
            class_name = ['Cellulitis',
                    'Impetigo','Athlete foot','Nail-fungus','Ringworm',
                    'Cutaneous larva migrans','Chickenpox','Shingles']
            
                
            predicted_disease = class_name[result_index]
            st.success(predicted_disease)
            # try:
            #     summary = get_wikipedia_summary(predicted_disease)
            #     st.markdown(f"### Summary for {predicted_disease}")
            #     st.success(summary)
            # except wikipedia.exceptions.PageError:
            #     st.error(f"No Wikipedia page found for {predicted_disease}.")
 
