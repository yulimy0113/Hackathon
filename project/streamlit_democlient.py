import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from PIL import Image


def get_lables(data, col):
    """
    Function to return a callable dictionary with index
    when gathering unique values from a selected column in a given data.
    
    data: df.DataFrame()
    col: the name of a column you select.
    """
    labels = sorted(list(data[col].unique()))
    label_dict = {i : key for i, key in enumerate(labels)}
    return label_dict

def load_image(file):
    """
    Function to show the uploaded picture and to return data fitting the models.
    file: np.array, input feature of the models.
    """
    img = Image.open(file)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    st.image(file)
    return img_array

def get_coupon(data, mood, place):
    """
    Function to select and return the most suitable coupon
    based on the location and the mood predicted from the picture.
    
    data: pd.DataFrame
    mood: str, predicted by a mood model
    place: str, predicted by a location model
    """
    pick_coupon = data[data['mood'] == mood][data['place']==place].reset_index(drop=True)
    name = pick_coupon.loc[0, 'name']
    coupon = pick_coupon.loc[0, 'coupon']
    st.markdown(name)
    st.markdown(f'Here is a special coupon only for you: {coupon}')


def main():
    """
    Main function.
    """
    st.title("Always with US.")
    st.subheader("How was your journey today?")
    st.subheader("Please share your day with us!")

    # load the coupon list and trained models with label lists
    coupon_list = pd.read_csv('coupon_list.csv')
    mood_model = keras.models.load_model('mood_classifier_dw.h5')
    mood_label = get_lables(coupon_list, 'mood')
    place_model = keras.models.load_model('MobileNet_layer18+lr1e-05+epochs30/')
    place_label = get_lables(coupon_list, 'place')

    # Upload picture
    pic = st.file_uploader('Upload picture', type=['png', 'jpg', 'jpeg'])
    if pic:
        img = load_image(pic)

        mood_i = mood_model.predict(img)
        mood_i = np.argmax(mood_i)
        mood = mood_label[mood_i]

        place_i = place_model.predict(img)
        place_i = np.argmax(place_i)        
        place = place_label[place_i]

        get_coupon(coupon_list, mood, place)


# Let's run it!
if __name__ == '__main__':
    main()
