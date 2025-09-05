from tensorflow.keras.preprocessing import image

def predict_malaria(img_path):
    img = image.load_img(img_path, target_size=(64,64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    return "Parasitized" if prediction > 0.5 else "Uninfected"

print(predict_malaria("cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png"))
