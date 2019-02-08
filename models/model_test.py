import keras_mobilenetv1
model = keras_mobilenetv1.ssd_mn_300((300, 300, 3), 6)
model.summary()
model.save('./mobilenetv1ssd.h5')