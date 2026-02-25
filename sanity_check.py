from tensorflow.keras.models import load_model
m= load_model("models/baseline_cnn_model.h5")
print(m.input_shape)
c= load_model("models/resnet50_final.h5")
print(c.input_shape)
d= load_model("models/effnetb0_final.h5")
print(d.input_shape)
