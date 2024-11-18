# from tensorflow.keras.models import load_model
#
# model = load_model('facenet_keras.h5')



import h5py
with h5py.File('facenet_keras.h5', 'r') as f:
    print(list(f.keys()))
    weights_keys = list(f['model_weights'].keys())
    print(weights_keys)
