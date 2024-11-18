import h5py

# Open the .h5 file
with h5py.File('facenet_keras.h5', 'r') as f:
    # List all file attributes (may include version info)
    print("File attributes:")
    for key, value in f.attrs.items():
        print(f"{key}: {value}")