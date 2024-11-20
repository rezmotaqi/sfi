import pickle

with open('mohamad.pkl', 'rb') as f:
    content = pickle.load(f)
    print(type(content))
    print(content)
