import numpy as np
from tensorflow.keras.models import model_from_json

class_labels = ['high arm wave', 'horizontal arm wave', 'hammer', 'hand catch', 'forward punch', 'high throw',
                'draw x', 'draw tick', 'draw circle', 'hand clap', 'two hand wave', 'side-boxing', 'bend',
                'forward kick', 'side kick', 'jogging', 'tennis swing', 'tennis serve', 'golf swing', 'pick up & throw']

def load_model():
    with open("model/model.json", "r") as f:
        model = model_from_json(f.read())
    model.load_weights("model/model_weights.h5")
    return model

def read_sample(file):
    action = np.loadtxt('MSRAction3DSkeleton(20joints)/' + file)[:, :3].flatten()
    frame_size = len(action) // 60
    action = action.reshape(frame_size, 60)
    return np.array([action[1].reshape(5, 4, 3)])

classifier = load_model()
samples = ['a01_s01_e02_skeleton.txt', 'a02_s08_e02_skeleton.txt']

for fname in samples:
    sample = read_sample(fname)
    prediction = classifier.predict(sample)
    label = np.argmax(prediction)
    print(f"{fname}: {class_labels[label]}")