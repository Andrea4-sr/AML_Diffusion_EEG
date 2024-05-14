path = '/Users/andreasantos/Projects/Coding/AML_EEG/data'

channels_of_interest = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]  # correspond to main brain regions

# do we train a diffusion models for all channels - check edf files first and check which channels have seizures

for i in tqdm.tqdm([x for x in os.listdir(path) if x.endswith('.edf')]):
    f = pyedflib.EdfReader(path + i)
    raw_ = np.empty((0, f.getNSamples()[0]))

    j = [x for x in f.getSignalLabels() if x.contains('F3') or x.contains('FZ') or x.contains('F4') or x.contains('C3') or x.contains('CZ') or x.contains('C4') or x.contains('P3') or x.contains('PZ') or x.contains('P4')]
    j = j[0:9]
    a

