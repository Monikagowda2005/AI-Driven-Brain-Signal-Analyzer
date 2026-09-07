import mne

raw = mne.io.read_raw_edf(
    r"C:\Users\monik\Downloads\Brain_Signal_Analyzer\dataset\files\S001\S001R03.edf",
    preload=False
)

print(raw.annotations)

events, event_id = mne.events_from_annotations(raw)

print("\nEvent IDs:")
print(event_id)

print("\nFirst 10 events:")
print(events[:10])