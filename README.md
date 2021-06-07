# Autoencoder-midi-generator
Uses deep learning to generate midi musics. Trained on about 3000 piano midi files. <br />
Some samples are under generated_music. Musics generated using bidirectional LSTM are samples 6 and after. <br />
Really looking forward to use this on my visual novel project XD. <br />

**Experiments with bidirectional LSTM:**<br />
Somehow the bidirectional LSTM can produce more interesting results. <br />
I thought I made a terrible mistake by using the bidirectional layers since I do data augmentation by playing the music normally and backward, and using bidirectional layers may defeat this method. But the results are suprisingly good.<br />
I have 2 hypotheses:<br />
1. The sequence is too long (about 1500 ticks) for the network to realize some of them are the reverse of the other. In this case, the model is very likely to behave similarly comparing to models without bidirectional layers.<br />
2. If the network is able to tell which 2 songs are the reverse of each other, it wll have to remember some long term features, which is benefitial when generating musics. <br />
I find the second one more convincing, because the loss changes very differently comparing to all previous models, and I observe some structures in the generated muscis, which are not found in previous models without bidirectional layers.<br />
