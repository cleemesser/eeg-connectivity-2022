# README


code to accompany the manuscript Baumer et al, 2022

This code describes the custom computations used to implement wPLI over time on specific segments
of EEG describes by epochs as defined the [mne-python package](https://mne.tools/stable/index.html).

This is extracted from a larger and messier private library. The emphasis here is on clarity, correctness and good documentation.

Approach taken is to use the "Filter-Hilbert" method of calculating wPLI as described in Michael X Cohen's book Analyzing Neural Time Series Data (2014).

The wPLI definition is taken from Vinck et al. 2011 NeurImage paper which original defined it.
(Of note, the edition we have of Cohen's book leaves out an absoluate value sign from equation 26.

)

### References
 @article{Gramfort_2013, title={MEG and EEG data analysis with MNE-Python}, volume={7}, ISSN={1662453X}, url={http://journal.frontiersin.org/article/10.3389/fnins.2013.00267/abstract}, DOI={10.3389/fnins.2013.00267}, journal={Frontiers in Neuroscience}, author={Gramfort, Alexandre}, year={2013} }

