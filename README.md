# CS7643_Final_Project

### Setup

Create env from updated yaml, and add a few separate installs:
 ```bash
    conda env create -f environment.yml
    conda activate TL_ERC
    pip install spacy==2.9.3
    python -m spacy download en_core_web_sm
    cd TL_ERC
    python setup.py
 ```

1. Download the pre-trained weights of HRED on [Cornell](https://drive.google.com/file/d/1OXtnyJ5nDMmK75L9kEQvKPIyO0xzyeVC/view?usp=sharing) dataset
2. Download [vocab data](http://nlp.stanford.edu/data/glove.840B.300d.zip) and extract data to TL-ERC\glove
3. IEMOCAP data already stored in datasets dir, including raw features with all modalities
4. Run `python iemocap_preprocess.py` to run pre-processing that now saves audio/visual data into train/valid/test folders along with text data
