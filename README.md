TSE-NER

The main goal of TSE-NER is to generate training data for long-tail entities and train a NER tagger, 
label such entities in text, and use them for document search and exploration.

Please refer to the paper for more info: 

This project can be approached in two main ways: as developer or user

* For developers, we try to provide all the code required, however, we take advantage of some great 
resources such as Gensim, Stanford NLP, and GROBID, so it may require some effort.

* For users, this first implementation will work only through an API (that we try to make as friendly 
as possible), so it still requires a bit of technical understanding.

Following the main goal described before, SmartPub-TSENER is divided in 3 main modules: 

The main goal of TSE-NER is to (1) Generate training data for long-tail entities and train a NER tagger, 
(2) label entities in text (documents), and (3) use named entities in documents for search and exploration.

Therefore in this repository we provide the code used for each one of the 3 main modules, 
as well as our approach for data collection and preparation:

### Data Collection, Extraction and Preparation
Our corpus consists of scientific publications, mainly from Computer Science, but also from the Biomedical
domain (PubMed Central) and master theses from TU Delft. 
The collection and extraction steps are source-dependent, for example:
* PubMed Central: We take advantage of the Open Subset of publications, available using OAI-PMH and ftp. 
These publications have metadata and full-text in XML format, and we use the PubmMed Parser 
(by Titipat Achakulvisut and Daniel E. Acuna (2015) "Pubmed Parser" http://github.com/titipata/pubmed_parser.)
to extract and store the information in MongoDB.
* TU Delft Master Theses: The collection is similar to PMC, we use OAI-PMH to get the metadata and download 
 links for the pdf of student's mather theses (with permission from the library, of course!), however, the
 actual content has to be extracted using GROBID (Grobid (2008-2017) https://github.com/kermitt2/grobid),
 which not always guarantees the best performance since theses are from different faculties and follow 
 a wide variety of formats.

The important part is that we need the full text in MongoDB, so we can index all the content in Elasticsearch.
This allows for the quick queries required for the processing in all modules.

In addition, we need to prepare data and train word2vec and doc2vec models used in the expansion and 
filtering steps.


### Module 1: NER Training
This first module provides with the environment for anyone interested to train a NER model (Stanford NER) 
for the labelling of long-tail entities, for example datasets used in scientific publications: 
“...in this work, we benchmark our algorithm using the *ImageNet* corpus...”.

### Module 2: NER Labelling
Once a model is trained, it can be used to label certain types of long-tail entities in text.
By selecting a model, and introducing a piece of text, the system will return a list of entities found.

### Module 3: NER Search and Navigation System
This is a basic approach at an interface for a collection of documents, it can be simply a metadata repository
with links to the actual content, allowing for a richer navigation than current systems.
