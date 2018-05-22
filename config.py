# folder config. Please take care that each path string ends with a /
folder_dblp_xml = './data/'
folder_content_xml = './data/content_xml/'
folder_pdf = './data/pdf/'
folder_log = './data/logs/'
folder_datasets = './data/datasets/'
folder_classifiers = './data/classifiers/'
folder_pickle = './data/pickle/'
folder_clusters = './data/clusters/'

# mongoDB
mongoDB_IP = '127.0.0.1'
mongoDB_Port = 27017  # default local port. change this if you use SSH tunneling on your machine (likely 4321 or 27017).
# mongoDB_db = 'pub'
mongoDB_db = 'TU_Delft_Library'

# pdf extraction
grobid_url = 'http://127.0.0.1:8080'

# conferences we like
# book_titles = ['JCDL','SIGIR','ECDL','TPDL','TREC', 'ICWSM', 'ESWC', 'ICSR','WWW', 'ICSE', 'HRI', 'VLDB', 'ICRA', 'ICARCV']
booktitles = ['test_no_conf']

# root to the project
ROOTPATH = '/data2/SmartPub-TSENER'
# ROOTPATH = 'C:/Users/mvall/PycharmProjects/SmartPub-TSENER'
STANFORD_NER_PATH = '/data2/SmartPub-TSENER/stanford_files/stanford-ner.jar'
# STANFORD_NER_PATH = 'C:/Users/mvall/PycharmProjects/SmartPub-TSENER/stanford_files/stanford-ner.jar'

# journals we like
# journals = ['IEEE Trans. Robotics' , 'IEEE Trans. Robotics and Automation', 'IEEE J. Robotics and Automation']

journals = ['I. J. Robotics and Automation', 'IEEE J. Biomedical and Health Informatics',
            'Journal of Intelligent and Robotic Systems']  # ieee and Springer

source = 'data/pdf/'
source_xml = 'data/xml/'

# Update process
overwriteDBLP_XML = False
updateNow = True
checkDaily = False
checkWeekly = False

# Only pdf download
only_pdf_download = False

# Only text extraction
only_text_extraction = False

# Only classify and name entity extraction
only_classify_nee = False

####################### XML processing configurations #######################

# set to true if you want to persist to a local mongo DB (default connection)
storeToMongo = True

# set to true if you want to skip downloading EE entries (pdf URLs) which have been accessed before (either
# successfully or unsuccessfully) this only works if storeToMongo is set to True because the MongoDB must be accessed
# for that. (if you set storeToMongo to false, I will just assume that MongoDB is simply not active / there
skipPreviouslyAccessedURLs = True

# the categories you are interested in
CATEGORIES = {'article', 'inproceedings', 'proceedings', 'book', 'incollection', 'phdthesis', 'mastersthesis', 'www'}

# the categories you are NOT interested in
SKIP_CATEGORIES = {'phdthesis', 'mastersthesis', 'www', 'proceedings'}

# the fields which should be in your each data item / mongo entry
DATA_ITEMS = ["title", "booktitle", "year", "journal", "crossref", "ee", "license"]

statusEveryXdownloads = 100
statusEveryXxmlLoops = 1000

###############################################################################
