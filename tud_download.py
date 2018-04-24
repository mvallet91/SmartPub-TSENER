import shutil, os, pickle, time, requests
from sickle import Sickle

working_dir = 'data/tudelft_repo/test/'
filter_types = ['master thesis', 'conference paper', 'doctoral thesis', 'journal article']
MAX_SIZE = 100  # In MB

types = []
items = {}
failed = []
downloaded = {}
x = 0
large = []

# Get list of OAI records ##############################

update = True

if update:
    sickle = Sickle('http://oai.tudelft.nl/ir')
    records = sickle.ListRecords(**{'metadataPrefix': 'oai_dc', 'ignore_deleted': 'True'})
    for r in records:
        uuid = ''
        uuid = r.metadata['identifier'][0][32:]
        items[uuid] = r.metadata
else:
    with open('tud_metadata.pickle', 'rb') as handle:
        items = pickle.load(handle)

print('items', len(items), '- types', len(types))

for dirpath, dirs, files in os.walk(working_dir):
    for file in files:
        path = os.path.join(dirpath, file)
        uuid = path.split('_')[-1].split('.')[0]
        downloaded[uuid] = items[uuid]

# Download files ##############################

for uuid in items.keys():
    download = 'https://repository.tudelft.nl/islandora/object/uuid:' + uuid + '/datastream/OBJ/download'
    name = working_dir + 'TUD_' + uuid + '.pdf'
    if os.path.exists(name):
        print('File exists:', name)
        print('.', end='')
        continue
    r = requests.get(download, stream=True)
    if r.status_code == 200 and items[uuid]['type'][0] in filter_types:
        with open(name, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
        print('Downloaded:', name)
        x += 1
    if r.status_code != 200:
        print('Download failed, status', r.status_code, 'link', download)
        failed.append(uuid)
    time.sleep(2)
    if x == 20:
        break

# Delete large files ##############################

target_size = MAX_SIZE * 1048514

for dirpath, dirs, files in os.walk(working_dir):
    for file in files:
        path = os.path.join(dirpath, file)
        if os.stat(path).st_size > target_size:
            os.remove(path)
            large.append(file)

print('Deleted large files:', large)
