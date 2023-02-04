
# Convert VAI (VisionAI) format from/into BDD+,COCO format

## BDD+

### VAI to BDD+

i.e :
`python vai_to_bdd.py --source ./test_data/vai_from_bdd --dst ./test_data/ --company-code 101 --storage-name storage_test --container-name container_test`

Arguments :
- `--src` : folder contains VAI format json file
- `--dst`  : BDD+ format file save destination
- `--company-code`  : company code
- `--storage-name`  : storage name
- `--container-name`  : container name


### BDD+ to VAI

i.e :
`python bdd_to_vai.py --src ./test_data/ --dst ./test_data/vai_from_bdd`

Arguments :
- `--src`  : path of file with BDD+ format
- `--dst` : folder destination to saves VAI format files
- `--sensor` : name of current sensor , optional, default : `camera1`

## COCO

### VAI to COCO

i.e :
`python vai_to_coco.py -s ./test_data/vai_data/ -d ./test_data/coco_data/ -oc "class1,class2,class3" --copy-image`

Arguments :
- `-s` : Folder contains VAI format json file
- `-d`  : COCO format save destination
- `-oc`  : Labels (or categories) of the training data
- `--copy-image` : Optional, enable to copy image

### COCO to VAI

i.e :
`python coco_to_vai.py -s ./test_data/vai_data/ -d ./test_data/vai_data/ --sensor camera1 --copy-image`

Arguments :
- `-s` : Path of COCO dataset containing 'data' and 'annotations' subfolder
- `-d`  : VAI format save destination
- `--sensor` : Sensor name ( i.e : `camera1`)
- `--copy-image` : Optional, enable to copy image
