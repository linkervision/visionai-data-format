
# Convert VAI (VisionAI) format from/into BDD+,COCO format

## BDD+

### VAI to BDD+

i.e :
`python vai_to_bdd.py -vai_src_folder ./test_data/vai_from_bdd -bdd_dest_file ./test_data/bdd_converted_2.json -company_code 101 -sequence_name test -storage_name storage_test -container_name container_test`

Arguments :
- `-vai_src_folder` : folder contains VAI format json file
- `-bdd_dest_file`  : BDD+ format file save destination
- `-company_code`  : company code
- `-sequence_name`  : sequence name
- `-storage_name`  : storage name
- `-container_name`  : container name


### BDD+ to VAI

i.e :
`python bdd_to_vai.py -bdd_src_file ./test_data/bdd_converted.json -vai_dest_folder ./test_data/vai_from_bdd`

Arguments :
- `-bdd_src_file`  : path of file with BDD+ format
- `-vai_dest_folder` : folder destination to saves VAI format files
- `-sensor` : name of current sensor , optional, default : `camera1`

## COCO

### VAI to COCO

i.e :
`python vai_to_coco.py -s ./test_data/vai_data/ -d ./test_data/coco_data/ -oc "class1,class2,class3" `

Arguments :
- `-s` : Folder contains VAI format json file
- `-d`  : COCO format save destination
- `-oc`  : Labels (or categories) of the training data

### COCO to VAI

i.e :
`python coco_to_vai.py -s ./test_data/vai_data/ -d ./test_data/vai_data/ --sensor camera1 `

Arguments :
- `-s` : Path of COCO dataset containing 'data' and 'annotations' subfolder
- `-d`  : VAI format save destination
- `--sensor` : Sensor name ( i.e : `camera1`)
