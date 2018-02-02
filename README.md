# AradoFaceNet 

This is the face recognizer to identify awesome Arado people. 
The work is done top of the wonderful [FaceNet GitHub project](https://github.com/davidsandberg/facenet). 

## Train&Test

- Download [pre-trained model](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit)
- Create a models folder in the root of the repository and copy the trained model there
- Copy the images into the raw folder, create a folder for each object/person you want the model to recognise and put the corresponding pictures there.
- Run ‘align_arado_data.sh’ to align pictures and copy them into arado160
- Run ‘python src/align/create_test_train_sets.py’ to create test and train data.
- To train run: `train_arado.sh`
- To test run: `test_arado.sh`

## Awesome results

These are the results we got out of our trained model.

   0  anssi: 0.397 arado/arado_160_test/anssi/anssi.png
   1  henrik: 0.404 arado/arado_160_test/henrik/arado-2.png
   2  jarno: 0.565 arado/arado_160_test/jarno/pic1.png
   3  jarno: 0.596 arado/arado_160_test/jarno/pic10.png
   4  jarno: 0.293 arado/arado_160_test/jarno/pic11.png
   5  jarno: 0.500 arado/arado_160_test/jarno/pic2.png
   6  markus: 0.383 arado/arado_160_test/markus/markus 2.png
   7  mika: 0.448 arado/arado_160_test/mika/arado-2.png
   8  mikko: 0.545 arado/arado_160_test/mikko/arado-2.png
   9  stefano: 0.361 arado/arado_160_test/stefano/arado-2.png
  10  teppo: 0.765 arado/arado_160_test/teppo/teppo.png
  11  timo: 0.422 arado/arado_160_test/timo/arado-2.png
  12  ville: 0.263 arado/arado_160_test/ville/arado.png
Accuracy: 1.000