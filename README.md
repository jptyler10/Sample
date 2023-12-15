PheDX ML Algorithm
============

Here, we give instructions for how to use the genetic testing model that was developed using the pediatric notes.  The model should takes as input the notes from the pediatric cohort and outputs a model trained to find patients in need of a genetic test.  The model can then be run on a subset of "new" notes to see if we can identify any patients that may benefit from a test.

The input file must be formatted as follows.  It should be a csv file called 'notes.csv' in the main directory with at least the following columns:

  * mrn
  * note_dttm - the datetime that the note was taken
  * note_text - the note text

Note that you can use the notes.csv file that was provided directly by Sinai to run this.  I have saved the ages at first mention of a genetic test/referral to process with the notes file provided.

Process the notes
------------

Then, run the following command:

    python scripts/process_notes.py


The output from this command should be a few new files:

1. ml_notes_model.csv - the notes that are used for the model training.  This includes the case notes as well as a random sample of control notes from 10 to 1 sampling of control to case.
2. phedx_dict.json - this is the json file that stores the target for the machine learning model.
3. id_map.csv - this file stores the mapping between the original mrn and the new id.  the new id has to be numeric so we map all mrns to a new id.

Run the Doc2Vec Model
------------

Next, the doc2vec model needs to be trained.  This can be done by running the following command first:

    python scripts/doc2vec_new_data.py --phase train

Once that has completed, the doc2vec results need to be inferred for all notes.  This can be done by running the command:

    python scripts/doc2vec_new_data.py --phase infer

Run the Main Model
------------

Then the model must be trained.  This can be done running the following commands:

    python scripts/main.py --phase train --epochs 20
    python scripts/main.py --phase test --resume models/best.ckpt 

Feel free to adjust the number of epochs as needed.

Run the model on a new set of notes
------------

If you would like to run the model on a new set of notes to get out a ranking of patients that might need a genetic test from most likely to least, follow the steps below.  

First, there needs to be a file in the main directory called 'new_notes.csv'.  This file should have the following columns:
1. mrn - a **numeric** value unique to each patient, not the character value used by Sinai
2. age_days - a number indicating the day the note was written relative to the patient's date of birth
3. note_text - the note_text

Then, run the following command 

    python scripts/doc2vec_new_data_new.py
    python scripts/main_new.py

Running these two commands should output a file called new_notes_results.csv listing the patient ID and the probability.

Please reach out to jptyler10@gmail.com if you have any questions or comments.  


