# Data

`KaggleV2-May-2016.csv` is the "Medical Appointment No Shows" dataset by Joni Hoppen, Kaggle id `joniarroba/noshowappointments`.

- 110,527 appointments, 14 columns, public clinics in Vitória, Brazil
- appointment dates 29 April to 8 June 2016 (27 days), 20.2% no-shows
- columns: PatientId, AppointmentID, Gender, ScheduledDay, AppointmentDay, Age, Neighbourhood, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received, No-show (`Yes` means the patient missed it)

## Why it is not in git

The file ships in `ml_lab.zip` but `.gitignore` keeps it out of the repo. We could not confirm the dataset's licence, so we don't republish it. Check the licence on the Kaggle page before making the repo public.

## Getting it on another machine

Download it from https://www.kaggle.com/datasets/joniarroba/noshowappointments (free Kaggle login), unzip, and put `KaggleV2-May-2016.csv` in this folder.

## Check it is the same file

```
shasum -a 256 topic7/data/KaggleV2-May-2016.csv
```

Our copy prints `9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d`. It came from a public GitHub mirror of the Kaggle file (`mriffaud/Medical-Appointment-No-Shows`); its rows and columns match the Kaggle description. If a fresh Kaggle download gives a different hash, rerun the prototypes and compare.
