BUCKET_NAME = 'bc5cdr-dataset'
TRAIN_DATA = 'train_bc5cdr.parquet'
TEST_DATA = 'test_bc5cdr.parquet'
VALID_DATA = 'valid_bc5cdr.parquet'

label2id = {
    "O": 0,
    "B-Chemical": 1,
    "B-Disease": 2,
    "I-Disease": 3,
    "I-Chemical": 4
}