| prior                  | stage             |   ML (s/n) |   posterior mean |   MAP (mode) |   P(theta > 0.8) | meets rule   |
|:-----------------------|:------------------|-----------:|-----------------:|-------------:|-----------------:|:-------------|
| Beta(2,3) [assignment] | after batch 1     |     0.6000 |           0.5333 |       0.5385 |           0.0116 | False        |
| Beta(2,3) [assignment] | after all batches |     0.8500 |           0.8286 |       0.8350 |           0.7883 | False        |
| Beta(1,1) [uniform]    | after batch 1     |     0.6000 |           0.5833 |       0.6000 |           0.0504 | False        |
| Beta(1,1) [uniform]    | after all batches |     0.8500 |           0.8431 |       0.8500 |           0.8811 | False        |
| Beta(8,2) [optimistic] | after batch 1     |     0.6000 |           0.7000 |       0.7222 |           0.1631 | False        |
| Beta(8,2) [optimistic] | after all batches |     0.8500 |           0.8455 |       0.8519 |           0.9009 | False        |