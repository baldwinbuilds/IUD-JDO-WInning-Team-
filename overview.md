Inter-uni Datathon Stream 3: Gas Sensor Array Drift Dataset
Goal is to win.

Chemical sensor arrays can identify gases by detecting patterns in their electrical responses. However, sensor behaviour may change over time because of ageing, repeated exposure, contamination, and environmental conditions. This phenomenon is known as sensor drift. It can cause a model trained on earlier measurements to become less reliable when deployed later. In this competition, you will classify sensor-array measurements into one of six gas categories while accounting for changes in sensor behaviour over time. Each observation contains: A chronological batch number The gas concentration used during measurement 128 numerical features derived from 16 chemical sensors Eight response descriptors for each sensor The competition data contain: 10,310 labelled measurements from batches 1–9 3,600 hidden measurements from batch 10 Six target classes 600 hidden test observations from each class The target, gas_class, is encoded as an integer from 1 to 6. The hidden test data come entirely from a later sensor batch. This measures whether a model can generalise under sensor drift rather than simply recognise patterns from randomly mixed measurements. Participants are encouraged to use batch-aware validation and investigate feature scaling, concentration effects, class imbalance, distribution shift, and drift-robust modelling techniques. The source, class-name mapping, and full attribution for the data will be disclosed after the competition. Participants must not attempt to identify the original dataset, use external copies of the data, or recover hidden test labels.

The test file is indeed 2288 rows, theres no missing rows


Evaluation Submissions are evaluated using macro-averaged F1. The F1 score for class (k) is: F1_k
\frac{ 2 \times \mathrm{Precision}_k \times \mathrm{Recall}_k }{ \mathrm{Precision}_k+\mathrm{Recall}_k }
The final competition score is: F1_{\mathrm{macro}}
\frac{1}{6} \sum_{k=1}^{6}F1_k
where: Precisionk

measures how many predictions of class (k) are correct.
Recallk

measures how many true examples of class (k) are correctly identified.
Higher scores are better. Macro averaging gives every class equal importance, regardless of how frequently it appears in the training data.
Data dictionary

measurement_id: unique competition identifier.
batch: chronological source batch, from 1 to 10.
concentration: gas concentration used during the measurement.
feat_1 … feat_128: eight response descriptors for each of 16 chemical sensors.
gas_class: target label: 1 = Ethanol, 2 = Ethylene, 3 = Ammonia, 4 = Acetaldehyde, 5 = Acetone, 6 = Toluene.